import sys
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from models.unet import UNet

"""Identification script

Usage examples:
  # Show top-3 matches using avg cosine similarity and display the image
  ./venv311/bin/python identification.py custom_000161.jpg

  # Save image only, use ensemble_max method, and save metadata JSON
  ./venv311/bin/python identification.py custom_000161.jpg 3 --method ensemble_max --no-show --save-metadata

  # Run with Mahalanobis scoring and custom threshold
  ./venv311/bin/python identification.py custom_000161.jpg 5 --method mahalanobis --threshold 0.4

Notes:
- Methods:
  - avg_cosine: compare query to the per-image averaged (and normalized) embedding (fast/simple).
  - ensemble_max: compare query to all per-image variant embeddings and use the maximum similarity (robust to partial matches).
  - ensemble_mean: average similarity across variants.
  - mahalanobis: use per-image covariance to compute a Mahalanobis-based score (more conservative for uncertain variants).
  - weighted_ensemble: intelligently weigh augmentation variants based on the input mask type (e.g., for a lower-face mask, give more weight to the 'top_half' variant).

- Outputs:
  - A PNG image `result_<masked_file>.png` is always saved with a visualization of masked input, reconstruction, and top-k matches.
  - If `--save-metadata` is given, a JSON `result_<masked_file>.json` is saved with per-match scores and detection metadata.

- Interpreting scores: cosine scores range roughly -1..1; a practical threshold for recognition is dataset-dependent (0.5 is a reasonable starting point). Mahalanobis returns 1/(1+d) where d is distance; higher is better.

"""

# script entry is guarded by main(); importable for tests

# Variables set at runtime inside main(); leave placeholders for import-time
masked_file = None
file = None
full_path = None
masked_path = None

# -------------------------------
# Transforms
# -------------------------------
transform_unet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

transform_facenet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match_score(entry, query_emb, method='avg_cosine'):
    # ensure query_emb is normalized
    q = query_emb / np.linalg.norm(query_emb)
    if method == 'avg_cosine':
        # prefer precomputed normalized avg
        if 'embedding_avg_norm' in entry:
            db_emb = entry['embedding_avg_norm'].reshape(-1)
        else:
            db_emb = entry['embedding_avg'].reshape(-1)
            db_emb = db_emb / np.linalg.norm(db_emb)
        return float(np.dot(q, db_emb))

    elif method == 'ensemble_max':
        vals = []
        for v in entry.get('embeddings', {}).values():
            v = np.array(v).reshape(-1)
            vals.append(float(np.dot(q, v)))
        return float(np.max(vals)) if vals else 0.0

    elif method == 'ensemble_mean':
        stack = entry.get('embeddings_list')
        if stack is None or len(stack) == 0:
            return 0.0
        sims = stack.dot(q)
        return float(np.mean(sims))

    elif method == 'weighted_ensemble':
        # Define weights based on mask type. Higher weight for relevant parts.
        weights = {
            'full': 1.0, 'flip': 1.0, 'bright': 0.8, 'dark': 0.8,
            'contrast': 0.8, 'blur': 0.5, 'noise': 0.5, 'gray': 0.7,
            'rot_plus10': 0.6, 'rot_minus10': 0.6, 'rot_plus5': 0.7, 'rot_minus5': 0.7,
            'left_half': 1.0, 'right_half': 1.0, 'top_half': 1.0, 'bottom_half': 1.0
        }
        # This relies on the global `masked_file` variable, which is not ideal but consistent with the script's design.
        if masked_file:
            mask_type = masked_file.split('_')[0]
            if mask_type == 'lower':
                weights['top_half'] = 2.0  # Emphasize the visible part
                weights['bottom_half'] = 0.2  # De-emphasize the masked part
            elif mask_type == 'upper':
                weights['bottom_half'] = 2.0
                weights['top_half'] = 0.2
            elif mask_type == 'left':
                weights['right_half'] = 2.0
                weights['left_half'] = 0.2
            elif mask_type == 'right':
                weights['left_half'] = 2.0
                weights['right_half'] = 0.2

        weighted_scores = []
        total_weight = 0.0
        for name, v_emb in entry.get('embeddings', {}).items():
            w = weights.get(name, 0.5)  # Default weight for any other variants
            v = np.array(v_emb).reshape(-1)
            sim = float(np.dot(q, v))
            weighted_scores.append(sim * w)
            total_weight += w
        return float(sum(weighted_scores) / total_weight) if total_weight > 0 else 0.0
    elif method == 'mahalanobis':
        # compute Mahalanobis distance between q and mean, then convert to similarity
        if 'embedding_avg_norm' in entry:
            mu = entry['embedding_avg_norm'].reshape(-1)
        else:
            mu = entry['embedding_avg'].reshape(-1)
            mu = mu / np.linalg.norm(mu)
        cov = entry.get('embedding_cov')
        if cov is None or cov.size == 0:
            # fallback to cosine
            return float(np.dot(q, mu))
        # Fast diagonal Mahalanobis approximation (avoid costly full-matrix inverse)
        try:
            diag = np.diag(cov)
            eps = 1e-6
            denom = diag + eps
            diff = q - mu
            d2 = float(((diff**2) / denom).sum())
            d = np.sqrt(max(d2, 0.0))
            return float(1.0 / (1.0 + d))
        except Exception:
            # final fallback to cosine
            return float(np.dot(q, mu))

    else:
        return 0.0


def resize_to_h(img, h):
    h0, w0 = img.shape[:2]
    new_w = max(1, int(w0 * (h / h0)))
    return cv2.resize(img, (new_w, h))


def compute_recon_embedding(masked_file, unet_path='models/unet.pth', device='cpu'):
    """Load masked image, run U-Net reconstruction and FaceNet, return L2-normalized embedding (1D numpy array).
    Returns (embedding, recon_rgb, masked_rgb) where recon_rgb is the reconstructed RGB image and masked_rgb is the loaded masked RGB image.
    """
    device = torch.device(device)
    file = masked_file.split("_", 1)[1]
    full_path = "data/celeba/full/" + file
    masked_path = "data/celeba/masked/" + masked_file

    full_img = cv2.imread(full_path)
    masked_img = cv2.imread(masked_path)
    if full_img is None or masked_img is None:
        raise FileNotFoundError("Could not load input images for embedding computation.")

    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

    # load unet on device
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.eval()

    masked_tensor_unet = transform_unet(masked_img).unsqueeze(0).float().to(device)
    with torch.no_grad():
        recon = unet(masked_tensor_unet)[0].permute(1, 2, 0).cpu().numpy()

    recon_resized = cv2.resize(recon, (full_img.shape[1], full_img.shape[0]))

    # facenet (local instance) on device
    facenet_local = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    recon_tensor = transform_facenet((recon_resized * 255).astype(np.uint8)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = facenet_local(recon_tensor)
    emb = emb.cpu().numpy().reshape(-1)
    emb = emb / np.linalg.norm(emb)

    return emb, (recon_resized, masked_img)


def compute_recon_embeddings(masked_files, unet_path='models/unet.pth', device='cpu', batch_size=8):
    """Compute recon embeddings for a list of masked filenames in batches.

    Returns list of (embedding, recon_resized, masked_img) in the same order as masked_files.
    """
    device = torch.device(device)
    # load models once
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.eval()
    facenet_local = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    out = []
    # prepare tensors list
    imgs = []
    metas = []
    for mf in masked_files:
        file = mf.split("_", 1)[1]
        full_path = "data/celeba/full/" + file
        masked_path = "data/celeba/masked/" + mf
        full_img = cv2.imread(full_path)
        masked_img = cv2.imread(masked_path)
        if full_img is None or masked_img is None:
            out.append((None, (None, None)))
            continue
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        imgs.append((full_img, masked_img))
        metas.append((full_img.shape[1], full_img.shape[0]))

    # process in batches
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i+batch_size]
        # unet inputs
        tensors = torch.stack([transform_unet(m[1]) for m in batch]).float().to(device)
        with torch.no_grad():
            recons = unet(tensors).cpu().permute(0, 2, 3, 1).numpy()
        # for each in batch, resize recon to full dimensions and run facenet in sub-batches
        recon_tensors = []
        for j, r in enumerate(recons):
            full_w, full_h = batch[j][0].shape[1], batch[j][0].shape[0]
            recon_resized = cv2.resize(r, (full_w, full_h))
            recon_img_uint8 = (np.clip(recon_resized * 255, 0, 255).astype(np.uint8))
            recon_tensors.append(transform_facenet(recon_img_uint8))
        recon_batch = torch.stack(recon_tensors).float().to(device)
        with torch.no_grad():
            embs = facenet_local(recon_batch).cpu().numpy()
        # normalize and append to out
        for k in range(embs.shape[0]):
            emb = embs[k].reshape(-1)
            emb = emb / np.linalg.norm(emb)
            recon_resized = (np.clip(recons[k] * 255, 0, 255).astype(np.uint8))
            masked_img = batch[k][1]
            out.append((emb, (recon_resized, masked_img)))

    # fix length: any skipped files earlier will be (None, ...); ensure same ordering
    return out

# Images are loaded at runtime inside main(); keep placeholders here for import-time
full_img = None
masked_img = None

# U-Net reconstruction moved inside main() to avoid heavy work on import
recon_resized = None  # set in main() when running as script

def main(argv=None):
    import argparse
    # parse arguments (backwards compatible with previous positional top_k)
    parser = argparse.ArgumentParser(description='Identify person from masked image using reconstructed face embeddings')
    parser.add_argument('masked_file', help='masked filename (e.g. lower_000161.jpg)')
    parser.add_argument('top_k', nargs='?', type=int, default=3, help='number of top matches to display (positional, optional)')
    parser.add_argument('--method', choices=['avg_cosine', 'ensemble_max', 'ensemble_mean', 'mahalanobis', 'weighted_ensemble'], default='avg_cosine', help='matching method')
    parser.add_argument('--threshold', type=float, default=None, help='recognition threshold (higher means stricter); if omitted use --auto-threshold')
    parser.add_argument('--auto-threshold', action='store_true', help='use recommended thresholds from a thresholds JSON file')
    parser.add_argument('--thresholds-file', type=str, default='thresholds_recommended.json', help='path to thresholds JSON (used with --auto-threshold)')
    parser.add_argument('--no-show', action='store_true', help='save result image but do not display GUI')
    parser.add_argument('--save-metadata', action='store_true', help='save per-match JSON metadata alongside the result image')
    parser.add_argument('--device', default='cpu', help='device to run models on, e.g. cpu or cuda')
    parser.add_argument('--use-index', action='store_true', help='use Faiss index to retrieve candidates')
    parser.add_argument('--index-path', default='embeddings_index.faiss', help='path to faiss index file (created with scripts/build_faiss_index.py)')
    parser.add_argument('--knn', type=int, default=200, help='number of neighbors to retrieve from index before rescoring')

    args = parser.parse_args(argv)

    masked_file = args.masked_file
    file = masked_file.split("_", 1)[1]

    full_path = "data/celeba/full/" + file
    masked_path = "data/celeba/masked/" + masked_file

    # If requested, load recommended thresholds and set default threshold for chosen method
    if args.auto_threshold:
        try:
            import json
            with open(args.thresholds_file, 'r') as _f:
                thr = json.load(_f)
            if args.method in thr and 'threshold' in thr[args.method]:
                args.threshold = float(thr[args.method]['threshold'])
                # if running headless, print short note
                if args.no_show:
                    print(f"Using auto-threshold for method {args.method}: {args.threshold:.4f}")
        except Exception:
            # if thresholds file missing or invalid, fall back to default 0.5
            args.threshold = args.threshold if args.threshold is not None else 0.5
    # If threshold still None, set a sensible default
    if args.threshold is None:
        args.threshold = 0.5

    # -------------------------------
    # Load images
    # -------------------------------
    full_img = cv2.imread(full_path)
    masked_img = cv2.imread(masked_path)
    if full_img is None or masked_img is None:
        raise FileNotFoundError("Could not load input images.")

    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

    # compute reconstruction and embedding on requested device
    emb_recon, (recon_resized, masked_img) = compute_recon_embedding(masked_file, device=args.device)

    # -------------------------------
    # Load detailed embedding database
    # -------------------------------
    db = np.load("embeddings_db_detailed.npy", allow_pickle=True).item()

    # -------------------------------
    # Compute scores (optionally use Faiss index to shortlist candidates)
    # -------------------------------
    method = args.method
    scores = []

    if args.use_index:
        try:
            import faiss
            names = list(db.keys())
            index_path = args.index_path
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
                q = emb_recon.astype('float32').reshape(1, -1)
                D, I = index.search(q, args.knn)
                cand_idxs = [int(i) for i in I[0] if i >= 0]
                cand_names = [names[i] for i in cand_idxs]
                for fname in cand_names:
                    entry = db.get(fname, {})
                    score = match_score(entry, emb_recon, method=method)
                    scores.append((fname, float(score)))
            else:
                print(f"Faiss index not found at {index_path}; falling back to full scan")
        except Exception as e:
            print(f"Faiss not available or failed to load: {e}; falling back to full scan")

    # fallback to full scan if index not used or returned no candidates
    if not scores:
        for fname, entry in db.items():
            score = match_score(entry, emb_recon, method=method)
            scores.append((fname, float(score)))

    # Sort by similarity (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # -------------------------------
    # Visualize top-k matches (default k from args.top_k)
    # -------------------------------
    top_k = args.top_k

    best_match, best_score = scores[0]
    ground_truth = file  # e.g. "000201.jpg"

    # Recognition logic (for overlay)
    topk = scores[:top_k]
    topk_files = [fname for fname, _ in topk]
    if ground_truth in topk_files:
        recognition_text = f"✅ Person recognized (in Top-{top_k})"
        recognized = True
    elif best_score > args.threshold:
        recognition_text = f"✅ Person recognized (Top-1 confident match, score={best_score:.3f})"
        recognized = True
    else:
        recognition_text = f"❌ Person not confidently recognized (best_score={best_score:.3f})"
        recognized = False

    # Build visualization
    import matplotlib.pyplot as plt

    # Helpers (resize_to_h already defined above)
    disp_h = 256
    padding = 10

    # Left column: masked input above reconstruction
    masked_disp = resize_to_h(masked_img, disp_h)
    recon_uint8 = np.clip(recon_resized * 255, 0, 255).astype(np.uint8)
    recon_disp = resize_to_h(recon_uint8, disp_h)

    left_w = max(masked_disp.shape[1], recon_disp.shape[1])
    left_col = np.ones((disp_h * 2 + padding, left_w, 3), dtype=np.uint8) * 255
    mx = (left_w - masked_disp.shape[1]) // 2
    rx = (left_w - recon_disp.shape[1]) // 2
    left_col[0:disp_h, mx:mx+masked_disp.shape[1]] = masked_disp
    left_col[disp_h+padding:disp_h+padding+disp_h, rx:rx+recon_disp.shape[1]] = recon_disp

    # Labels on left column
    cv2.putText(left_col, "Masked Input", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, cv2.LINE_AA)
    cv2.putText(left_col, "Masked Input", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(left_col, "Reconstruction", (10, disp_h+padding+28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, cv2.LINE_AA)
    cv2.putText(left_col, "Reconstruction", (10, disp_h+padding+28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

    # Right columns: top-k matches with thumbnails and metadata
    import json

    match_panels = []
    metadata = {'query': masked_file, 'method': method, 'top_k': top_k, 'matches': []}
    for fname, score in topk:
        entry = db.get(fname, {})
        img_path = entry.get('file', "data/celeba/full/" + fname)
        img = cv2.imread(img_path)
        if img is None:
            panel = np.ones((disp_h, disp_h, 3), dtype=np.uint8) * 200
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # full thumbnail (top portion)
            top_h = int(disp_h * 0.6)
            full_thumb = resize_to_h(img, top_h)
            # face crop thumbnail
            box = entry.get('box')
            if box and len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                # clamp
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    face = img[y1:y2, x1:x2]
                else:
                    face = img
            else:
                # center crop fallback
                h0, w0 = img.shape[:2]
                s = int(min(h0, w0) * 0.6)
                cx, cy = w0//2, h0//2
                face = img[max(0, cy-s//2):min(h0, cy+s//2), max(0, cx-s//2):min(w0, cx+s//2)]

            small_h = int(disp_h * 0.36)
            face_thumb = resize_to_h(face, small_h)
            # left/right halves of face
            fh, fw = face_thumb.shape[:2]
            left = cv2.resize(face_thumb[:, :fw//2], (fw, fh))
            right = cv2.resize(face_thumb[:, fw//2:], (fw, fh))

            panel_w = max(full_thumb.shape[1], left.shape[1]*2 + padding)
            panel = np.ones((disp_h, panel_w, 3), dtype=np.uint8) * 255
            # place full_thumb at top center
            x0 = (panel_w - full_thumb.shape[1]) // 2
            panel[0:full_thumb.shape[0], x0:x0+full_thumb.shape[1]] = full_thumb
            # place left/right at bottom
            y0 = full_thumb.shape[0] + 6
            lx = (panel_w - (left.shape[1]*2 + padding)) // 2
            panel[y0:y0+left.shape[0], lx:lx+left.shape[1]] = left
            panel[y0:y0+right.shape[0], lx+left.shape[1]+padding:lx+left.shape[1]+padding+right.shape[1]] = right

        # overlay texts: score and detection prob
        text_score = f"{score:.4f}"
        detp = entry.get('det_prob', 0.0)
        text_det = f"det:{detp:.2f}"
        # shadow for readability
        cv2.putText(panel, text_score, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(panel, text_score, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(panel, text_det, (8, panel.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 2, cv2.LINE_AA)

        match_panels.append(panel)
        metadata['matches'].append({'file': fname, 'score': float(score), 'det_prob': float(detp), 'box': entry.get('box')})

    # build right column from panels
    right_w = sum(p.shape[1] for p in match_panels) + padding * (len(match_panels)-1)
    right_col = np.ones((disp_h, right_w, 3), dtype=np.uint8) * 255 if right_w > 0 else np.ones((disp_h, 1, 3), dtype=np.uint8) * 255
    x = 0
    for p in match_panels:
        right_col[0:disp_h, x:x+p.shape[1]] = p
        x += p.shape[1] + padding

    # optionally save metadata JSON
    if getattr(args, 'save_metadata', False):
        json_path = f"result_{masked_file}.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {json_path}")

    # Compose final canvas
    header_h = 60
    canvas_h = left_col.shape[0] + header_h
    canvas_w = left_col.shape[1] + padding + right_col.shape[1]
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    canvas[header_h:header_h+left_col.shape[0], 0:left_col.shape[1]] = left_col
    # center right_col vertically relative to content
    y = header_h + (left_col.shape[0] - right_col.shape[0]) // 2
    canvas[y:y+right_col.shape[0], left_col.shape[1]+padding:left_col.shape[1]+padding+right_col.shape[1]] = right_col

    # Header text (include method)
    header = f"Method: {method} | {recognition_text}"
    cv2.putText(canvas, header, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,0) if recognized else (0,0,255), 2, cv2.LINE_AA)

    # Save and display (no terminal text output)
    out_path = f"result_{masked_file}.png"
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    if not args.no_show:
        plt.figure(figsize=(12, 6))
        plt.imshow(canvas)
        plt.axis('off')
        plt.title(f"Query: {masked_file}  |  Method: {method}  |  Top-{top_k} matches")
        plt.show()
    else:
        print(f"Result saved to {out_path}")


if __name__ == '__main__':
    main()
