import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

full_dir = "data/celeba/full/"
output_file = "embeddings_db_detailed.npy"

transform_facenet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Models (CPU by default; set device if GPU is available)
device = 'cpu'
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=False, device=device)


def l2_norm(v):
    v = v.reshape(-1)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_embedding(img):
    """Return a flattened numpy embedding (L2-normalized) for an RGB image (numpy array)."""
    tensor = transform_facenet(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = facenet(tensor).cpu().numpy().reshape(-1)
    return l2_norm(emb)


def safe_crop_face(rgb_img):
    """Detect face with MTCNN and return a cropped face (RGB numpy array of variable size).
    If detection fails, returns a center crop of the image as fallback."""
    pil = Image.fromarray(rgb_img)
    try:
        boxes, probs = mtcnn.detect(pil)
    except Exception:
        boxes, probs = None, None

    if boxes is not None and len(boxes) > 0 and probs is not None and probs[0] is not None and probs[0] > 0.3:
        x1, y1, x2, y2 = boxes[0]
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(rgb_img.shape[1], x2)), int(min(rgb_img.shape[0], y2))
        # add small padding
        w = x2 - x1
        h = y2 - y1
        pad = int(0.2 * max(w, h))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(rgb_img.shape[1], x2 + pad)
        y2 = min(rgb_img.shape[0], y2 + pad)
        crop = rgb_img[y1:y2, x1:x2]
        return crop, [x1, y1, x2, y2], float(probs[0])
    # fallback: center crop square
    h, w = rgb_img.shape[:2]
    s = int(min(h, w) * 0.8)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - s // 2)
    y1 = max(0, cy - s // 2)
    x2 = min(w, cx + s // 2)
    y2 = min(h, cy + s // 2)
    return rgb_img[y1:y2, x1:x2], [x1, y1, x2, y2], 0.0


# Augmentation helpers

def augment_variants(face_rgb):
    """Return a dict of augmentation name -> image (RGB numpy arrays).
    Includes flips, brightness, rotations, blur, and left/right halves.
    """
    variants = {}
    variants['full'] = face_rgb
    variants['flip'] = cv2.flip(face_rgb, 1)
    variants['bright'] = cv2.convertScaleAbs(face_rgb, alpha=1.2, beta=30)
    variants['dark'] = cv2.convertScaleAbs(face_rgb, alpha=0.8, beta=-30)
    variants['contrast'] = cv2.convertScaleAbs(face_rgb, alpha=1.5, beta=0)
    variants['blur'] = cv2.GaussianBlur(face_rgb, (5, 5), 0)

    # Gaussian Noise
    noise = np.random.normal(0, 15, face_rgb.shape).astype(np.int16)
    variants['noise'] = np.clip(face_rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Grayscale (simulated as 3-channel)
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    variants['gray'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # rotations
    pil = Image.fromarray(face_rgb)
    variants['rot_plus10'] = np.array(pil.rotate(10, resample=Image.BILINEAR))
    variants['rot_minus10'] = np.array(pil.rotate(-10, resample=Image.BILINEAR))
    variants['rot_plus5'] = np.array(pil.rotate(5, resample=Image.BILINEAR))
    variants['rot_minus5'] = np.array(pil.rotate(-5, resample=Image.BILINEAR))
    # halves (helpful for partial face matching)
    h, w = face_rgb.shape[:2]
    left = face_rgb[:, :w//2]
    right = face_rgb[:, w//2:]
    # resize halves back to face size for consistent embeddings
    left_resized = cv2.resize(left, (w, h))
    right_resized = cv2.resize(right, (w, h))
    variants['left_half'] = left_resized
    variants['right_half'] = right_resized

    # top/bottom halves
    top = face_rgb[:h//2, :]
    bottom = face_rgb[h//2:, :]
    variants['top_half'] = cv2.resize(top, (w, h))
    variants['bottom_half'] = cv2.resize(bottom, (w, h))

    return variants


# Main loop

db = {}
failed = 0
for i, fname in enumerate(sorted(os.listdir(full_dir))):
    img_path = os.path.join(full_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        failed += 1
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_crop, box, prob = safe_crop_face(img)

    # ensure color and reasonable size
    if face_crop is None or face_crop.size == 0:
        failed += 1
        continue

    # generate variants and embeddings
    variants = augment_variants(face_crop)
    emb_dict = {}
    emb_list = []
    for name, vimg in variants.items():
        try:
            emb = get_embedding(vimg)
            emb_dict[name] = emb
            emb_list.append(emb)
        except Exception:
            # skip bad transforms
            continue

    if len(emb_list) == 0:
        failed += 1
        continue

    emb_stack = np.vstack(emb_list)
    emb_avg = np.mean(emb_stack, axis=0)
    emb_avg_norm = l2_norm(emb_avg)
    emb_cov = np.cov(emb_stack, rowvar=False)

    db[fname] = {
        'file': img_path,
        'detected': prob > 0,
        'box': box,
        'det_prob': prob,
        'embeddings': emb_dict,  # each is 1D L2-normed
        'embeddings_list': emb_stack,  # shape (n_variants, dim)
        'embedding_avg': emb_avg.reshape(1, -1),
        'embedding_avg_norm': emb_avg_norm.reshape(1, -1),
        'embedding_cov': emb_cov,
        'n_variants': emb_stack.shape[0],
    }

print(f"Done. Images processed: {len(db)}, failed: {failed}. Saving detailed database...")
np.save(output_file, db)
print(f"✅ Detailed embedding database saved to {output_file}")
