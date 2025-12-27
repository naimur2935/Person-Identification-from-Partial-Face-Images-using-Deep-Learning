import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from models.unet import UNet
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -------------------------------
# Paths
# -------------------------------
full_dir = "data/celeba/full/"
masked_dir = "data/celeba/masked/"

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
    transforms.Resize((160, 160)),  # FaceNet expects 160x160
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -------------------------------
# Load U-Net model
# -------------------------------
unet = UNet()
unet.load_state_dict(torch.load("models/unet.pth", map_location="cpu"))
unet.eval()

# -------------------------------
# Load FaceNet
# -------------------------------
facenet = InceptionResnetV1(pretrained='vggface2').eval()
cos = torch.nn.CosineSimilarity(dim=1)

# -------------------------------
# Metrics storage
# -------------------------------
psnr_scores, ssim_scores, cos_scores = [], [], []

# -------------------------------
# Loop through dataset
# -------------------------------
for fname in os.listdir(masked_dir):
    full_path = os.path.join(full_dir, fname.split("_", 1)[1])
    masked_path = os.path.join(masked_dir, fname)

    # Load images
    full_img = cv2.imread(full_path)
    masked_img = cv2.imread(masked_path)
    if full_img is None or masked_img is None:
        print(f"Skipping {fname}, image not found.")
        continue

    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # U-Net reconstruction
    # -------------------------------
    masked_tensor_unet = transform_unet(masked_img).unsqueeze(0).float()
    with torch.no_grad():
        recon = unet(masked_tensor_unet)[0].permute(1, 2, 0).cpu().numpy()

    # Resize recon to match full image dimensions
    recon_resized = cv2.resize(recon, (full_img.shape[1], full_img.shape[0]))

    # -------------------------------
    # PSNR & SSIM
    # -------------------------------
    psnr_scores.append(
        peak_signal_noise_ratio(full_img, (recon_resized * 255).astype(np.uint8), data_range=255)
    )
    ssim_scores.append(
        structural_similarity(full_img, (recon_resized * 255).astype(np.uint8), channel_axis=2)
    )

    # -------------------------------
    # FaceNet embeddings
    # -------------------------------
    full_tensor = transform_facenet(full_img).unsqueeze(0)
    recon_tensor = transform_facenet((recon_resized * 255).astype(np.uint8)).unsqueeze(0)

    with torch.no_grad():
        emb_full = facenet(full_tensor)
        emb_recon = facenet(recon_tensor)

    cos_scores.append(cos(emb_full, emb_recon).item())

# -------------------------------
# Print results
# -------------------------------
print("===================================")
print(f"Average PSNR: {np.mean(psnr_scores):.4f}")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
print(f"Average Cosine Similarity: {np.mean(cos_scores):.4f}")
print("===================================")
