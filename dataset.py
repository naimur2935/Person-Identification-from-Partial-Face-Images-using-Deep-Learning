import os
import cv2
from torch.utils.data import Dataset

class FaceReconstructionDataset(Dataset):
    def __init__(self, full_dir, masked_dir, mask_types=None, transform=None):
        self.full_dir = full_dir
        self.masked_dir = masked_dir
        self.files = os.listdir(full_dir)
        # Default: train on all mask types
        self.mask_types = mask_types if mask_types else ["lower", "upper", "left", "right", "random"]
        self.transform = transform

    def __len__(self):
        return len(self.files) * len(self.mask_types)

    def __getitem__(self, idx):
        # Pick which file and mask type
        file_idx = idx // len(self.mask_types)
        mask_idx = idx % len(self.mask_types)
        file = self.files[file_idx]
        mask_type = self.mask_types[mask_idx]

        # Full image (ground truth)
        full_img = cv2.imread(os.path.join(self.full_dir, file))
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

        # Masked image (input)
        masked_file = f"{mask_type}_{file}"
        masked_img = cv2.imread(os.path.join(self.masked_dir, masked_file))
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            full_img = self.transform(full_img)
            masked_img = self.transform(masked_img)

        return masked_img, full_img
