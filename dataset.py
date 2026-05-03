import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import kagglehub


# ─────────────────────────────────────────────
# Raw pickle loader
# ─────────────────────────────────────────────
def _load_pickle(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    # keys are bytes in the original CIFAR pickle
    images = d[b"data"]                        # (N, 3072) uint8
    labels = d[b"fine_labels"]                 # list of ints  (0-99)
    images = images.reshape(-1, 3, 32, 32)     # (N, 3, 32, 32)
    return images, np.array(labels, dtype=np.int64)


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────
class CIFAR100Kaggle(Dataset):
    """
    Drop-in for torchvision.datasets.CIFAR100.
    Downloads once via kagglehub, then loads from cache.
    """

    def __init__(self, train: bool = True, transform=None):
        super().__init__()
        self.transform = transform

        # Download / use cache
        root = kagglehub.dataset_download("fedesoriano/cifar100")

        split_file = os.path.join(root, "train" if train else "test")
        self.images, self.labels = _load_pickle(split_file)

        print(f"  Loaded CIFAR-100 {'train' if train else 'val'} "
              f"— {len(self.images):,} samples from {split_file}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # images are (3, 32, 32) uint8 — convert to PIL for transforms
        from PIL import Image
        img = self.images[idx].transpose(1, 2, 0)   # (32, 32, 3) HWC
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, int(self.labels[idx])


# ─────────────────────────────────────────────
# Transforms (shared between both training scripts)
# ─────────────────────────────────────────────
NORMALIZE = T.Normalize(
    mean=[0.5071, 0.4867, 0.4408],
    std =[0.2675, 0.2565, 0.2761],
)

TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    NORMALIZE,
])

VAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    NORMALIZE,
])


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing dataset loader...")
    ds = CIFAR100Kaggle(train=True, transform=TRAIN_TRANSFORM)
    img, label = ds[0]
    print(f"  Image tensor shape : {img.shape}")
    print(f"  Label              : {label}")
    print(f"  Dataset length     : {len(ds)}")
    print("OK")