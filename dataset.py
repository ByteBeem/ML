import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ─────────────────────────────────────────────
# Raw pickle loader
# ─────────────────────────────────────────────
def _load_pickle(path: str):
    print(f"  [DEBUG] Loading pickle: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    images = d[b"data"]                        # (N, 3072) uint8
    labels = d[b"fine_labels"]                 # list[int] 0-99
    images = images.reshape(-1, 3, 32, 32)
    print(f"  [DEBUG] Pickle loaded — images {images.shape}, labels {len(labels)}")
    return images, np.array(labels, dtype=np.int64)


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────
class CIFAR100Kaggle(Dataset):
    """
    Drop-in for torchvision.datasets.CIFAR100.
    Downloads once via kagglehub, then loads from cache.
    Requires:  pip install kagglehub
    """

    def __init__(self, train: bool = True, transform=None):
        super().__init__()
        self.transform = transform

        try:
            import kagglehub
        except ImportError:
            print(
                "[ERROR] kagglehub is not installed.\n"
                "        Run:  pip install kagglehub\n"
                "        Then set up your Kaggle API key — see:\n"
                "        https://github.com/Kaggle/kagglehub#authentication"
            )
            sys.exit(1)

        print(f"  [DEBUG] Downloading / using cached CIFAR-100 via kagglehub …")
        root = kagglehub.dataset_download("fedesoriano/cifar100")
        print(f"  [DEBUG] Dataset root: {root}")

        split_file = os.path.join(root, "train" if train else "test")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Expected CIFAR-100 split file not found: {split_file}\n"
                f"Contents of {root}: {os.listdir(root)}"
            )

        self.images, self.labels = _load_pickle(split_file)

        print(f"  [DEBUG] CIFAR-100 {'train' if train else 'val'} — "
              f"{len(self.images):,} samples from {split_file}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = self.images[idx].transpose(1, 2, 0)   # (32, 32, 3) HWC uint8
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


# ─────────────────────────────────────────────
# Transforms (shared)
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
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing dataset loader …")
    ds = CIFAR100Kaggle(train=True, transform=TRAIN_TRANSFORM)
    img, label = ds[0]
    print(f"  Image tensor shape : {img.shape}")
    print(f"  Label              : {label}")
    print(f"  Dataset length     : {len(ds)}")
    print("OK")