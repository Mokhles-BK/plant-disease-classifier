"""Dataset loading, stratified split, transforms and DataLoaders.

Loads the Hugging Face PlantVillage mirror, performs a stratified
train/val split (test comes from HF's predefined split), wraps the result in a
plain ``torch.utils.data.Dataset`` and exposes train/val/test ``DataLoader``s
plus class weights / weighted sampler from ``utils.compute_class_weights``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets import load_dataset

try:
    from . import config
    from .utils import compute_class_weights, set_seed
except ImportError:  # allow ``python src/data.py`` from the repo root
    import config  # type: ignore
    from utils import compute_class_weights, set_seed  # type: ignore

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
# Lazily populated by ``load_raw_dataset`` so the module never fails to import
# if the HF dataset is unavailable (e.g. offline smoke tests).
NUM_CLASSES: int = 0
CLASS_NAMES: list[str] = []


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def train_transforms() -> transforms.Compose:
    """Augmentation for the training split."""
    return transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ]
    )


def eval_transforms() -> transforms.Compose:
    """Deterministic transforms for validation and test splits."""
    return transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class PlantVillageDataset(Dataset):
    """Thin ``torch`` Dataset over an HF ``Dataset`` split.

    Each row is a dict with an ``image`` (PIL) and an integer ``label``.
    """

    def __init__(self, hf_split, transform=None):
        self.hf = hf_split
        self.transform = transform or eval_transforms()

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.hf[idx]
        image = row["image"].convert("RGB")
        label = int(row["label"])
        return self.transform(image), label


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_raw_dataset():
    """Download / load the HF dataset and populate module-level constants."""
    global NUM_CLASSES, CLASS_NAMES
    ds = load_dataset(
        config.HF_DATASET_ID,
        cache_dir=str(config.DATA_DIR / ".hf_cache"),
    )
    names = ds["train"].features["label"].names
    CLASS_NAMES = list(names)
    NUM_CLASSES = len(names)
    return ds


def _cap_per_class(hf_split, max_samples: int):
    """Subsample so no class has more than ``max_samples`` rows (dev mode)."""
    import numpy as np

    labels = np.array(hf_split["label"])
    keep = []
    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0]
        if len(idx) > max_samples:
            idx = idx[:max_samples]
        keep.extend(idx.tolist())
    keep.sort()
    return hf_split.select(keep)


def build_splits(max_samples: Optional[int] = None):
    """Return stratified train/val/test HF splits.

    ``max_samples`` caps every class to that many rows in train+val — a dev-mode
    knob for smoke-testing the pipeline on CPU in minutes instead of hours.
    """
    set_seed(config.RANDOM_SEED)
    ds = load_raw_dataset()

    train_full = ds["train"]
    if max_samples is not None:
        train_full = _cap_per_class(train_full, max_samples)

    split = train_full.train_test_split(
        test_size=1.0 - config.TRAIN_RATIO,
        stratify_by_column="label",
        seed=config.RANDOM_SEED,
    )
    train_hf, val_hf = split["train"], split["test"]
    test_hf = ds[config.HF_TEST_SPLIT]

    return train_hf, val_hf, test_hf


def make_loaders(max_samples: Optional[int] = None) -> Tuple[
    DataLoader, DataLoader, DataLoader, torch.Tensor, Optional[object]
]:
    """Build train/val/test loaders + class weights + (optional) sampler.

    Returns ``(train_loader, val_loader, test_loader, class_weights, sampler)``.
    """
    train_hf, val_hf, test_hf = build_splits(max_samples=max_samples)

    # Class weights / weighted sampler computed on the (possibly capped) train
    # split so they stay consistent with what the model actually sees.
    labels = list(train_hf["label"])
    class_weights, sampler = compute_class_weights(labels, NUM_CLASSES)

    train_set = PlantVillageDataset(train_hf, transform=train_transforms())
    val_set = PlantVillageDataset(val_hf, transform=eval_transforms())
    test_set = PlantVillageDataset(test_hf, transform=eval_transforms())

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights, sampler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and inspect data loaders.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap each class to N rows (dev mode, e.g. --max-samples 200).",
    )
    args = parser.parse_args()

    tl, vl, tl_test, weights, sampler = make_loaders(max_samples=args.max_samples)
    print(f"NUM_CLASSES={NUM_CLASSES}  CLASS_NAMES={CLASS_NAMES[:3]}...")
    print(f"train={len(tl.dataset)} val={len(vl.dataset)} test={len(tl_test.dataset)}")
    print(f"class_weights shape={tuple(weights.shape)} sampler={sampler is not None}")
    x, y = next(iter(tl))
    print(f"batch x={tuple(x.shape)} y={tuple(y.shape)}")