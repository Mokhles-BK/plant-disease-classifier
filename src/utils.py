"""Shared helpers used across data.py, train.py, evaluate.py and api/."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

try:
    from . import config
except ImportError:  # allow ``python src/utils.py`` from the repo root
    import config  # type: ignore


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available torch device (CPU here, CUDA if present)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(
    labels: List[int], num_classes: int
) -> Tuple[torch.Tensor, WeightedRandomSampler | None]:
    """Return inverse-frequency class weights + an optional weighted sampler.

    Returns (weights, sampler). sampler is None when the data is roughly
    balanced (max/min < IMBALANCE_RATIO_THRESHOLD).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    # Inverse frequency, smoothed so a single rare class doesn't explode.
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * num_classes

    # Build a weighted sampler for the training loader when imbalance is high.
    sampler = None
    ratio = counts.max() / max(counts.min(), 1e-9)
    if ratio > config.IMBALANCE_RATIO_THRESHOLD:
        sample_weights = 1.0 / np.sqrt(counts)[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(labels),
            replacement=True,
        )

    return torch.as_tensor(weights, dtype=torch.float32), sampler


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def topk_from_logits(logits: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (values, indices) of the top-k softmax probabilities."""
    probs = torch.softmax(logits, dim=-1)
    return torch.topk(probs, k, dim=-1)