"""Central configuration for the plant disease classifier project.

All paths, hyperparameters, and dataset choices live here so the training,
evaluation, and API code share a single source of truth.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Well-maintained PlantVillage mirror on Hugging Face (~38 classes, color
# leaf images). Sourced from the original PlantVillage paper repo.
HF_DATASET_ID = "mohanty/PlantVillage"
# Which split variant to use. "color" keeps the original RGB photos; the
# "segmented" variant has background removed which helps some models but
# costs more to process. We use color for a fair, representative comparison.
HF_VARIANT = "color"

# Image preprocessing
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)   # ImageNet-normalized stats, fine for leaves
STD = (0.229, 0.224, 0.225)

# Train / val / test split (stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Batch size tuned for CPU training (no GPU in this environment)
BATCH_SIZE = 32
NUM_WORKERS = 2

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_PATIENCE = 3
# If the largest class is more than this many times the smallest, we use a
# weighted loss / weighted sampler to fight imbalance.
IMBALANCE_RATIO_THRESHOLD = 3.0

# Where checkpoints live
BASELINE_CKPT = MODELS_DIR / "baseline_cnn.pt"
TRANSFER_CKPT = MODELS_DIR / "transfer_resnet18.pt"
BEST_MODEL_FOR_API = TRANSFER_CKPT  # updated after comparison

# Which backbone to use for transfer learning
TRANSFER_BACKBONE = "resnet18"  # options: "resnet18", "efficientnet_b0"