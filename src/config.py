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
# Working PlantVillage mirror on Hugging Face: 38 classes, 43,503 train /
# 10,878 test images, 256x256 RGB. Loads with plain
# datasets.load_dataset("GVJahnavi/PlantVillage_dataset"), no auth needed.
# (mohanty/PlantVillage is broken — Python 2.7 loading script + 401 on
# data.zip — and is deliberately NOT used.)
HF_DATASET_ID = "GVJahnavi/PlantVillage_dataset"
HF_TEST_SPLIT = "test"   # HF's predefined test split, used as-is

# Image preprocessing
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)   # ImageNet-normalized stats, fine for leaves
STD = (0.229, 0.224, 0.225)

# Train / val split (stratified on the HF train split; test = HF predefined)
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
TEST_RATIO = 0.0        # test comes from HF's predefined split, not ours
RANDOM_SEED = 42

# Batch size tuned for CPU training (no GPU in this environment)
BATCH_SIZE = 32
# Windows uses the "spawn" multiprocessing start method, so non-zero
# NUM_WORKERS reliably trips "attempt to start a new process before the
# current process has finished its bootstrapping phase". On Linux (e.g.
# Colab's T4) a couple of workers keep the GPU fed while the model runs.
import platform

NUM_WORKERS = 2 if platform.system() != "Windows" else 0

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