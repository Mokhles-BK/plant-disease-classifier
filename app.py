"""Gradio web app for the plant disease classifier.

Upload a leaf photo -> top-3 disease predictions with confidence percentages.

Run locally:
    python app.py

Then open the printed http://127.0.0.1:7860 URL in a browser. The interface
is a single gr.Interface: image upload in, top-3 predictions out. The same
app.py is what gets deployed to a Hugging Face Space (see README).

The checkpoint is loaded once at startup (config.TRANSFER_CKPT) and the model
is rebuilt from the checkpoint's own ``model_name`` and inferred
``num_classes`` (from the checkpoint's ``class_weights`` tensor), so the app
never needs to know the dataset layout at import time. Class names come from
the checkpoint's ``class_names`` field when present; older checkpoints fall
back to loading the HF dataset once just to read the 38 label strings.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import torch
from PIL import Image

try:
    from . import config
    from .data import eval_transforms
    from .models import build_model
    from .utils import topk_from_logits
except ImportError:  # allow ``python app.py`` from the repo root
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))
    import config  # type: ignore
    from data import eval_transforms  # type: ignore
    from models import build_model  # type: ignore
    from utils import topk_from_logits  # type: ignore


# ---------------------------------------------------------------------------
# Model + metadata loading (runs once at startup)
# ---------------------------------------------------------------------------
def _infer_num_classes(ckpt: dict) -> int:
    """Resolve the class count from the checkpoint, avoiding a dataset load.

    Mirrors src/evaluate.py: the class_weights tensor's first dimension is
    exactly the number of classes.
    """
    cw = ckpt.get("class_weights")
    if cw is not None:
        return int(cw.shape[0])
    # Last resort: load the dataset to learn NUM_CLASSES.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))
    from data import load_raw_dataset  # type: ignore

    load_raw_dataset()
    from data import NUM_CLASSES  # type: ignore

    return int(NUM_CLASSES)


def _load_class_names(ckpt: dict, num_classes: int) -> List[str]:
    """Return human-readable class names, preferring the checkpoint."""
    names = ckpt.get("class_names")
    if names and len(names) == num_classes:
        return list(names)
    # Fallback for checkpoints saved before class_names was persisted:
    # load the HF dataset once purely to read the 38 label strings.
    # NOTE: import the *module*, not the name — ``from data import CLASS_NAMES``
    # binds the value at import time, and load_raw_dataset() reassigns
    # data.CLASS_NAMES to a new list afterwards, so the bound name would
    # otherwise stay empty.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))
    from data import load_raw_dataset  # type: ignore
    from data import CLASS_NAMES as _cn  # type: ignore

    if not _cn:
        load_raw_dataset()
        from data import CLASS_NAMES as _cn  # type: ignore
    if _cn and len(_cn) == num_classes:
        return list(_cn)
    return [str(i) for i in range(num_classes)]


def load_model():
    """Load the transfer checkpoint and rebuild the model. Returns (model, class_names, device)."""
    ckpt_path = Path(config.BASELINE_CKPT)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_name = ckpt.get("model_name", "transfer")
    num_classes = _infer_num_classes(ckpt)
    class_names = _load_class_names(ckpt, num_classes)

    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, class_names, device


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _predict(model, transform, class_names, image: Image.Image, device, top_k: int = 3):
    """Run one forward pass and return a markdown-formatted top-k string."""
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    values, indices = topk_from_logits(logits, k=top_k)

    lines = []
    for value, idx in zip(values[0], indices[0]):
        pct = float(value) * 100
        lines.append(f"**{class_names[int(idx)]}** — {pct:.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the interface
# ---------------------------------------------------------------------------
model, CLASS_NAMES, DEVICE = load_model()
transform = eval_transforms()


def predict(image: Image.Image) -> str:
    if image is None:
        return "Please upload a leaf image."
    return _predict(model, transform, CLASS_NAMES, image, DEVICE, top_k=3)


app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload leaf photo"),
    outputs=gr.Markdown(label="Top-3 predictions"),
    title="Plant Disease Classifier",
    description=(
        "Upload a photo of a leaf and get the top-3 predicted plant diseases "
        "with confidence scores. Model: CNN trained from scratch on the "
        "Hugging Face PlantVillage dataset (38 classes, 97.7% test accuracy)."
    ),
    examples=None,
    cache_examples=False,
)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)