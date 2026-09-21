"""Evaluate a trained checkpoint against the HF predefined test split.

Usage:
    python src/evaluate.py
    python src/evaluate.py --ckpt models/transfer_resnet18.pt
    python src/evaluate.py --ckpt models/baseline_cnn.pt --no-cm

Writes:
    reports/<model_name>_metrics.json   - full classification_report dict
    reports/<model_name>_confusion.png  - labeled confusion matrix (unless --no-cm)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

try:
    from . import config
    from .data import make_loaders
    from .models import build_model
    from .utils import get_device, save_json
except ImportError:  # allow ``python src/evaluate.py`` from the repo root
    import config  # type: ignore
    from data import make_loaders  # type: ignore
    from models import build_model  # type: ignore
    from utils import get_device, save_json  # type: ignore


def _infer_num_classes(ckpt: Dict) -> int:
    """Resolve the class count from the checkpoint, avoiding a dataset load."""
    cw = ckpt.get("class_weights")
    if cw is not None:
        return int(cw.shape[0])
    # Last resort: load the dataset to learn NUM_CLASSES.
    from data import load_raw_dataset  # type: ignore

    load_raw_dataset()
    from data import NUM_CLASSES  # type: ignore

    return int(NUM_CLASSES)


def _collect_predictions(
    model: torch.nn.Module, loader, device: torch.device
) -> Tuple[List[int], List[int]]:
    """Run the model over ``loader`` in eval mode, returning (y_true, y_pred)."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_true.extend(labels.tolist())
            y_pred.extend(preds)
    return y_true, y_pred


def _plot_confusion_matrix(
    cm,
    class_names: List[str],
    out_path: Path,
    normalize: bool = False,
) -> None:
    """Render and save a labeled confusion-matrix heatmap to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype("float64")
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(
            data, row_sums, out=np.zeros_like(data), where=row_sums != 0
        )

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(class_names)
    ax.set(
        xticks=range(n),
        yticks=range(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix" + (" (normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Annotate cells: counts from the original integer matrix, ratios when
    # normalized. Keep the two separate so the integer counts stay readable.
    fmt = ".2f" if normalize else "d"
    thresh = data.max() / 2.0 if data.max() > 0 else 0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j] if normalize else cm[i, j]
            ax.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                color="white" if data[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    ckpt_path: Path,
    reports_dir: Path | None = None,
    plot_cm: bool = True,
) -> Dict:
    """Evaluate a checkpoint and return the classification report dict."""
    ckpt_path = Path(ckpt_path)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", "model")
    num_classes = _infer_num_classes(ckpt)

    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    device = get_device()
    model.to(device)

    _, _, test_loader, _, _ = make_loaders()
    y_true, y_pred = _collect_predictions(model, test_loader, device)

    # Class names come from the data module (populated by make_loaders).
    from data import CLASS_NAMES  # type: ignore

    class_names = list(CLASS_NAMES) if CLASS_NAMES else [str(i) for i in range(num_classes)]

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
    )

    metrics_path = reports_dir / f"{model_name}_metrics.json"
    save_json(report, metrics_path)
    print(f"metrics saved to {metrics_path}")

    if plot_cm:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        cm_path = reports_dir / f"{model_name}_confusion.png"
        _plot_confusion_matrix(cm, class_names, cm_path)
        print(f"confusion matrix saved to {cm_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--ckpt", type=Path, default=config.BASELINE_CKPT)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--no-cm", action="store_true", help="skip confusion matrix plot")
    args = parser.parse_args()

    report = evaluate(args.ckpt, args.reports_dir, plot_cm=not args.no_cm)

    acc = report.get("accuracy", float("nan"))
    macro_f1 = report.get("macro avg", {}).get("f1-score", float("nan"))
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", float("nan"))
    print("\n=== Summary ===")
    print(f"accuracy    : {acc:.4f}")
    print(f"macro F1    : {macro_f1:.4f}")
    print(f"weighted F1 : {weighted_f1:.4f}")


if __name__ == "__main__":
    main()