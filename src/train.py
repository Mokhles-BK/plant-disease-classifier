"""Training loop for either model.

Usage:
    python src/train.py --model baseline
    python src/train.py --model transfer
    python src/train.py --model baseline --max-samples 200 --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from . import config
    from . import data
    from .data import make_loaders
    from .models import build_model
    from .utils import get_device, save_json, set_seed
except ImportError:  # allow ``python src/train.py`` from the repo root
    import config  # type: ignore
    import data  # type: ignore
    from data import make_loaders  # type: ignore
    from models import build_model  # type: ignore
    from utils import get_device, save_json, set_seed  # type: ignore


def _accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> tuple[float, float]:
    """One train or eval pass. Returns (loss, accuracy)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total = 0
    correct = 0
    n_samples = 0
    loss_sum = 0.0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            loss_sum += loss.item() * bs
            n_samples += bs
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += bs

    return loss_sum / n_samples, correct / total


def train(
    model_name: str = "baseline",
    epochs: int = config.EPOCHS,
    max_samples: Optional[int] = None,
    ckpt_path: Optional[Path] = None,
    lr: float = config.LEARNING_RATE,
) -> dict:
    """Train the requested model and return a run-history dict."""
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"device={device} model={model_name} epochs={epochs} max_samples={max_samples}")

    train_loader, val_loader, _, class_weights, sampler = make_loaders(
        max_samples=max_samples
    )

    model = build_model(model_name).to(device)

    # Weighted cross-entropy only when the sampler is NOT already balancing
    # (otherwise we'd double-count the imbalance correction).
    if sampler is None:
        loss_weight = class_weights.to(device)
    else:
        loss_weight = None
    criterion = nn.CrossEntropyLoss(weight=loss_weight)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    if ckpt_path is None:
        ckpt_path = config.BASELINE_CKPT if model_name == "baseline" else config.TRANSFER_CKPT
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device
        )

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "lr": lr_now,
            }
        )
        print(
            f"[{epoch:3d}/{epochs}] train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | lr={lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_weights": class_weights.cpu(),
                    "sampler": sampler is not None,
                    # 38 human-readable label strings, so consumers (the Gradio
                    # app, evaluate.py) never need to reload the full HF dataset
                    # just to map an index back to a class name.
                    "class_names": list(data.CLASS_NAMES) if data.CLASS_NAMES else None,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"early stopping: no val improvement for "
                    f"{config.EARLY_STOPPING_PATIENCE} epochs"
                )
                break

    run_info = {
        "model_name": model_name,
        "epochs_run": len(history),
        "best_val_loss": best_val_loss,
        "history": history,
        "ckpt": str(ckpt_path),
    }
    save_json(run_info, ckpt_path.with_suffix(".json"))
    print(f"checkpoint saved to {ckpt_path}")
    return run_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a plant disease classifier.")
    parser.add_argument("--model", choices=["baseline", "transfer"], required=True)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap each class to N rows (dev mode).",
    )
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--ckpt", type=Path, default=None)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        max_samples=args.max_samples,
        lr=args.lr,
        ckpt_path=args.ckpt,
    )