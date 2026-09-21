"""Two model architectures: a small CNN from scratch and a transfer model.

(a) ``SmallCNN`` — 3-4 conv blocks, no pretrained weights, lightweight enough
    to train on CPU in reasonable time.
(b) ``TransferModel`` — backbone from ``TRANSFER_BACKBONE`` (resnet18 to
    start), backbone frozen, a fresh classification head replacing it.
Both match ``NUM_CLASSES`` from the dataset.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from . import config
    from . import data as _data  # noqa: F401  (ensures constants load)
except ImportError:  # pragma: no cover - direct execution
    import config  # type: ignore
    import data as _data  # type: ignore


def _num_classes(num_classes: Optional[int] = None) -> int:
    """Resolve the class count. If provided, use it; otherwise lazily load from data module.

    Callers should pass an explicit `num_classes` whenever possible to avoid
    triggering a full HF dataset download during model construction.
    """
    if num_classes is not None:
        return num_classes
    if _data.NUM_CLASSES:
        return _data.NUM_CLASSES
    _data.load_raw_dataset()
    return _data.NUM_CLASSES


# ---------------------------------------------------------------------------
# (a) Small CNN from scratch
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """4 conv-block CNN with no pretrained weights.

    Architecture: 4 conv blocks (32/64/128/256 channels) -> global average
    pooling -> 2-layer MLP head -> ``NUM_CLASSES`` logits.
    """

    def __init__(self, num_classes: Optional[int] = None):
        super().__init__()
        num_classes = num_classes or _num_classes()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# (b) Transfer learning
# ---------------------------------------------------------------------------
def _build_backbone(name: str) -> nn.Module:
    """Construct a torchvision backbone and return its feature extractor."""
    if name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        # resnet18: conv1 -> layer1-4 -> avgpool -> fc
        return nn.Sequential(*list(model.children())[:-2])  # up to layer4 output
    if name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        return nn.Sequential(*list(model.children())[:-1])  # drop final classifier
    raise ValueError(f"Unknown TRANSFER_BACKBONE: {name!r}")


class TransferModel(nn.Module):
    """Frozen backbone + fresh classification head.

    The backbone is built from ``config.TRANSFER_BACKBONE`` and its parameters
    are frozen (``requires_grad=False``) so only the new head trains. The head
    is a global-average-pool -> linear -> ReLU -> dropout -> linear stack that
    matches ``NUM_CLASSES``.
    """

    def __init__(self, num_classes: Optional[int] = None, freeze: bool = True):
        super().__init__()
        num_classes = num_classes or _num_classes()

        self.backbone = _build_backbone(config.TRANSFER_BACKBONE)
        self.freeze_backbone(freeze)

        # Infer feature dim by a dry forward on a dummy input.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE)
            feat = self.backbone(dummy)
            in_channels = feat.shape[1]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def freeze_backbone(self, freeze: bool) -> None:
        # requires_grad=True means trainable, so freeze=True -> requires_grad=False
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def build_model(name: str = "baseline", num_classes: Optional[int] = None) -> nn.Module:
    """Return the requested model by name: ``baseline`` or ``transfer``.

    Pass ``num_classes`` explicitly to avoid triggering a HF dataset download
    during model construction (the class count is otherwise inferred lazily).
    """
    name = name.lower()
    if name == "baseline":
        return SmallCNN(num_classes=num_classes)
    if name == "transfer":
        return TransferModel(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name!r} (expected 'baseline' or 'transfer')")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and inspect a model.")
    parser.add_argument("--model", choices=["baseline", "transfer"], default="baseline")
    args = parser.parse_args()

    model = build_model(args.model)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={args.model} params={n_params:,} trainable={n_train:,}")
    x = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    out = model(x)
    print(f"input={tuple(x.shape)} output={tuple(out.shape)}")