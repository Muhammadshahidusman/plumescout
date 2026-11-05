"""U-Net plume segmenter (TRD 5.4): ResNet-34 encoder, ImageNet weights, N input channels.

smp adapts the first convolution to `in_channels` by reusing the pretrained RGB filters.
Output: one logit map per image, same H x W as the input (H, W multiples of 32).
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_unet(
    encoder: str, encoder_weights: str | None, in_channels: int, classes: int
) -> nn.Module:
    """Return an smp U-Net."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


class BceDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice (binary). Plumes are under 5% of pixels."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.wb, self.wd = bce_weight, dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.wb * self.bce(logits, target) + self.wd * self.dice(logits, target)
