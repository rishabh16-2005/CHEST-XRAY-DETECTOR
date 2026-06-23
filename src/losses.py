"""
losses.py — Loss functions for binary chest X-ray classification.

Two implementations are provided, both binary-ready (single sigmoid output):

  WeightedBCELoss — standard BCEWithLogitsLoss with pos_weight from class
                    frequency (computed in dataset.compute_pos_weight).
                    Use this first — simple, stable, and effective for our
                    moderate 26/74 imbalance (NORMAL/PNEUMONIA).

  FocalLoss       — down-weights easy/confident examples, forcing the model
                    to keep learning from hard or rare cases. Most useful
                    later when extending to the full 14-class NIH set, where
                    Pneumonia alone drops to <2% prevalence.

Both expect:
    logits  : [B, 1]  raw model output (no sigmoid applied)
    targets : [B, 1]  0.0 or 1.0

Use get_loss_fn(name, pos_weight) as the single entry point so run_training.py
doesn't need to know about both classes directly.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import CONFIG

log = logging.getLogger(__name__)


class WeightedBCELoss(nn.Module):
    """
    Thin wrapper around BCEWithLogitsLoss(pos_weight=...).

    Kept as its own class (rather than using BCEWithLogitsLoss directly in
    train.py) so get_loss_fn() has a uniform interface alongside FocalLoss,
    and so pos_weight moves correctly with .to(device) via the wrapped
    submodule's registered buffer.
    """

    def __init__(self, pos_weight: Optional[Tensor] = None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.bce(logits, targets)


class FocalLoss(nn.Module):
    """
    Binary focal loss: FL(p) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where p_t is the model's predicted probability for the TRUE class
    (not just the positive class) — this is what makes focal loss down-weight
    *easy* examples regardless of which class they belong to, rather than
    just down-weighting the majority class the way pos_weight does.

    gamma controls how aggressively easy examples are down-weighted:
        gamma=0   → identical to (alpha-weighted) BCE
        gamma=2.0 → standard starting point (Lin et al. 2017, RetinaNet paper)

    alpha balances overall positive/negative class weight, similar in spirit
    to pos_weight but applied multiplicatively per-sample rather than as a
    single scalar inside the BCE formula.

    Optionally pass pos_weight too — combines class-frequency weighting with
    focal down-weighting. Not necessary for the binary task here (the 26/74
    split isn't extreme), but available for the 14-class extension where
    Pneumonia alone is <2% of the NIH set.
    """

    def __init__(
        self,
        alpha: float = CONFIG.FOCAL_ALPHA,
        gamma: float = CONFIG.FOCAL_GAMMA,
        pos_weight: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # reduction="none" → per-sample loss, so we can apply focal weighting
        # before reducing to a scalar.
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)          # prob of the TRUE class
        focal_term = (1 - p_t).clamp(min=1e-6) ** self.gamma  # 0 for confident-correct, →1 for wrong
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss   # reduction == "none"


def get_loss_fn(name: str, pos_weight: Optional[Tensor] = None) -> nn.Module:
    """
    Factory — single entry point for selecting a loss function by name.

    Args:
        name:       "weighted_bce" (or "bce") or "focal".
        pos_weight: Scalar Tensor from dataset.compute_pos_weight(). Passed
                    through to either loss; ignored is never silent — both
                    losses accept it.

    Returns:
        An nn.Module loss — call criterion.to(device) before training.

    Raises:
        ValueError: If name is not a recognised loss.
    """
    key = name.lower()

    if key in ("weighted_bce", "bce"):
        log.info(f"Using WeightedBCELoss | pos_weight={pos_weight}")
        return WeightedBCELoss(pos_weight=pos_weight)

    if key == "focal":
        log.info(
            f"Using FocalLoss | alpha={CONFIG.FOCAL_ALPHA} | "
            f"gamma={CONFIG.FOCAL_GAMMA} | pos_weight={pos_weight}"
        )
        return FocalLoss(pos_weight=pos_weight)

    raise ValueError(f"Unknown loss name '{name}'. Use 'weighted_bce' or 'focal'.")