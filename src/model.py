"""
model.py — EfficientNetB3 backbone + custom binary classification head.

build_model() loads ImageNet-pretrained EfficientNetB3 from torchvision and
replaces its 1000-class ImageNet classifier with a single-logit head suited
to binary PNEUMONIA vs NORMAL classification.

Architecture (confirmed against torchvision 0.27 at build time):
    model.features  — Sequential of 9 stages:
        [0] Conv2dNormActivation   stem,        3   →  40
        [1] Sequential             MBConv stage 40  →  24   (x2 blocks)
        [2] Sequential             MBConv stage 24  →  32   (x3 blocks)
        [3] Sequential             MBConv stage 32  →  48   (x3 blocks)
        [4] Sequential             MBConv stage 48  →  96   (x5 blocks)
        [5] Sequential             MBConv stage 96  → 136   (x5 blocks)
        [6] Sequential             MBConv stage 136 → 232   (x6 blocks)
        [7] Sequential             MBConv stage 232 → 384   (x2 blocks)
        [8] Conv2dNormActivation   head conv,  384  → 1536
    model.avgpool   — AdaptiveAvgPool2d(1), no parameters
    model.classifier — replaced below with Dropout + Linear(1536 → num_classes)

Two-phase freeze strategy (see freeze_backbone / unfreeze_top_layers):
    Phase 1: only model.classifier is trainable (stages 0–8 frozen).
    Phase 2: stages 5, 6, 7 (top 3 MBConv stages) PLUS stage 8 (head conv)
             are unfrozen, in addition to the classifier.

Design note on the head conv (stage 8): the blueprint's architecture diagram
marks blocks 1–4 as explicitly FROZEN and blocks 5–7 as explicitly UNFROZEN
in Phase 2, but leaves the head conv's frozen/unfrozen status unmarked.
Because it sits directly between the last MBConv stage and the classifier —
projecting 384 → 1536 channels — fine-tuning it alongside blocks 5–7 lets the
final feature representation adapt to chest X-ray patterns rather than
staying locked to generic ImageNet statistics. This is a deliberate choice;
if you want to mirror the blueprint literally, set UNFREEZE_BLOCKS so the
head conv stays excluded (not recommended — see unfreeze_top_layers docstring).
"""
from __future__ import annotations

import logging

import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from src.config import CONFIG

log = logging.getLogger(__name__)


# ── Model construction ────────────────────────────────────────────────────────

def build_model(
    num_classes: int = CONFIG.NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = CONFIG.DROPOUT,
) -> nn.Module:
    """
    Build EfficientNetB3 with a custom binary classification head.

    Args:
        num_classes: Output logits. 1 for binary (sigmoid), >1 for multi-label
                     (used later when extending to the full 14-class NIH set).
        pretrained:  If True, downloads ImageNet1K weights (requires network
                     access — works on Colab, may need --break-system-packages
                     network allowance in sandboxed/offline environments).
        dropout:     Dropout probability before the final Linear layer.

    Returns:
        nn.Module with model(x).shape == [batch, num_classes] (raw logits,
        no sigmoid applied — sigmoid is applied at inference/loss time).
    """
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights)

    # classifier[1].in_features is 1536 regardless of torchvision version —
    # reading it dynamically is safer than hardcoding BACKBONE_OUT_FEATURES.
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    log.info(
        f"Built EfficientNetB3 | pretrained={pretrained} | "
        f"num_classes={num_classes} | dropout={dropout} | "
        f"head: {in_features} → {num_classes}"
    )
    return model


# ── Freeze / unfreeze (two-phase training) ─────────────────────────────────────

def freeze_backbone(model: nn.Module) -> None:
    """
    Phase 1 setup: freeze every parameter except the classifier head.

    Prevents destroying pretrained ImageNet features before the randomly
    initialised head has learned anything useful — training the head against
    a moving backbone in epoch 1 produces noisy gradients that can corrupt
    otherwise-good pretrained weights.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    log.info("Backbone frozen — only classifier head is trainable (Phase 1)")


def unfreeze_top_layers(model: nn.Module, n_blocks: int = CONFIG.UNFREEZE_BLOCKS) -> None:
    """
    Phase 2 setup: unfreeze the top n_blocks MBConv stages, the head conv,
    and the classifier. Lower stages stay frozen — they learned universal
    low-level features (edges, textures) that don't need X-ray-specific
    adaptation.

    With n_blocks=3 (default), this unfreezes feature stages [5, 6, 7, 8]
    out of [0..8] — i.e. MBConv stages 5/6/7 plus the head conv (stage 8).
    See module docstring for why the head conv is included.

    Args:
        model:    Model previously passed through freeze_backbone().
        n_blocks: Number of top MBConv stages to unfreeze (not counting the
                  head conv, which is always unfrozen alongside them).

    Raises:
        ValueError: If n_blocks is too large for the number of feature stages.
    """
    feature_children = list(model.features.children())
    total_stages = len(feature_children)   # 9 for EfficientNetB3

    stage_start = total_stages - 1 - n_blocks   # -1 excludes head conv from the count
    if stage_start < 1:
        raise ValueError(
            f"n_blocks={n_blocks} is too large for {total_stages} feature "
            f"stages — at least the stem (stage 0) must stay frozen."
        )

    unfrozen_indices = []
    for idx, stage in enumerate(feature_children):
        if idx >= stage_start:
            for param in stage.parameters():
                param.requires_grad = True
            unfrozen_indices.append(idx)

    for param in model.classifier.parameters():
        param.requires_grad = True   # Idempotent — already trainable from Phase 1

    log.info(
        f"Unfroze feature stages {unfrozen_indices} "
        f"(top {n_blocks} MBConv blocks + head conv) + classifier (Phase 2)"
    )


# ── Grad-CAM support (used in src/gradcam.py, Week 4) ──────────────────────────

def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the last convolutional layer for Grad-CAM hook registration.

    This is model.features[-1] (the head conv, 384→1536 channels), which
    produces the [batch, 1536, 7, 7] feature map Grad-CAM needs — the same
    feature map that feeds AdaptiveAvgPool2d before the classifier.

    Returns here now so gradcam.py (Week 4) doesn't need to re-derive the
    architecture; defined alongside the model it targets.
    """
    return model.features[-1]