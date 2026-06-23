"""
train.py — Training and validation loops, plus the full two-phase orchestration.

This module is backend-agnostic: it doesn't read CLI args or know about
argparse. That separation (handled by run_training.py) keeps train_epoch()
and val_epoch() independently testable and reusable directly from a Colab
notebook cell if you'd rather skip the CLI entirely.

Validation here computes loss + a single scalar AUC (ROC-AUC on the positive
PNEUMONIA class). That's enough to gate Phase 1 (target: val AUC > 0.70) and
Phase 2 (target: test AUC > 0.82) per the blueprint's weekly build gates.
Full per-class AUC tables, confusion matrices, and ROC curve plots arrive in
src/evaluate.py (Week 3) — this module deliberately doesn't duplicate that.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import CONFIG, Config
from src.model import freeze_backbone, unfreeze_top_layers
from src.utils import count_parameters, save_checkpoint

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    log.warning("wandb not installed — experiment tracking will be skipped even if USE_WANDB=True")


# ── Single-epoch loops ──────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    device: torch.device,
    grad_clip: float = CONFIG.GRAD_CLIP,
    log_every: int = CONFIG.LOG_EVERY_N_BATCHES,
) -> float:
    """
    Run one training epoch with mixed precision and gradient clipping.

    Mixed precision (autocast + GradScaler):
        Forward pass runs in float16 on CUDA — roughly 2x faster and half the
        VRAM of float32, which matters on a 16GB T4 with EfficientNetB3 at
        batch_size=32. GradScaler prevents float16 gradient underflow during
        the backward pass by dynamically scaling the loss before backward()
        and unscaling before the optimizer step.

    Gradient clipping:
        Phase 2 unfreezes backbone layers at a 10x lower LR than Phase 1, but
        large gradients can still occasionally spike (especially in the first
        few Phase 2 batches as the optimizer's momentum readjusts). Clipping
        to grad_clip=1.0 norm prevents a single bad batch from corrupting
        already-good pretrained weights.

    Returns:
        Average training loss across the full epoch (mean over all samples,
        not just batches — important when the last batch is smaller, though
        drop_last=True in our train DataLoader makes batches uniform anyway).
    """
    model.train()
    running_loss = 0.0
    n_batches = len(loader)
    use_amp = CONFIG.USE_AMP and device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)   # Required before clipping — clip on true-scale gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
            log.info(f"  batch {batch_idx + 1:4d}/{n_batches} | loss {loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run one validation epoch — no augmentation, no gradient computation.

    Computes AUC-ROC (not accuracy) because, per the blueprint, a model that
    predicts the majority class for every image still scores well on accuracy
    on imbalanced data; AUC-ROC measures ranking ability independent of any
    decision threshold.

    Returns:
        (avg_loss, auc) — auc is float('nan') if the validation batch happens
        to contain only one class (roc_auc_score is undefined in that case;
        this can occur with very small validation sets, so we warn instead
        of crashing the whole training run).
    """
    model.eval()
    running_loss = 0.0
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    avg_loss = running_loss / len(loader.dataset)

    probs = torch.sigmoid(torch.cat(all_logits)).numpy().ravel()
    labels_np = torch.cat(all_labels).numpy().ravel()

    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        log.warning(
            "Could not compute AUC — validation set contains only one class "
            "in this batch/split. Returning NaN for this epoch."
        )
        auc = float("nan")

    return avg_loss, auc


# ── Full two-phase orchestration ────────────────────────────────────────────────

def run_training(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    config: Config = CONFIG,
) -> Dict[str, list]:
    """
    Orchestrate the full two-phase training run: Phase 1 (head only) →
    Phase 2 (fine-tune top blocks). Handles checkpointing every epoch and
    per-phase early stopping; logs to Weights & Biases if config.USE_WANDB.

    Why two separate optimizers (one per phase) rather than one with a
    scheduled LR drop: each phase trains a different set of parameters
    (Phase 1: classifier only; Phase 2: classifier + top blocks). AdamW's
    per-parameter momentum and variance estimates from Phase 1 wouldn't be
    meaningful for the newly-unfrozen Phase 2 parameters anyway, so starting
    fresh is both simpler and correct.

    Args:
        model:     Freshly built model from build_model() — freeze/unfreeze
                   is applied internally per phase, do not pre-freeze it.
        loaders:   Dict with "train" and "val" DataLoaders (from
                   dataset.get_dataloaders()). "test" is unused here —
                   reserved for src/evaluate.py in Week 3.
        criterion: Loss module from losses.get_loss_fn(), already .to(device).
        device:    torch.device from utils.get_device().
        config:    CONFIG singleton (or an override for testing).

    Returns:
        history dict with keys: phase, epoch, train_loss, val_loss, val_auc
        — one entry per completed epoch across both phases. Useful for
        plotting training curves in the README later.
    """
    model.to(device)

    history: Dict[str, list] = {
        "phase": [], "epoch": [], "train_loss": [], "val_loss": [], "val_auc": [],
    }
    best_auc = 0.0
    global_epoch = 0
    checkpoint_dir = config.checkpoint_path

    use_wandb = config.USE_WANDB and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY or None,
            config={k: v for k, v in vars(config).items() if not k.startswith("_")},
        )

    # (phase_name, n_epochs, lr, setup_fn, use_scheduler)
    phases = [
        ("phase1", config.EPOCHS_PHASE1, config.LR_PHASE1, freeze_backbone, False),
        (
            "phase2",
            config.EPOCHS_PHASE2,
            config.LR_PHASE2,
            lambda m: unfreeze_top_layers(m, config.UNFREEZE_BLOCKS),
            True,
        ),
    ]

    for phase_name, n_epochs, lr, setup_fn, use_scheduler in phases:
        if n_epochs == 0:
            log.info(f"=== Skipping {phase_name} (0 epochs requested) ===")
            continue

        log.info(f"=== Starting {phase_name} ({n_epochs} epochs, lr={lr}) ===")
        setup_fn(model)
        count_parameters(model)   # Logs trainable vs frozen — confirms freeze worked

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=config.WEIGHT_DECAY)

        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=config.ETA_MIN
            )

        scaler = torch.amp.GradScaler(device.type, enabled=config.USE_AMP and device.type == "cuda")

        phase_best_auc = 0.0
        patience_counter = 0

        for epoch in range(n_epochs):
            t0 = time.time()

            train_loss = train_epoch(
                model, loaders["train"], criterion, optimizer, scaler, device,
                grad_clip=config.GRAD_CLIP, log_every=config.LOG_EVERY_N_BATCHES,
            )
            val_loss, val_auc = val_epoch(model, loaders["val"], criterion, device)

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            log.info(
                f"[{phase_name}] epoch {epoch + 1}/{n_epochs} | "
                f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"val_auc {val_auc:.4f} | {elapsed:.0f}s"
            )

            history["phase"].append(phase_name)
            history["epoch"].append(global_epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            if use_wandb:
                wandb.log({
                    "phase": phase_name,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/auc": val_auc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": global_epoch,
                })

            is_best = val_auc > best_auc   # NaN comparisons are False — safe, no crash
            if is_best:
                best_auc = val_auc

            if config.CHECKPOINT_EVERY and (epoch + 1) % config.CHECKPOINT_EVERY == 0:
                save_checkpoint(
                    model, optimizer, global_epoch, best_auc, config,
                    checkpoint_dir, is_best=is_best, scaler=scaler,
                )

            if val_auc > phase_best_auc:
                phase_best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOP_PATIENCE:
                    log.info(
                        f"Early stopping {phase_name} at epoch {epoch + 1} "
                        f"(no val AUC improvement for {config.EARLY_STOP_PATIENCE} epochs)"
                    )
                    global_epoch += 1
                    break

            global_epoch += 1

        log.info(f"=== {phase_name} complete | best val AUC this phase: {phase_best_auc:.4f} ===")

    if use_wandb:
        wandb.finish()

    log.info(f"Training complete. Best overall val AUC: {best_auc:.4f}")
    return history