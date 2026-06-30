"""
train_nih.py — Training loops for NIH ChestX-ray14 14-class multi-label classification.

Key differences from train.py (binary):
    val_epoch_nih()  — returns per-class AUC [14] + mean AUC, not a scalar
    run_training_nih() — logs per-class AUC table every epoch, W&B per-class metrics

Everything else (train_epoch, AMP, gradient clipping, phase transitions) is
identical to binary training — so train_epoch() is imported directly from
train.py rather than duplicated here.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config_nih import CONFIG_NIH, ConfigNIH, NIH_CLASSES
from src.model import freeze_backbone, unfreeze_top_layers
from src.train import train_epoch      # reuse binary train_epoch — identical logic
from src.utils import count_parameters, print_auc_table, save_checkpoint

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── Multi-label validation epoch ──────────────────────────────────────────────

@torch.no_grad()
def val_epoch_nih(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[float]]:
    """
    Run one validation epoch for 14-class multi-label output.

    Per-class AUC:
        For each of the 14 pathology classes, collect predicted sigmoid
        probabilities and ground-truth binary labels, then compute AUC-ROC
        independently. Some rare classes (e.g. Hernia) may have no positive
        examples in a val batch — roc_auc_score is undefined in that case,
        so we return float('nan') for that class and exclude it from the mean.
        This is the correct handling — not reporting 0.5 or crashing.

    Returns:
        avg_loss      — float, mean BCE loss across the epoch
        mean_auc      — float, nanmean of per_class_auc (excludes nan)
        per_class_auc — List[float] of length 14, one AUC per NIH_CLASSES entry
    """
    model.eval()
    running_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)                     # [B, 14]
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    avg_loss = running_loss / len(loader.dataset)

    probs_matrix = torch.sigmoid(torch.cat(all_logits)).numpy()   # [N, 14]
    labels_matrix = torch.cat(all_labels).numpy()                  # [N, 14]

    per_class_auc: List[float] = []
    for class_idx in range(len(NIH_CLASSES)):
        class_probs = probs_matrix[:, class_idx]
        class_labels = labels_matrix[:, class_idx]

        if class_labels.sum() == 0 or class_labels.sum() == len(class_labels):
            # All same label — AUC undefined
            per_class_auc.append(float("nan"))
            log.debug(f"AUC undefined for {NIH_CLASSES[class_idx]} (single class in val)")
        else:
            auc = roc_auc_score(class_labels, class_probs)
            per_class_auc.append(float(auc))

    mean_auc = float(np.nanmean(per_class_auc))
    return avg_loss, mean_auc, per_class_auc


# ── Full two-phase orchestration ──────────────────────────────────────────────

def run_training_nih(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    config: ConfigNIH = CONFIG_NIH,
) -> Dict[str, list]:
    """
    Two-phase training for NIH 14-class multi-label classification.

    Identical phase logic to run_training() in train.py — freeze backbone,
    train head (Phase 1), unfreeze top blocks (Phase 2) — but validation
    now returns and logs 14 per-class AUCs instead of a single scalar.

    Args:
        model:     Fresh model from build_model(num_classes=14). Not pre-frozen.
        loaders:   Dict from get_dataloaders_nih() — "train", "val".
        criterion: BCEWithLogitsLoss(pos_weight=pos_weight_14.to(device)).
        device:    From utils.get_device().
        config:    CONFIG_NIH (or an override for testing).

    Returns:
        history dict: phase, epoch, train_loss, val_loss, val_mean_auc,
                      + val_auc_{class_name} for each of 14 classes.
    """
    model.to(device)

    history: Dict[str, list] = {
        "phase": [], "epoch": [], "train_loss": [], "val_loss": [],
        "val_mean_auc": [],
        **{f"val_auc_{cls}": [] for cls in NIH_CLASSES},
    }

    best_auc = 0.0
    global_epoch = 0

    use_wandb = config.USE_WANDB and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY or None,
            config={k: v for k, v in vars(config).items() if not k.startswith("_")},
        )

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
            log.info(f"=== Skipping {phase_name} (0 epochs) ===")
            continue

        log.info(f"=== Starting {phase_name} ({n_epochs} epochs, lr={lr}) ===")
        setup_fn(model)
        count_parameters(model)

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=config.WEIGHT_DECAY
        )

        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=config.ETA_MIN
            )

        scaler = torch.amp.GradScaler(
            device.type, enabled=config.USE_AMP and device.type == "cuda"
        )

        phase_best_auc = 0.0
        patience_counter = 0

        for epoch in range(n_epochs):
            t0 = time.time()

            train_loss = train_epoch(
                model, loaders["train"], criterion, optimizer, scaler, device,
                grad_clip=config.GRAD_CLIP, log_every=config.LOG_EVERY_N_BATCHES,
            )
            val_loss, mean_auc, per_class_auc = val_epoch_nih(
                model, loaders["val"], criterion, device
            )

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            log.info(
                f"[{phase_name}] epoch {epoch+1}/{n_epochs} | "
                f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"mean_auc {mean_auc:.4f} | {elapsed:.0f}s"
            )

            # Print per-class AUC table every epoch
            valid_aucs = [a if not np.isnan(a) else 0.0 for a in per_class_auc]
            print_auc_table(NIH_CLASSES, valid_aucs, mean_auc, epoch=global_epoch)

            # Update history
            history["phase"].append(phase_name)
            history["epoch"].append(global_epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mean_auc"].append(mean_auc)
            for cls, auc in zip(NIH_CLASSES, per_class_auc):
                history[f"val_auc_{cls}"].append(auc)

            if use_wandb:
                wandb.log({
                    "phase": phase_name,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/mean_auc": mean_auc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": global_epoch,
                    **{f"val/auc_{cls}": auc
                       for cls, auc in zip(NIH_CLASSES, per_class_auc)
                       if not np.isnan(auc)},
                })

            is_best = mean_auc > best_auc
            if is_best:
                best_auc = mean_auc

            if config.CHECKPOINT_EVERY and (epoch + 1) % config.CHECKPOINT_EVERY == 0:
                save_checkpoint(
                    model, optimizer, global_epoch, best_auc, config,
                    config.checkpoint_path, is_best=is_best, scaler=scaler,
                )

            if mean_auc > phase_best_auc:
                phase_best_auc = mean_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOP_PATIENCE:
                    log.info(
                        f"Early stopping {phase_name} at epoch {epoch+1} "
                        f"({config.EARLY_STOP_PATIENCE} epochs without improvement)"
                    )
                    global_epoch += 1
                    break

            global_epoch += 1

        log.info(f"=== {phase_name} complete | best mean AUC: {phase_best_auc:.4f} ===")

    if use_wandb:
        wandb.finish()

    log.info(f"NIH training complete. Best mean AUC: {best_auc:.4f}")
    return history
