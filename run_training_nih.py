"""
run_training_nih.py — CLI entry point for NIH ChestX-ray14 14-class training.

Usage:
    # Full Phase 1 + Phase 2
    python run_training_nih.py --data-root /content/data/nih_chestxray

    # Phase 1 only (gate check: mean AUC > 0.70)
    python run_training_nih.py --data-root /content/data/nih_chestxray --epochs2 0

    # Smoke test (1 epoch each, no pretrained, no wandb)
    python run_training_nih.py --data-root /content/data/nih_chestxray \\
        --epochs1 1 --epochs2 0 --no-wandb --no-pretrained

    # Focal loss (recommended for Pneumonia <2% and Hernia <0.2%)
    python run_training_nih.py --data-root /content/data/nih_chestxray --loss focal
"""
from __future__ import annotations
import argparse
import logging

from src.config_nih import CONFIG_NIH
from src.dataset_nih import get_dataloaders_nih
from src.losses import get_loss_fn
from src.model import build_model
from src.train_nih import run_training_nih
from src.utils import get_device, seed_everything, setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train EfficientNetB3 on NIH ChestX-ray14 (14-class multi-label)."
    )
    p.add_argument("--data-root", type=str, default=CONFIG_NIH.DATA_ROOT)
    p.add_argument("--checkpoint-dir", type=str, default=CONFIG_NIH.CHECKPOINT_DIR)
    p.add_argument("--loss", type=str, default="weighted_bce",
                   choices=["weighted_bce", "focal"],
                   help="Use focal for extreme class imbalance (Pneumonia 1.3%%, Hernia 0.2%%).")
    p.add_argument("--batch-size", type=int, default=CONFIG_NIH.BATCH_SIZE)
    p.add_argument("--epochs1", type=int, default=CONFIG_NIH.EPOCHS_PHASE1)
    p.add_argument("--epochs2", type=int, default=CONFIG_NIH.EPOCHS_PHASE2)
    p.add_argument("--seed", type=int, default=CONFIG_NIH.SEED)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    CONFIG_NIH.DATA_ROOT = args.data_root
    CONFIG_NIH.CHECKPOINT_DIR = args.checkpoint_dir
    CONFIG_NIH.BATCH_SIZE = args.batch_size
    CONFIG_NIH.EPOCHS_PHASE1 = args.epochs1
    CONFIG_NIH.EPOCHS_PHASE2 = args.epochs2
    CONFIG_NIH.SEED = args.seed
    if args.no_wandb:
        CONFIG_NIH.USE_WANDB = False

    seed_everything(CONFIG_NIH.SEED)
    device = get_device()

    log.info("Loading NIH ChestX-ray14 dataset...")
    loaders, pos_weight = get_dataloaders_nih(
        data_root=CONFIG_NIH.DATA_ROOT,
        batch_size=CONFIG_NIH.BATCH_SIZE,
    )
    pos_weight = pos_weight.to(device)

    log.info("Building EfficientNetB3 (num_classes=14)...")
    model = build_model(
        num_classes=CONFIG_NIH.NUM_CLASSES,       # 14 — key difference from binary
        pretrained=not args.no_pretrained,
        dropout=CONFIG_NIH.DROPOUT,
    )

    criterion = get_loss_fn(args.loss, pos_weight=pos_weight)
    criterion.to(device)

    log.info(f"Loss: {args.loss} | pos_weight shape: {pos_weight.shape}")

    history = run_training_nih(model, loaders, criterion, device, config=CONFIG_NIH)

    final_mean_auc = history["val_mean_auc"][-1] if history["val_mean_auc"] else float("nan")
    log.info(f"Final val mean AUC: {final_mean_auc:.4f}")
    log.info(
        "Next: python run_evaluation_nih.py "
        f"--checkpoint {CONFIG_NIH.CHECKPOINT_DIR}/best_model.pth"
    )


if __name__ == "__main__":
    main()
