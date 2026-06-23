"""
run_training.py — CLI entry point for training the binary chest X-ray model.

Usage:
    # Default run (full Phase 1 + Phase 2, weighted BCE loss)
    python run_training.py --data-root /content/chest_xray

    # Try focal loss instead
    python run_training.py --data-root /content/chest_xray --loss focal

    # Quick smoke test — confirm the pipeline runs end-to-end before a full run
    python run_training.py --data-root /content/chest_xray --epochs1 1 --epochs2 1 --no-wandb

On Google Colab — mount Drive FIRST so checkpoints survive disconnects:
    from google.colab import drive
    drive.mount('/content/drive')

    !python run_training.py \\
        --data-root /content/chest_xray \\
        --checkpoint-dir /content/drive/MyDrive/checkpoints/chest_xray

Week 2 gate: Phase 1 complete, val mean AUC > 0.70.
"""
from __future__ import annotations

import argparse
import logging

from src.config import CONFIG
from src.dataset import get_dataloaders
from src.losses import get_loss_fn
from src.model import build_model
from src.train import run_training
from src.utils import get_device, seed_everything, setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train EfficientNetB3 on binary chest X-ray classification (NORMAL vs PNEUMONIA)."
    )
    p.add_argument(
        "--data-root", type=str, default=CONFIG.DATA_ROOT,
        help="Path to the chest_xray/ folder containing train/val/test subdirs.",
    )
    p.add_argument(
        "--checkpoint-dir", type=str, default=CONFIG.CHECKPOINT_DIR,
        help="Directory to save checkpoints. Use a Google Drive path on Colab.",
    )
    p.add_argument(
        "--loss", type=str, default="weighted_bce", choices=["weighted_bce", "focal"],
        help="Loss function. Start with weighted_bce; try focal if rare-class recall lags.",
    )
    p.add_argument("--batch-size", type=int, default=CONFIG.BATCH_SIZE)
    p.add_argument("--epochs1", type=int, default=CONFIG.EPOCHS_PHASE1, help="Phase 1 epochs (head only).")
    p.add_argument("--epochs2", type=int, default=CONFIG.EPOCHS_PHASE2, help="Phase 2 epochs (fine-tune).")
    p.add_argument("--seed", type=int, default=CONFIG.SEED)
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    p.add_argument(
        "--no-pretrained", action="store_true",
        help="Skip downloading ImageNet weights — random init. Only useful for fast pipeline smoke tests.",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    # Apply CLI overrides onto the CONFIG singleton — every downstream module
    # (dataset, model, train) reads from this same object.
    CONFIG.DATA_ROOT = args.data_root
    CONFIG.CHECKPOINT_DIR = args.checkpoint_dir
    CONFIG.BATCH_SIZE = args.batch_size
    CONFIG.EPOCHS_PHASE1 = args.epochs1
    CONFIG.EPOCHS_PHASE2 = args.epochs2
    CONFIG.SEED = args.seed
    if args.no_wandb:
        CONFIG.USE_WANDB = False

    log.info(f"Config: {CONFIG}")

    seed_everything(CONFIG.SEED)
    device = get_device()

    log.info("Loading data...")
    loaders, pos_weight = get_dataloaders(data_root=CONFIG.DATA_ROOT, batch_size=CONFIG.BATCH_SIZE)
    pos_weight = pos_weight.to(device)

    log.info("Building model...")
    model = build_model(
        num_classes=CONFIG.NUM_CLASSES,
        pretrained=not args.no_pretrained,
        dropout=CONFIG.DROPOUT,
    )
    model.to(device)

    criterion = get_loss_fn(args.loss, pos_weight=pos_weight)
    criterion.to(device)

    history = run_training(model, loaders, criterion, device, config=CONFIG)

    final_auc = history["val_auc"][-1] if history["val_auc"] else float("nan")
    log.info(f"Final validation AUC: {final_auc:.4f}")
    log.info(
        "Next: run_evaluation.py (Week 3) for full test-set AUC, F1, and "
        "confusion matrix — checkpoints/best_model.pth is ready to load."
    )


if __name__ == "__main__":
    main()