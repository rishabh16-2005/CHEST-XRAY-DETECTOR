"""
run_evaluation.py — Evaluate a trained checkpoint on the test set.

Usage:
    python run_evaluation.py --checkpoint checkpoints/best_model.pth

    # Custom threshold (maximise recall for clinical setting)
    python run_evaluation.py --checkpoint checkpoints/best_model.pth --threshold 0.3

    # Save plots + JSON to a custom assets folder
    python run_evaluation.py --checkpoint checkpoints/best_model.pth --assets-dir my_assets/

Outputs (written to --assets-dir, default: assets/):
    confusion_matrix.png      — 2×2 seaborn heatmap
    roc_curve.png             — ROC curve with AUC annotated
    evaluation_results.json   — metrics dict for the Streamlit Model Performance page

Week 3 gate: test mean AUC > 0.82. Numbers ready for README.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.config import CONFIG
from src.dataset import get_dataloaders
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    print_metrics_report,
    save_evaluation_results,
)
from src.model import build_model
from src.utils import get_device, load_checkpoint, setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained EfficientNetB3 checkpoint on the held-out test set."
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best_model.pth or last_model.pth checkpoint file.",
    )
    p.add_argument(
        "--data-root", type=str, default=CONFIG.DATA_ROOT,
        help="Path to chest_xray/ folder. Defaults to CONFIG.DATA_ROOT.",
    )
    p.add_argument(
        "--threshold", type=float, default=CONFIG.THRESHOLD,
        help="Decision threshold for PNEUMONIA prediction (default: 0.5). "
             "Try 0.3 to maximise recall in a clinical screening context.",
    )
    p.add_argument(
        "--assets-dir", type=str, default="assets",
        help="Directory to save confusion_matrix.png, roc_curve.png, evaluation_results.json.",
    )
    p.add_argument(
        "--batch-size", type=int, default=CONFIG.BATCH_SIZE,
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    device = get_device()
    assets_dir = Path(args.assets_dir)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    log.info(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(num_classes=CONFIG.NUM_CLASSES, pretrained=False, dropout=CONFIG.DROPOUT)
    ckpt_meta = load_checkpoint(
        path=args.checkpoint,
        model=model,
        device=device,
    )
    log.info(f"Checkpoint: epoch={ckpt_meta['epoch'] + 1}, best val AUC={ckpt_meta['best_auc']:.4f}")
    model.to(device)
    model.eval()

    # ── Load test DataLoader only ────────────────────────────────────────────
    log.info("Loading test split...")
    loaders, _ = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
    )
    test_loader = loaders["test"]
    log.info(f"Test set: {len(test_loader.dataset)} images")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    log.info("Running inference on test set...")
    metrics = evaluate_model(model, test_loader, device, threshold=args.threshold)

    # ── Console report ───────────────────────────────────────────────────────
    print_metrics_report(metrics)

    # ── Plots ────────────────────────────────────────────────────────────────
    import numpy as np
    cm = np.array(metrics["confusion_matrix"])

    plot_confusion_matrix(
        cm=cm,
        classes=metrics["class_names"],
        save_path=assets_dir / "confusion_matrix.png",
        title=f"Confusion Matrix — Test Set (threshold={args.threshold:.2f})",
    )

    plot_roc_curve(
        fpr=metrics["fpr"],
        tpr=metrics["tpr"],
        roc_auc=metrics["auc"],
        save_path=assets_dir / "roc_curve.png",
        title=f"ROC Curve — PNEUMONIA vs NORMAL  (AUC = {metrics['auc']:.4f})",
    )

    # ── Save JSON for Streamlit ───────────────────────────────────────────────
    save_evaluation_results(
        metrics=metrics,
        path=assets_dir / "evaluation_results.json",
    )

    # ── Gate check ───────────────────────────────────────────────────────────
    print()
    if metrics["auc"] >= 0.82:
        print(f"✓  Week 3 gate CLEARED — test AUC {metrics['auc']:.4f} ≥ 0.82")
        print("   Numbers are ready for the README. Move to Week 4 (Grad-CAM).")
    else:
        print(f"✗  Week 3 gate NOT cleared — test AUC {metrics['auc']:.4f} < 0.82")
        print("   Consider: longer Phase 2 training, lower threshold, or focal loss.")

    print(f"\n   Plots saved to: {assets_dir.resolve()}/")


if __name__ == "__main__":
    main()
