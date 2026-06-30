"""
run_evaluation_nih.py — Evaluate NIH ChestX-ray14 14-class checkpoint on test set.

Usage:
    python run_evaluation_nih.py --checkpoint checkpoints/nih_best_model.pth

Saves to assets/nih/:
    roc_curves_14class.png        multi-line ROC (one per pathology)
    auc_comparison_chexnet.png    bar chart vs CheXNet benchmark
    evaluation_results_nih.json   metrics JSON for Streamlit Model Performance page

Week 3 gate (NIH): test mean AUC > 0.82 across all 14 classes.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.config_nih import CONFIG_NIH
from src.dataset_nih import get_dataloaders_nih
from src.evaluate_nih import (
    evaluate_model_nih,
    plot_auc_comparison,
    plot_roc_curves_nih,
    print_metrics_report_nih,
    save_evaluation_results_nih,
)
from src.model import build_model
from src.utils import get_device, load_checkpoint, setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default=CONFIG_NIH.DATA_ROOT)
    p.add_argument("--threshold", type=float, default=CONFIG_NIH.DEFAULT_THRESHOLD)
    p.add_argument("--assets-dir", type=str, default="assets/nih")
    p.add_argument("--batch-size", type=int, default=CONFIG_NIH.BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    device = get_device()
    assets = Path(args.assets_dir)

    model = build_model(num_classes=14, pretrained=False, dropout=CONFIG_NIH.DROPOUT)
    meta = load_checkpoint(path=args.checkpoint, model=model, device=device)
    log.info(f"Checkpoint: epoch={meta['epoch']+1}, best val AUC={meta['best_auc']:.4f}")
    model.to(device).eval()

    loaders, _ = get_dataloaders_nih(
        data_root=args.data_root, batch_size=args.batch_size
    )

    log.info("Running inference on NIH test set (25,596 images)...")
    metrics = evaluate_model_nih(model, loaders["test"], device, threshold=args.threshold)

    print_metrics_report_nih(metrics)

    plot_roc_curves_nih(metrics, save_path=assets / "roc_curves_14class.png")
    plot_auc_comparison(metrics, save_path=assets / "auc_comparison_chexnet.png")
    save_evaluation_results_nih(metrics, path=assets / "evaluation_results_nih.json")

    print()
    if metrics["mean_auc"] >= 0.82:
        print(f"✓  Mean AUC {metrics['mean_auc']:.4f} ≥ 0.82 — gate CLEARED")
    else:
        print(f"✗  Mean AUC {metrics['mean_auc']:.4f} < 0.82 — consider focal loss or more Phase 2 epochs")

    print(f"\n  Plots and JSON saved to: {assets.resolve()}/")


if __name__ == "__main__":
    main()
