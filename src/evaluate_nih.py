"""
evaluate_nih.py — Evaluation pipeline for NIH ChestX-ray14 14-class classification.

Extends evaluate.py (binary) to handle multi-label output:
    per-class AUC for all 14 pathologies
    mean AUC across classes (primary metric)
    per-class F1, precision, recall at default threshold
    multi-line ROC curve plot (one curve per class, colour-coded)
    comparison table against CheXNet Stanford 2017 benchmark

Public API
----------
evaluate_model_nih(model, loader, device)  → metrics dict
compute_all_metrics_nih(probs, labels)     → metrics dict
plot_roc_curves_nih(metrics, save_path)    → saves multi-line ROC plot
plot_auc_comparison(metrics, save_path)    → saves bar chart vs CheXNet
print_metrics_report_nih(metrics)          → formatted console output
save_evaluation_results_nih(metrics, path) → JSON for Streamlit
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader

from src.config_nih import CONFIG_NIH, NIH_CLASSES

log = logging.getLogger(__name__)

# ── CheXNet Stanford 2017 benchmark AUCs ─────────────────────────────────────
# Source: Rajpurkar et al. 2017 "CheXNet: Radiologist-Level Pneumonia Detection"
# DenseNet121, full NIH dataset. Use these as your comparison target.
CHEXNET_AUCS: Dict[str, float] = {
    "Atelectasis": 0.8094,
    "Cardiomegaly": 0.9248,
    "Effusion": 0.8638,
    "Infiltration": 0.7345,
    "Mass": 0.8676,
    "Nodule": 0.7802,
    "Pneumonia": 0.7680,
    "Pneumothorax": 0.8887,
    "Consolidation": 0.7901,
    "Edema": 0.8878,
    "Emphysema": 0.9371,
    "Fibrosis": 0.8047,
    "Pleural_Thickening": 0.8062,
    "Hernia": 0.9164,
}
CHEXNET_MEAN_AUC = float(np.mean(list(CHEXNET_AUCS.values())))


# ── Metric computation ────────────────────────────────────────────────────────

def compute_all_metrics_nih(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = CONFIG_NIH.DEFAULT_THRESHOLD,
) -> Dict:
    """
    Compute per-class and aggregate metrics for 14-class multi-label output.

    Args:
        probs:     [N, 14] float array of sigmoid probabilities.
        labels:    [N, 14] float array of ground-truth multi-hot labels.
        threshold: Decision threshold for binary prediction per class (default 0.5).

    Returns:
        metrics dict with:
            per_class_auc    — dict {class_name: auc_float}
            mean_auc         — nanmean across 14 classes
            per_class_f1     — dict {class_name: f1_float}
            per_class_recall — dict {class_name: recall_float}
            per_class_precision — dict {class_name: precision_float}
            fpr_tpr          — dict {class_name: {"fpr": list, "tpr": list}}
            chexnet_aucs     — benchmark comparison dict
            threshold_used
    """
    per_class_auc: Dict[str, float] = {}
    per_class_f1: Dict[str, float] = {}
    per_class_recall: Dict[str, float] = {}
    per_class_precision: Dict[str, float] = {}
    fpr_tpr: Dict[str, Dict] = {}

    for i, cls in enumerate(NIH_CLASSES):
        class_probs = probs[:, i]
        class_labels = labels[:, i].astype(int)
        class_preds = (class_probs >= threshold).astype(int)

        # AUC — undefined if only one class present in labels
        if class_labels.sum() == 0 or class_labels.sum() == len(class_labels):
            per_class_auc[cls] = float("nan")
            fpr_tpr[cls] = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}   # diagonal
            log.warning(f"AUC undefined for {cls} — only one class in eval set.")
        else:
            per_class_auc[cls] = float(roc_auc_score(class_labels, class_probs))
            fpr, tpr, _ = roc_curve(class_labels, class_probs)
            fpr_tpr[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

        # Threshold-based metrics (handle zero-division)
        per_class_f1[cls] = float(
            f1_score(class_labels, class_preds, zero_division=0)
        )
        per_class_recall[cls] = float(
            recall_score(class_labels, class_preds, zero_division=0)
        )
        per_class_precision[cls] = float(
            precision_score(class_labels, class_preds, zero_division=0)
        )

    valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    return {
        "per_class_auc": per_class_auc,
        "mean_auc": mean_auc,
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "fpr_tpr": fpr_tpr,
        "chexnet_aucs": CHEXNET_AUCS,
        "chexnet_mean_auc": CHEXNET_MEAN_AUC,
        "threshold_used": threshold,
        "n_samples": len(probs),
        "class_names": NIH_CLASSES,
    }


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model_nih(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = CONFIG_NIH.DEFAULT_THRESHOLD,
) -> Dict:
    """
    Run inference on a DataLoader and return full 14-class metrics.

    Returns:
        compute_all_metrics_nih(...) result plus "probs" and "labels" arrays.
    """
    model.eval()
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)                         # [B, 14]
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    probs = torch.sigmoid(torch.cat(all_logits)).numpy()   # [N, 14]
    labels_np = torch.cat(all_labels).numpy()               # [N, 14]

    metrics = compute_all_metrics_nih(probs, labels_np, threshold=threshold)
    metrics["probs"] = probs.tolist()
    metrics["labels"] = labels_np.tolist()

    log.info(
        f"NIH evaluation: {metrics['n_samples']} samples | "
        f"mean AUC = {metrics['mean_auc']:.4f} | "
        f"CheXNet mean = {CHEXNET_MEAN_AUC:.4f}"
    )
    return metrics


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_roc_curves_nih(
    metrics: Dict,
    save_path: Optional[Path] = None,
    highlight_classes: Optional[List[str]] = None,
) -> "matplotlib.figure.Figure":
    """
    Multi-line ROC curve: one curve per pathology class, coloured by AUC rank.
    Adds CheXNet benchmark AUCs as horizontal dashed reference lines.

    Args:
        highlight_classes: If provided, only these classes are plotted in full
                           colour; others are grey. Useful for dense subplots.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(9, 8))

    # Sort classes by AUC descending for legend readability
    sorted_classes = sorted(
        NIH_CLASSES,
        key=lambda c: metrics["per_class_auc"].get(c, 0.0),
        reverse=True,
    )

    colours = cm.tab20(np.linspace(0, 1, len(NIH_CLASSES)))

    for colour, cls in zip(colours, sorted_classes):
        auc = metrics["per_class_auc"].get(cls, float("nan"))
        fpr = metrics["fpr_tpr"][cls]["fpr"]
        tpr = metrics["fpr_tpr"][cls]["tpr"]

        if highlight_classes and cls not in highlight_classes:
            ax.plot(fpr, tpr, color="lightgrey", lw=0.8, alpha=0.5)
        else:
            label = f"{cls} ({auc:.3f})"
            if not np.isnan(auc):
                chexnet_auc = CHEXNET_AUCS.get(cls)
                delta = auc - chexnet_auc if chexnet_auc else 0
                arrow = "↑" if delta >= 0 else "↓"
                label += f" {arrow}{abs(delta):.3f} vs CheXNet"
            ax.plot(fpr, tpr, color=colour, lw=1.5, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC Curves — 14 NIH Pathologies\n"
        f"Mean AUC: {metrics['mean_auc']:.4f} | CheXNet: {CHEXNET_MEAN_AUC:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=7.5, ncol=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"ROC curves saved → {save_path}")
    return fig


def plot_auc_comparison(
    metrics: Dict,
    save_path: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """
    Grouped bar chart comparing your per-class AUC to CheXNet benchmark.
    Bars where you beat CheXNet are green; below are orange.
    """
    import matplotlib.pyplot as plt

    classes = NIH_CLASSES
    your_aucs = [metrics["per_class_auc"].get(c, 0.0) for c in classes]
    chex_aucs = [CHEXNET_AUCS.get(c, 0.0) for c in classes]
    deltas = [y - cx for y, cx in zip(your_aucs, chex_aucs)]

    x = np.arange(len(classes))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={"height_ratios": [3, 1]})

    bars_yours = ax1.bar(x - width/2, your_aucs, width,
                         label="Your EfficientNetB3", color="#2563EB", alpha=0.85)
    bars_chex = ax1.bar(x + width/2, chex_aucs, width,
                        label="CheXNet (Stanford 2017, DenseNet121)",
                        color="#94A3B8", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("AUC-ROC", fontsize=11)
    ax1.set_ylim([0.5, 1.0])
    ax1.set_title(
        f"Per-Class AUC: EfficientNetB3 vs CheXNet\n"
        f"Your mean AUC: {metrics['mean_auc']:.4f}  |  CheXNet mean: {CHEXNET_MEAN_AUC:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(y=metrics["mean_auc"], color="#2563EB", ls="--", lw=1.2,
                label=f"Your mean ({metrics['mean_auc']:.3f})")
    ax1.axhline(y=CHEXNET_MEAN_AUC, color="#94A3B8", ls="--", lw=1.2)

    colours = ["#16a34a" if d >= 0 else "#dc2626" for d in deltas]
    ax2.bar(x, deltas, color=colours, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Δ AUC vs CheXNet", fontsize=10)
    ax2.set_title("AUC difference (green = above CheXNet)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"AUC comparison chart saved → {save_path}")
    return fig


# ── Console report ─────────────────────────────────────────────────────────────

def print_metrics_report_nih(metrics: Dict) -> None:
    """Print a formatted per-class AUC table with CheXNet comparison."""
    sep = "═" * 74
    thin = "─" * 74

    print(f"\n{sep}")
    print(f"  NIH ChestX-ray14 EVALUATION REPORT   ({metrics['n_samples']} test samples)")
    print(sep)
    print(f"  {'Class':<22} {'Your AUC':>9}  {'CheXNet':>9}  {'Δ':>7}  {'Recall':>8}  {'F1':>7}")
    print(thin)

    for cls in NIH_CLASSES:
        your = metrics["per_class_auc"].get(cls, float("nan"))
        chex = CHEXNET_AUCS.get(cls, float("nan"))
        delta = your - chex if not np.isnan(your) else float("nan")
        recall = metrics["per_class_recall"].get(cls, float("nan"))
        f1 = metrics["per_class_f1"].get(cls, float("nan"))

        arrow = ("↑" if delta >= 0 else "↓") if not np.isnan(delta) else " "
        delta_str = f"{arrow}{abs(delta):.4f}" if not np.isnan(delta) else "   n/a"

        print(
            f"  {cls:<22} {your:>9.4f}  {chex:>9.4f}  {delta_str:>7}  "
            f"{recall:>8.4f}  {f1:>7.4f}"
        )

    print(thin)
    chex_mean = CHEXNET_MEAN_AUC
    your_mean = metrics["mean_auc"]
    delta_mean = your_mean - chex_mean
    arrow = "↑" if delta_mean >= 0 else "↓"
    print(
        f"  {'MEAN AUC':<22} {your_mean:>9.4f}  {chex_mean:>9.4f}  "
        f"{arrow}{abs(delta_mean):.4f}"
    )
    print(sep)
    print(f"\n  Threshold: {metrics['threshold_used']:.2f}")
    print(f"\n  Interview quote:")
    print(f'  "My mean AUC of {your_mean:.4f} is {"competitive with" if your_mean >= chex_mean else "close to"}')
    print(f'   CheXNet ({chex_mean:.4f}) using a more parameter-efficient EfficientNetB3."')
    print()


# ── Persistence ────────────────────────────────────────────────────────────────

def save_evaluation_results_nih(metrics: Dict, path: Path) -> None:
    """Save metrics to JSON (excludes large probs/labels/fpr_tpr arrays)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: v for k, v in metrics.items()
                    if k not in ("probs", "labels", "fpr_tpr")}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info(f"NIH evaluation results saved → {path}")
