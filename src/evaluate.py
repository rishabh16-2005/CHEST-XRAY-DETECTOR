"""
evaluate.py — Evaluation pipeline for binary chest X-ray classification.

Separates metric computation (pure numpy/sklearn, no display dependencies)
from visualization (matplotlib/seaborn), so metric functions are independently
testable without a display backend.

Public API
----------
compute_all_metrics(probs, labels, threshold)  → metrics dict (no plots)
evaluate_model(model, loader, device)          → runs inference + metrics
plot_confusion_matrix(cm, classes, save_path)  → saves PNG figure
plot_roc_curve(fpr, tpr, auc, save_path)       → saves PNG figure
print_metrics_report(metrics)                  → formatted console output
save_evaluation_results(metrics, path)         → saves JSON for Streamlit app
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.utils import print_auc_table

log = logging.getLogger(__name__)


# ── Metric computation (no matplotlib) ────────────────────────────────────────

def compute_all_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = CONFIG.THRESHOLD,
    class_names: Optional[list] = None,
) -> Dict:
    """
    Compute the full set of evaluation metrics from raw probabilities.

    This function has zero side effects — no plots, no file writes.
    All visualization helpers call this first.

    Args:
        probs:       1-D float array of PNEUMONIA probabilities (sigmoid output).
        labels:      1-D int/float array of ground-truth labels (0=NORMAL, 1=PNEUMONIA).
        threshold:   Decision threshold: prob >= threshold → PNEUMONIA predicted.
        class_names: [NORMAL_name, PNEUMONIA_name]. Defaults to CONFIG.CLASS_NAMES.

    Returns:
        Dict with keys:
            auc         — scalar float
            fpr, tpr    — arrays for ROC curve
            threshold_used
            f1, precision, recall (sensitivity), specificity
            confusion_matrix    — 2×2 numpy array [[TN,FP],[FN,TP]]
            TP, TN, FP, FN      — raw counts
            n_positive, n_negative
            best_threshold      — threshold maximising Youden's J (sensitivity + specificity - 1)
    """
    if class_names is None:
        class_names = CONFIG.CLASS_NAMES

    probs = probs.ravel().astype(np.float32)
    labels = labels.ravel().astype(np.int32)

    # AUC-ROC
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")
        log.warning("AUC undefined — only one class present in labels.")

    fpr, tpr, thresholds = roc_curve(labels, probs)

    # Youden's J = sensitivity + specificity - 1 → optimal threshold
    youden_j = tpr - fpr
    best_thresh_idx = int(np.argmax(youden_j))
    best_threshold = float(thresholds[best_thresh_idx])

    # Threshold-based metrics
    preds = (probs >= threshold).astype(np.int32)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))         # sensitivity / true positive rate
    specificity = float(tn / (tn + fp + 1e-8))     # true negative rate
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    return {
        "auc": float(roc_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": thresholds.tolist(),
        "threshold_used": float(threshold),
        "best_threshold": best_threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,           # sensitivity
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
        "class_names": class_names,
    }


# ── Inference + metrics ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = CONFIG.THRESHOLD,
) -> Dict:
    """
    Run inference on all batches in loader, then compute metrics.

    Args:
        model:     Model in eval() mode (will be set here too — safe to call
                   from either train or eval mode).
        loader:    DataLoader (val or test — never train).
        device:    torch.device from utils.get_device().
        threshold: Decision threshold for PNEUMONIA. Default CONFIG.THRESHOLD=0.5.

    Returns:
        metrics dict (same as compute_all_metrics) plus:
            "probs"  — 1-D numpy array of predicted probabilities
            "labels" — 1-D numpy array of ground-truth labels
            "n_samples" — total images evaluated
    """
    model.eval()
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)                    # [B, 1]
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    probs = torch.sigmoid(torch.cat(all_logits)).numpy().ravel()
    labels_np = torch.cat(all_labels).numpy().ravel()

    metrics = compute_all_metrics(probs, labels_np, threshold=threshold)
    metrics["probs"] = probs.tolist()
    metrics["labels"] = labels_np.tolist()
    metrics["n_samples"] = len(probs)

    log.info(
        f"Evaluation on {metrics['n_samples']} samples → "
        f"AUC={metrics['auc']:.4f} | F1={metrics['f1']:.4f} | "
        f"Recall={metrics['recall']:.4f} | Precision={metrics['precision']:.4f}"
    )
    return metrics


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
) -> "matplotlib.figure.Figure":
    """
    Seaborn heatmap of the 2×2 confusion matrix.

    Annotates cells with both raw counts and percentages.
    Rows = actual class, columns = predicted class.

    Args:
        cm:        2×2 numpy array [[TN, FP], [FN, TP]] from sklearn.
        classes:   [NORMAL_name, PNEUMONIA_name].
        save_path: If provided, saves the figure as PNG.
        title:     Figure title (include split name, e.g. "Confusion Matrix — Test").

    Returns:
        matplotlib Figure (caller can call plt.show() or further customize).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_arr = np.array(cm)
    cm_pct = cm_arr.astype(float) / (cm_arr.sum() + 1e-8) * 100

    labels_annot = np.array([
        [f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
        for row_v, row_p in zip(cm_arr, cm_pct)
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=labels_annot,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Confusion matrix saved → {save_path}")

    return fig


def plot_roc_curve(
    fpr: list,
    tpr: list,
    roc_auc: float,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve — PNEUMONIA vs NORMAL",
) -> "matplotlib.figure.Figure":
    """
    ROC curve plot with AUC annotated and random-chance baseline.

    Args:
        fpr, tpr:  Lists from sklearn.metrics.roc_curve (or the metrics dict).
        roc_auc:   Scalar AUC to annotate on the plot.
        save_path: If provided, saves the figure as PNG.
        title:     Figure title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr, tpr,
        color="#2563EB",
        lw=2,
        label=f"EfficientNetB3 (AUC = {roc_auc:.4f})",
    )
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="#94A3B8",
        lw=1.2,
        label="Random chance (AUC = 0.50)",
    )

    # CheXNet benchmark for Pneumonia (AUC = 0.768)
    ax.axhline(y=0.768, color="#F59E0B", linestyle=":", lw=1.2,
               label="CheXNet Pneumonia benchmark (0.768)")

    ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=8)
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"ROC curve saved → {save_path}")

    return fig


# ── Console report ─────────────────────────────────────────────────────────────

def print_metrics_report(metrics: Dict) -> None:
    """
    Print a formatted evaluation report to stdout.

    Example:
        ══════════════════════════════════════════
        EVALUATION REPORT — PNEUMONIA vs NORMAL
        ══════════════════════════════════════════
        Samples          : 624  (NORMAL: 234 | PNEUMONIA: 390)
        Threshold        : 0.50  (best Youden: 0.42)
        ──────────────────────────────────────────
        AUC-ROC          : 0.9712   ← PRIMARY METRIC
        ──────────────────────────────────────────
        F1 (PNEUMONIA)   : 0.9247
        Precision        : 0.9311
        Recall/Sensitivity: 0.9185
        Specificity      : 0.9316
        ──────────────────────────────────────────
        Confusion Matrix:
          TP: 358   FP:  26
          FN:  32   TN: 218
        ══════════════════════════════════════════
    """
    classes = metrics.get("class_names", CONFIG.CLASS_NAMES)
    sep = "═" * 46
    thin = "─" * 46

    print(f"\n{sep}")
    print(f"  EVALUATION REPORT — {classes[1]} vs {classes[0]}")
    print(sep)
    n_pos, n_neg = metrics["n_positive"], metrics["n_negative"]
    print(f"  Samples           : {metrics['n_samples']}  ({classes[0]}: {n_neg} | {classes[1]}: {n_pos})")
    print(f"  Threshold used    : {metrics['threshold_used']:.2f}  (best Youden: {metrics['best_threshold']:.2f})")
    print(thin)
    print(f"  AUC-ROC           : {metrics['auc']:.4f}   ← PRIMARY METRIC")
    print(thin)
    print(f"  F1  ({classes[1]}) : {metrics['f1']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall/Sensitivity: {metrics['recall']:.4f}")
    print(f"  Specificity       : {metrics['specificity']:.4f}")
    print(thin)
    print(f"  Confusion Matrix:")
    print(f"    TP: {metrics['TP']:5d}   FP: {metrics['FP']:5d}")
    print(f"    FN: {metrics['FN']:5d}   TN: {metrics['TN']:5d}")
    print(sep)


# ── Persistence ────────────────────────────────────────────────────────────────

def save_evaluation_results(metrics: Dict, path: Path) -> None:
    """
    Serialise metrics to JSON so the Streamlit app can display them without
    running inference again (inference on the full test set takes ~30 seconds).

    Excludes large list fields (probs, labels, fpr, tpr) that would bloat the
    file — those are only needed for plotting, which is done here and saved
    as PNGs.

    Args:
        metrics: Dict from evaluate_model().
        path:    Output path (e.g. assets/evaluation_results.json).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep only scalars and small arrays for the JSON
    serialisable = {k: v for k, v in metrics.items()
                    if k not in ("probs", "labels", "fpr", "tpr", "roc_thresholds")}

    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info(f"Evaluation results saved → {path}")
