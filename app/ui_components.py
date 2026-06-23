"""
app/ui_components.py — Reusable Streamlit UI building blocks.

All functions here take data as arguments and render Streamlit components.
They contain Streamlit code but no ML code — the split is:
    inference.py  → all ML logic (torch, numpy, PIL, gradcam)
    ui_components → all display logic (st.*, matplotlib, plotly)
    main.py       → wires them together (thin orchestration layer)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import streamlit as st
from PIL import Image

from src.config import CONFIG


# ── Header ────────────────────────────────────────────────────────────────────

def render_header() -> None:
    """App title and subtitle with a thin HTML separator."""
    st.markdown(
        """
        <h1 style='margin-bottom:0'>🫁 Chest X-Ray Anomaly Detector</h1>
        <p style='color:#64748b; font-size:1.05rem; margin-top:4px'>
          EfficientNetB3 · Grad-CAM Explainability · Binary Classification
        </p>
        <hr style='margin:12px 0 20px 0; border-color:#e2e8f0'>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> Dict:
    """
    Render all sidebar controls and return a dict of current values.

    Returns:
        {
          "threshold": float,   # PNEUMONIA detection threshold
          "alpha":     float,   # heatmap opacity
          "colormap":  str,     # matplotlib colormap name
          "page":      str,     # current page selection
        }
    """
    with st.sidebar:
        st.markdown("### 🔬 Navigation")
        page = st.radio(
            "Page",
            options=["Upload & Analyse", "Model Performance", "About"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Detection Settings")

        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0, max_value=1.0, value=CONFIG.THRESHOLD, step=0.05,
            help=(
                "Sigmoid probability above which PNEUMONIA is flagged. "
                "Lower = more sensitive (fewer missed cases, more false alarms). "
                "Higher = more specific (fewer false alarms, more missed cases). "
                "Default 0.5. Clinical screening typically uses 0.3–0.4."
            ),
        )

        st.markdown("### 🎨 Heatmap Settings")

        alpha = st.slider(
            "Heatmap Opacity",
            min_value=0.1, max_value=0.9, value=0.4, step=0.05,
            help="Blend ratio of heatmap over original X-ray. 0.4 is a good default.",
        )

        colormap = st.selectbox(
            "Heatmap Color",
            options=["jet", "hot", "plasma", "inferno", "viridis"],
            index=0,
            help=(
                "jet: blue→green→red (most common). "
                "hot: black→red→yellow. "
                "plasma/inferno: purple→orange (colorblind-friendly)."
            ),
        )

        st.markdown("---")
        st.markdown(
            "<small style='color:#94a3b8'>⚠️ For research use only. "
            "Not a medical diagnostic tool.</small>",
            unsafe_allow_html=True,
        )

    return {
        "threshold": threshold,
        "alpha": alpha,
        "colormap": colormap,
        "page": page,
    }


# ── Image display ─────────────────────────────────────────────────────────────

def render_image_pair(
    original_pil: Image.Image,
    heatmap_pil: Image.Image,
    top_class: str,
) -> None:
    """
    Side-by-side original X-ray and Grad-CAM overlay with captions.
    """
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(
            original_pil,
            caption="Original X-Ray",
            use_container_width=True,
        )
    with col2:
        st.image(
            heatmap_pil,
            caption=f"Grad-CAM — explaining: {top_class}",
            use_container_width=True,
        )


# ── Prediction bars ────────────────────────────────────────────────────────────

def render_prediction_bars(
    predictions: Dict[str, float],
    threshold: float,
) -> None:
    """
    Progress bars for each class probability, sorted high → low.
    Flagged classes (prob ≥ threshold) shown with a red DETECTED badge.
    """
    st.markdown("### Pathology Confidence Scores")

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    for class_name, prob in sorted_preds:
        detected = prob >= threshold
        col_label, col_bar = st.columns([1, 3], gap="small")

        with col_label:
            if detected:
                st.markdown(
                    f"<span style='color:#dc2626; font-weight:600'>"
                    f"🔴 {class_name}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='color:#475569'>{class_name}</span>",
                    unsafe_allow_html=True,
                )

        with col_bar:
            bar_color = "#dc2626" if detected else "#2563eb"
            pct = int(prob * 100)
            st.markdown(
                f"""
                <div style='display:flex; align-items:center; gap:8px; margin-top:4px'>
                  <div style='flex:1; background:#e2e8f0; border-radius:4px; height:20px; overflow:hidden'>
                    <div style='width:{pct}%; background:{bar_color}; height:100%;
                                border-radius:4px; transition:width 0.3s'></div>
                  </div>
                  <span style='min-width:40px; color:#374151; font-weight:600'>{pct}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)


# ── Class selector for heatmap ─────────────────────────────────────────────────

def render_class_selector(predictions: Dict[str, float]) -> int:
    """
    Dropdown to pick which class's Grad-CAM heatmap to display.

    Returns:
        class_idx: int (0 for binary; maps to CONFIG.CLASS_NAMES index)
    """
    class_names = list(predictions.keys())
    selected = st.selectbox(
        "Explain prediction for class:",
        options=class_names,
        index=0,
        help="Grad-CAM highlights the image regions that most influenced this class prediction.",
    )
    return class_names.index(selected)


# ── Model performance page ────────────────────────────────────────────────────

def render_model_performance(eval_results: Optional[Dict]) -> None:
    """
    Render the Model Performance page showing test-set metrics, ROC curve,
    and confusion matrix loaded from assets/evaluation_results.json.
    """
    st.markdown("## Model Performance — Test Set")

    if eval_results is None:
        st.info(
            "Evaluation results not found. Run:\n"
            "```\npython run_evaluation.py "
            "--checkpoint checkpoints/best_model.pth\n```\n"
            "Then relaunch the app."
        )
        return

    # ── Metric tiles ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    metric_style = "font-size:2rem; font-weight:700; color:#2563eb"

    with col1:
        st.markdown(f"<p style='{metric_style}'>{eval_results['auc']:.4f}</p>", unsafe_allow_html=True)
        st.caption("AUC-ROC ← Primary metric")
    with col2:
        st.markdown(f"<p style='{metric_style}'>{eval_results['recall']:.4f}</p>", unsafe_allow_html=True)
        st.caption("Recall / Sensitivity")
    with col3:
        st.markdown(f"<p style='{metric_style}'>{eval_results['f1']:.4f}</p>", unsafe_allow_html=True)
        st.caption("F1 Score")
    with col4:
        st.markdown(f"<p style='{metric_style}'>{eval_results['precision']:.4f}</p>", unsafe_allow_html=True)
        st.caption("Precision")

    st.markdown("---")

    # ── Confusion matrix image ─────────────────────────────────────────────────
    col_cm, col_roc = st.columns(2, gap="large")

    cm_path = Path("assets/confusion_matrix.png")
    roc_path = Path("assets/roc_curve.png")

    with col_cm:
        st.markdown("#### Confusion Matrix")
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            _render_confusion_matrix_from_dict(eval_results)

    with col_roc:
        st.markdown("#### ROC Curve")
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("Run run_evaluation.py to generate the ROC curve plot.")

    # ── Raw counts ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Test Set Breakdown")
    n_pos = eval_results.get("n_positive", "—")
    n_neg = eval_results.get("n_negative", "—")
    tp, tn = eval_results.get("TP", "—"), eval_results.get("TN", "—")
    fp, fn = eval_results.get("FP", "—"), eval_results.get("FN", "—")
    threshold = eval_results.get("threshold_used", CONFIG.THRESHOLD)
    best_t = eval_results.get("best_threshold", "—")

    st.markdown(
        f"**Samples:** {eval_results.get('n_samples', '—')} "
        f"({n_neg} NORMAL · {n_pos} PNEUMONIA)  |  "
        f"**Threshold used:** {threshold:.2f}  |  "
        f"**Best Youden threshold:** {best_t if isinstance(best_t, str) else f'{best_t:.2f}'}"
    )

    cols = st.columns(4)
    for col, label, val, color in [
        (cols[0], "True Positive", tp, "#16a34a"),
        (cols[1], "True Negative", tn, "#16a34a"),
        (cols[2], "False Positive", fp, "#dc2626"),
        (cols[3], "False Negative", fn, "#dc2626"),
    ]:
        with col:
            col.markdown(
                f"<div style='text-align:center'>"
                f"<p style='font-size:1.8rem; font-weight:700; color:{color}; margin:0'>{val}</p>"
                f"<p style='color:#64748b; font-size:0.85rem; margin:0'>{label}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<small style='color:#94a3b8'>CheXNet (Stanford 2017) benchmark: "
        "Pneumonia AUC = 0.768. This model uses a more parameter-efficient "
        "EfficientNetB3 backbone.</small>",
        unsafe_allow_html=True,
    )


def _render_confusion_matrix_from_dict(eval_results: Dict) -> None:
    """Fallback: render confusion matrix as a simple table when the PNG isn't available."""
    import pandas as pd
    classes = eval_results.get("class_names", CONFIG.CLASS_NAMES)
    cm = eval_results.get("confusion_matrix", [[0, 0], [0, 0]])
    df = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes],
                      columns=[f"Predicted {c}" for c in classes])
    st.dataframe(df, use_container_width=True)


# ── About page ────────────────────────────────────────────────────────────────

def render_about() -> None:
    """Static About / methodology page."""
    st.markdown("## About This Project")
    st.markdown("""
**Project 03 — Medical Image Anomaly Detector** is a deep learning portfolio project
that trains EfficientNetB3 on real chest X-ray data to classify NORMAL vs PNEUMONIA scans,
then wraps the model in this Streamlit app with Grad-CAM explainability.

### Architecture
- **Backbone**: EfficientNetB3 pretrained on ImageNet (12M parameters)
- **Head**: AdaptiveAvgPool → Dropout(0.4) → Linear(1536 → 1)
- **Training**: Two-phase transfer learning
  - Phase 1: Head-only training (5 epochs, LR=1e-3)
  - Phase 2: Fine-tune top 3 MBConv blocks (10 epochs, LR=1e-4)
- **Loss**: BCEWithLogitsLoss with inverse-frequency class weighting

### Data
- **Dataset**: Kaggle Chest X-Ray Images (Pneumonia) — 5,863 images
- **Split**: Stratified 80/20 train/val from merged train+val pool; fixed test set
- **Augmentation**: HorizontalFlip, Rotation±10°, ColorJitter (train only)

### Why Grad-CAM?
A model that says "78% PNEUMONIA" is a black box. Grad-CAM shows *which region
of the lung* drove that prediction — making the model auditable by a clinician.
In regulated medical AI, explainability is not optional.

### Metric choice
Accuracy is meaningless on this dataset (a model predicting NORMAL for everything
scores >50%). AUC-ROC measures ranking ability at all thresholds — the right
metric for imbalanced binary classification.

---
*Built with PyTorch · EfficientNetB3 · Grad-CAM · Streamlit*
""")


# ── Warning banner ─────────────────────────────────────────────────────────────

def render_disclaimer() -> None:
    """Persistent clinical disclaimer shown on the analysis page."""
    st.warning(
        "⚠️ **Research use only.** This tool is not a certified medical device and "
        "must not be used for clinical diagnosis. Always consult a qualified radiologist.",
        icon=None,
    )