"""
app/main.py — Streamlit entry point.

Run with:
    streamlit run app/main.py

This file is intentionally thin. All ML logic lives in app/inference.py,
all UI building blocks in app/ui_components.py. This file only wires them.

Page structure (controlled by sidebar):
    Upload & Analyse  — upload X-ray → predict → show heatmap
    Model Performance — test-set metrics, ROC curve, confusion matrix
    About             — architecture, data, methodology
"""
import logging
from pathlib import Path

import re
import streamlit as st
from PIL import Image

# ── Page config — MUST be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="THORAEXPLAIN",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.inference import (          # noqa: E402 — must come after set_page_config
    generate_heatmap,
    get_top_prediction,
    load_evaluation_results,
    load_model,
    predict,
    preprocess_image,
)
from app.ui_components import (      # noqa: E402
    render_about,
    render_class_selector,
    render_disclaimer,
    render_header,
    render_image_pair,
    render_model_performance,
    render_prediction_bars,
    render_sidebar,
)
from src.config_nih import CONFIG_NIH as CONFIG

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("all_outputs/checkpoints/nih_chestxray/best_model.pth")
EVAL_RESULTS_PATH = Path("assets/evaluation_results.json")
SAMPLE_DIR = Path("assets/sample_xrays")


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights...")
def get_model():
    """
    Load the model once per Streamlit server process and cache it.
    @st.cache_resource keeps a single model instance across all user sessions —
    correct for a stateless inference model.

    If the checkpoint doesn't exist yet (user hasn't trained), returns None
    so the app can display a helpful message instead of crashing.
    """
    if not CHECKPOINT_PATH.exists():
        return None, None
    return load_model(CHECKPOINT_PATH)


@st.cache_data(show_spinner=False)
def get_eval_results():
    """Load evaluation JSON once per session."""
    return load_evaluation_results(EVAL_RESULTS_PATH)


@st.cache_data(show_spinner=False)
def get_sample_images():
    """Return a dict of {display_name: PIL Image} for sidebar sample buttons."""
    if not SAMPLE_DIR.exists():
        return {}
    samples = {}
    for p in sorted(SAMPLE_DIR.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            samples[p.stem.replace("_", " ").title()] = Image.open(p).convert("RGB")
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Inject custom CSS
    css_path = Path("app/styles.css")
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)  # strip comments
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True) 
    render_header()
    controls = render_sidebar()
    page = controls["page"]

    # ── About page ────────────────────────────────────────────────────────────
    if page == "About":
        render_about()
        return

    # ── Model Performance page ────────────────────────────────────────────────
    if page == "Model Performance":
        eval_results = get_eval_results()
        render_model_performance(eval_results)
        return

    # ── Upload & Analyse page ─────────────────────────────────────────────────
    model, device = get_model()

    if model is None:
        st.error(
            f"No trained checkpoint found at `{CHECKPOINT_PATH}`.\n\n"
            "Train the model first:\n"
            "```\npython run_training.py --data-root /path/to/chest_xray\n```\n"
            "Then relaunch the app once `checkpoints/best_model.pth` exists."
        )
        return

    # ── File upload + sample image selector ──────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    sample_images = get_sample_images()
    if sample_images:
        st.markdown("**Or try a sample image:**")
        sample_cols = st.columns(len(sample_images))
        for col, (name, img) in zip(sample_cols, sample_images.items()):
            with col:
                if st.button(name, use_container_width=True):
                    st.session_state["sample_image"] = img
                    st.session_state["sample_name"] = name

    # Resolve which image to use: uploaded > sample button > nothing
    pil_image = None
    image_source = None

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_source = uploaded_file.name
        # Clear any cached sample so uploaded file takes precedence
        st.session_state.pop("sample_image", None)
    elif "sample_image" in st.session_state:
        pil_image = st.session_state["sample_image"]
        image_source = st.session_state.get("sample_name", "Sample image")

    if pil_image is None:
        st.markdown(
            "<div style='text-align:center; color:#94a3b8; padding:60px 0'>"
            "<p style='font-size:3rem'>🩻</p>"
            "<p>Upload a chest X-ray above to get started.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Inference ─────────────────────────────────────────────────────────────
    with st.spinner("Analysing X-ray..."):
        image_tensor = preprocess_image(pil_image)
        predictions = predict(model, image_tensor, device)
        top_class, top_prob = get_top_prediction(predictions)

        # class_idx is always 0 for binary (the single PNEUMONIA logit)
        class_idx = controls.get("class_idx", 0)
        heatmap_pil, cam_array = generate_heatmap(
            model=model,
            pil_image=pil_image,
            class_idx=class_idx,
            device=device,
            alpha=controls["alpha"],
            colormap=controls["colormap"],
        )

    # ── Results header ────────────────────────────────────────────────────────
    st.markdown(f"**Source:** {image_source}")

    # NIH multi-label: a scan is "normal" when nothing clears the threshold
    detected_classes = [
        cls for cls, prob in predictions.items()
        if prob >= controls["threshold"]
    ]
    detected_classes.sort(key=lambda c: predictions[c], reverse=True)

    if detected_classes:
        top_detected      = detected_classes[0]
        top_detected_prob = predictions[top_detected]
        badge_color       = "#dc2626"
        is_normal         = False

        if len(detected_classes) == 1:
            badge_text = (
                f"🔴 {top_detected.upper()} DETECTED "
                f"— {top_detected_prob:.1%} confidence"
            )
        elif len(detected_classes) <= 3:
            others     = ", ".join(detected_classes[1:])
            badge_text = (
                f"🔴 {top_detected.upper()} DETECTED "
                f"— {top_detected_prob:.1%}  |  also: {others}"
            )
        else:
            badge_text = (
                f"🔴 {top_detected.upper()} DETECTED "
                f"— {top_detected_prob:.1%}  |  +{len(detected_classes)-1} other findings"
            )

        # class_idx for the initial heatmap = top detected pathology
        class_idx = list(predictions.keys()).index(top_detected)
        top_class = top_detected   # keep top_class consistent for render_image_pair

    else:
        top_class, top_prob = get_top_prediction(predictions)
        badge_color = "#16a34a"
        badge_text  = (
            f"🟢 NO FINDINGS DETECTED "
            f"— highest: {top_class} {top_prob:.1%} "
            f"(threshold: {controls['threshold']:.0%})"
        )
        is_normal = True
        class_idx = list(predictions.keys()).index(top_class)

    # ── THIS WAS THE MISSING LINE — badge was computed but never shown ─────────
    st.markdown(
        f"<div style='"
        f"background:{badge_color}22; border-left:4px solid {badge_color}; "
        f"padding:12px 16px; border-radius:6px; margin-bottom:12px; "
        f"font-size:1.05rem; font-weight:600; color:{badge_color}'>"
        f"{badge_text}</div>",
        unsafe_allow_html=True,
    )

    # ── Image pair + confidence bars ──────────────────────────────────────────
    render_image_pair(pil_image, heatmap_pil, top_class)
    render_prediction_bars(predictions, threshold=controls["threshold"])

    # ── Class selector for heatmap re-generation ──────────────────────────────
    with st.expander("🔁 Change which class the heatmap explains"):

        if is_normal:
            # Grad-CAM on a normal scan is meaningless — no class to back-propagate
            st.info(
                "No pathology detected above the threshold — "
                "Grad-CAM requires a positive prediction to explain.\n\n"
                "Lower the threshold in the sidebar to explore the scan anyway."
            )
        else:
            new_class_idx = render_class_selector(predictions)
            if new_class_idx != class_idx:
                with st.spinner("Regenerating heatmap..."):
                    heatmap_pil, _ = generate_heatmap(
                        model=model,
                        pil_image=pil_image,
                        class_idx=new_class_idx,
                        device=device,
                        alpha=controls["alpha"],
                        colormap=controls["colormap"],
                    )
                selected_class = CONFIG.CLASS_NAMES[new_class_idx]
                st.image(
                    heatmap_pil,
                    caption=f"Grad-CAM for class: {selected_class}",
                    use_container_width=True,
                )

    render_disclaimer()


if __name__ == "__main__":
    main()