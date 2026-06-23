"""
app/inference.py — ML inference backend for the Streamlit app.

Design rule: zero Streamlit imports in this file. Every function here is
independently callable from a Python script, a Jupyter notebook, or a test.
app/main.py imports exclusively from this module for all ML operations.

Public API
----------
load_model(checkpoint_path)                     → (model, device)
preprocess_image(pil_image)                     → Tensor [1, 3, 224, 224]
predict(model, image_tensor, device)            → dict {class_name: probability}
generate_heatmap(model, pil_image, class_idx, device, alpha, colormap)
                                                → PIL Image (overlay)
get_top_prediction(predictions)                 → (class_name, probability)
load_evaluation_results(json_path)              → dict | None
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image

from src.config import CONFIG
from src.dataset import get_transforms
from src.gradcam import get_gradcam_overlay
from src.model import build_model, get_gradcam_target_layer
from src.utils import load_checkpoint

log = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str | Path,
) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load a trained EfficientNetB3 checkpoint and return the model in eval mode.

    Wrapped by @st.cache_resource in main.py — called once per Streamlit session,
    not once per uploaded image.

    For Hugging Face Spaces deployment, checkpoint_path should be the path
    returned by huggingface_hub.hf_hub_download().

    Args:
        checkpoint_path: Path to best_model.pth or last_model.pth.

    Returns:
        (model, device) — model is on device and in eval() mode.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading model from {checkpoint_path} on {device}")

    model = build_model(
        num_classes=CONFIG.NUM_CLASSES,
        pretrained=False,   # weights come from checkpoint, not ImageNet download
        dropout=CONFIG.DROPOUT,
    )

    load_checkpoint(path=Path(checkpoint_path), model=model, device=device)
    model.to(device)
    model.eval()

    log.info("Model loaded and ready for inference.")
    return model, device


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Apply the validation transform pipeline to a PIL Image.

    Converts grayscale to RGB (chest X-rays are often single-channel PNGs),
    applies Resize(256) → CenterCrop(224) → ToTensor → ImageNet normalisation.

    Returns:
        Tensor of shape [1, 3, 224, 224] (batch dimension added).
    """
    transform = get_transforms("val")
    rgb = pil_image.convert("RGB")
    tensor = transform(rgb)          # [3, 224, 224]
    return tensor.unsqueeze(0)       # [1, 3, 224, 224]


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run forward pass and return per-class probabilities.

    The model outputs a single logit for binary classification. Sigmoid converts
    this to P(PNEUMONIA). P(NORMAL) = 1 - P(PNEUMONIA).

    Args:
        model:        Loaded model from load_model() — must be in eval() mode.
        image_tensor: Output of preprocess_image() — shape [1, 3, 224, 224].
        device:       torch.device matching the model's device.

    Returns:
        Dict mapping class name → probability, e.g.:
        {"NORMAL": 0.23, "PNEUMONIA": 0.77}
        All values in [0, 1], all values sum to 1.0 (binary case).
    """
    image_tensor = image_tensor.to(device)
    logit = model(image_tensor)                        # [1, 1]
    pneumonia_prob = torch.sigmoid(logit).item()       # scalar
    normal_prob = 1.0 - pneumonia_prob

    return {
        CONFIG.CLASS_NAMES[0]: round(normal_prob, 4),      # NORMAL
        CONFIG.CLASS_NAMES[1]: round(pneumonia_prob, 4),   # PNEUMONIA
    }


def get_top_prediction(predictions: Dict[str, float]) -> Tuple[str, float]:
    """
    Return the class with the highest predicted probability.

    Args:
        predictions: Output of predict().

    Returns:
        (class_name, probability) e.g. ("PNEUMONIA", 0.77)
    """
    top_class = max(predictions, key=predictions.get)
    return top_class, predictions[top_class]


# ── Grad-CAM heatmap ──────────────────────────────────────────────────────────

def generate_heatmap(
    model: torch.nn.Module,
    pil_image: Image.Image,
    class_idx: int,
    device: torch.device,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Tuple[Image.Image, "np.ndarray"]:
    """
    Generate a Grad-CAM heatmap overlay for a given image and class.

    Grad-CAM requires a backward pass (gradients must flow), so this function
    does NOT use torch.no_grad(). The model is kept in eval() mode throughout
    so BatchNorm uses running statistics (not batch statistics) — this is
    correct for inference.

    Args:
        model:     Loaded model from load_model().
        pil_image: Original PIL Image (any size — resized internally).
        class_idx: 0 for binary classification (explains the single PNEUMONIA logit).
                   For 14-class extension: the pathology index to explain.
        device:    torch.device matching the model's device.
        alpha:     Heatmap opacity [0, 1]. 0.4 works well for clinical readability.
        colormap:  Matplotlib colormap name ("jet", "hot", "plasma", "inferno").

    Returns:
        (overlay_pil, cam_array)
        overlay_pil — PIL Image [224×224] of heatmap blended onto original X-ray.
        cam_array   — [224, 224] float32 numpy array in [0, 1] (raw heatmap values).
    """
    import numpy as np

    image_tensor = preprocess_image(pil_image).to(device)
    target_layer = get_gradcam_target_layer(model)

    cam, overlay_pil = get_gradcam_overlay(
        model=model,
        target_layer=target_layer,
        image_tensor=image_tensor,
        original_pil=pil_image,
        class_idx=class_idx,
        alpha=alpha,
        colormap=colormap,
    )
    return overlay_pil, cam


# ── Evaluation results loader ─────────────────────────────────────────────────

def load_evaluation_results(json_path: str | Path) -> Optional[Dict]:
    """
    Load pre-computed evaluation metrics from assets/evaluation_results.json.

    The Streamlit Model Performance page displays these instead of re-running
    inference on the test set every time (which would take ~30 seconds on CPU).

    Returns:
        Metrics dict, or None if the file does not exist (user has not yet
        run run_evaluation.py — the app shows a placeholder message instead).
    """
    import json
    path = Path(json_path)
    if not path.exists():
        log.warning(
            f"Evaluation results not found at {path}. "
            "Run: python run_evaluation.py --checkpoint checkpoints/best_model.pth"
        )
        return None
    with open(path) as f:
        return json.load(f)