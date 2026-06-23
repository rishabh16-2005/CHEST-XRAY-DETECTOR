"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM).

Implements the four-step algorithm from Selvaraju et al. (2017):
  1. Forward pass → capture feature maps at the target conv layer
  2. Backward pass on the target class score → capture gradients
  3. Global-average-pool the gradients (per-channel importance weights)
     then compute the weighted sum of feature maps
  4. ReLU → upsample to input resolution → normalise to [0, 1]

Binary classification note:
  The model has a single output logit (num_classes=1). Grad-CAM on binary
  classification is called with class_idx=0 always — it explains "what made
  the model predict PNEUMONIA?" whether or not the prediction is high.
  When extending to 14-class NIH, pass class_idx = pathology index.

Why the head conv (features[-1]) is the right target layer:
  The final feature map before AdaptiveAvgPool2d has shape [B, 1536, 7, 7].
  It captures the highest-level semantics the model has learned — at 224×224
  input, 7×7 spatial resolution means each cell covers 32×32 pixels, which
  is sufficient granularity to localise lung regions in chest X-rays.
  Earlier layers have finer spatial resolution but lower semantic level;
  the resulting heatmaps are noisier and harder for clinicians to interpret.

Hook management:
  Both forward and backward hooks are registered in __init__ and stored in
  _handles. Call remove_hooks() or use as a context manager to clean up —
  failing to remove hooks causes memory leaks and incorrect gradients if
  multiple GradCAM instances share the same layer.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

log = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM for any PyTorch model with a spatial convolutional target layer.

    Usage (binary):
        target_layer = model.features[-1]
        grad_cam = GradCAM(model, target_layer)

        image_tensor = preprocess(pil_image).unsqueeze(0)   # [1, 3, 224, 224]
        cam = grad_cam.generate(image_tensor, class_idx=0)  # [224, 224], values in [0, 1]

        heatmap_pil = overlay_heatmap(cam, pil_image, alpha=0.4)
        grad_cam.remove_hooks()   # always clean up

    Usage as context manager:
        with GradCAM(model, target_layer) as gc:
            cam = gc.generate(image_tensor, class_idx=0)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles = []

        def _save_activation(module, inp, output):
            # Store the raw tensor (not detached) so the computational graph
            # is preserved for the backward hook. We detach for the Grad-CAM
            # computation in generate() after backward() has already fired.
            self._activations = output

        def _save_gradient(module, grad_input, grad_output):
            # grad_output[0] is the gradient of the loss w.r.t. the layer's output
            self._gradients = grad_output[0].detach()

        self._handles.append(
            target_layer.register_forward_hook(_save_activation)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(_save_gradient)
        )

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int = 0,
        target_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single image and target class.

        Args:
            image_tensor: Preprocessed image tensor of shape [1, 3, H, W].
                          Must NOT be under torch.no_grad() context.
            class_idx:    Index of the class to explain. 0 for binary (the
                          single PNEUMONIA logit). For 14-class extension,
                          pass the pathology index (e.g. 5 for Pneumonia).
            target_size:  (height, width) to upsample the heatmap to.
                          Should match the original image display size.

        Returns:
            cam: numpy float32 array of shape target_size, values in [0, 1].
                 Higher values = stronger influence on predicted class.

        Raises:
            RuntimeError: If hooks didn't fire (wrong target layer, or model
                          was under torch.no_grad() context).
        """
        self.model.eval()
        self.model.zero_grad()
        self._activations = None
        self._gradients = None

        # Forward pass — activations are captured by the forward hook
        output = self.model(image_tensor)   # [1, num_classes]
        if self._activations is None:
            raise RuntimeError(
                "Forward hook did not fire. Make sure target_layer is a "
                "submodule of model and the model was called with the image tensor."
            )

        # Backward pass on the target class score — gradients captured by backward hook
        target_score = output[0, class_idx]
        target_score.backward()

        if self._gradients is None:
            raise RuntimeError(
                "Backward hook did not fire. Ensure the model's classifier has "
                "requires_grad=True parameters so the computational graph "
                "reaches the target layer. Do not call within torch.no_grad()."
            )

        # Step 3: global-average-pool gradients → per-channel importance weights
        # activations: [1, C, H, W]   gradients: [1, C, H, W]
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]

        cam = (weights * self._activations.detach()).sum(dim=1)     # [1, H, W]
        cam = torch.relu(cam)                                        # keep only positive influence
        cam = cam.squeeze().cpu().numpy()                            # [H, W]

        # Step 4: upsample from [7,7] → target_size
        if cam.shape[0] != target_size[0] or cam.shape[1] != target_size[1]:
            cam = cv2.resize(cam, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove all registered hooks. Call after you're done with this GradCAM instance."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        log.debug("GradCAM hooks removed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    def __del__(self):
        if self._handles:
            self.remove_hooks()


# ── Heatmap overlay ────────────────────────────────────────────────────────────

def overlay_heatmap(
    cam: np.ndarray,
    original_pil: Image.Image,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """
    Blend a Grad-CAM heatmap with the original X-ray image.

    Args:
        cam:          [H, W] float32 numpy array, values in [0, 1].
                      Typically shape [224, 224] from GradCAM.generate().
        original_pil: Original PIL Image (any size — will be resized to match cam).
        alpha:        Heatmap opacity [0.0, 1.0]. 0=original only, 1=heatmap only.
                      0.4 is a good default for clinical readability.
        colormap:     Any matplotlib colormap name. Common choices:
                      "jet"     — blue→green→red (most common for Grad-CAM)
                      "hot"     — black→red→yellow→white
                      "plasma"  — purple→red→yellow
                      "inferno" — black→purple→red→yellow

    Returns:
        PIL Image (RGB) of size cam.shape (224×224 by default), ready for
        st.image() or cv2.imwrite().
    """
    h, w = cam.shape

    # Apply matplotlib colormap → [H, W, 4] RGBA float, drop alpha channel
    cmap = plt.get_cmap(colormap)
    heatmap_rgba = cmap(cam)                            # [H, W, 4], values in [0, 1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)   # [H, W, 3]

    # Resize original to match cam dimensions
    orig_resized = original_pil.resize((w, h), Image.LANCZOS).convert("RGB")
    orig_arr = np.array(orig_resized, dtype=np.float32)

    # Alpha-blend: result = alpha * heatmap + (1-alpha) * original
    blended = (alpha * heatmap_rgb.astype(np.float32)
               + (1.0 - alpha) * orig_arr).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# ── Convenience: generate + overlay in one call ────────────────────────────────

def get_gradcam_overlay(
    model: nn.Module,
    target_layer: nn.Module,
    image_tensor: torch.Tensor,
    original_pil: Image.Image,
    class_idx: int = 0,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> tuple[np.ndarray, Image.Image]:
    """
    One-shot helper: generate heatmap and return both the raw cam array and
    the blended PIL image. Automatically removes hooks after use.

    Returns:
        (cam, overlay_pil)
        cam          — [224, 224] float32 numpy array in [0, 1]
        overlay_pil  — PIL Image ready for st.image() or saving
    """
    with GradCAM(model, target_layer) as gc:
        cam = gc.generate(image_tensor, class_idx=class_idx)
    overlay_pil = overlay_heatmap(cam, original_pil, alpha=alpha, colormap=colormap)
    return cam, overlay_pil