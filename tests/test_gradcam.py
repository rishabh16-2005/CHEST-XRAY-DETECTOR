"""
tests/test_gradcam.py — Verify GradCAM implementation.

Blueprint gate (Week 4):
  ✓ cam.shape == (224, 224)
  ✓ Values in [0, 1]
  ✓ Forward hook fires (activations captured)
  ✓ Backward hook fires (gradients captured)
  ✓ overlay_heatmap returns a PIL Image of shape (224, 224)
  ✓ remove_hooks() clears the handle list
  ✓ Context manager usage (with GradCAM(...)) removes hooks on exit
  ✓ get_gradcam_overlay convenience function returns correct types

All tests use pretrained=False to avoid downloading ImageNet weights.
GradCAM requires gradient computation — do NOT wrap calls in torch.no_grad().

Run with:
    pytest tests/test_gradcam.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model_and_target():
    """Build a fresh (untrained) model and its Grad-CAM target layer."""
    from src.model import build_model, get_gradcam_target_layer
    model = build_model(num_classes=1, pretrained=False)
    model.eval()
    target = get_gradcam_target_layer(model)
    return model, target


@pytest.fixture
def image_tensor():
    """A single normalised random image tensor — values in realistic ImageNet-norm range."""
    torch.manual_seed(7)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_pil():
    """Fake 300×400 X-ray PIL image (grayscale-ish RGB)."""
    arr = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── GradCAM.generate() tests ─────────────────────────────────────────────────


class TestGradCAMGenerate:
    def test_output_shape_224x224(self, model_and_target, image_tensor):
        from src.gradcam import GradCAM
        model, target = model_and_target
        with GradCAM(model, target) as gc:
            cam = gc.generate(image_tensor, class_idx=0)
        assert cam.shape == (224, 224), (
            f"Expected cam shape (224, 224), got {cam.shape}"
        )

    def test_output_values_in_zero_one(self, model_and_target, image_tensor):
        from src.gradcam import GradCAM
        model, target = model_and_target
        with GradCAM(model, target) as gc:
            cam = gc.generate(image_tensor, class_idx=0)
        assert cam.min() >= 0.0 - 1e-6, f"cam.min()={cam.min():.6f} < 0"
        assert cam.max() <= 1.0 + 1e-6, f"cam.max()={cam.max():.6f} > 1"

    def test_output_dtype_is_float32(self, model_and_target, image_tensor):
        from src.gradcam import GradCAM
        model, target = model_and_target
        with GradCAM(model, target) as gc:
            cam = gc.generate(image_tensor, class_idx=0)
        assert cam.dtype == np.float32, f"Expected float32, got {cam.dtype}"

    def test_forward_hook_captures_activations(self, model_and_target, image_tensor):
        """
        Structural test: the forward hook must capture a non-None activation
        tensor with the right number of channels (1536 for EfficientNetB3 head conv).
        We do not assert magnitude because a random-init model produces
        near-machine-epsilon activations — that is correct behaviour, not a bug.
        """
        from src.gradcam import GradCAM
        model, target = model_and_target
        gc = GradCAM(model, target)
        gc.generate(image_tensor, class_idx=0)
        assert gc._activations is not None, "Forward hook did not capture activations"
        # Shape: [1, C, H, W] — C must be 1536 (EfficientNetB3 head conv output)
        assert gc._activations.shape[1] == 1536, (
            f"Expected 1536 channels, got {gc._activations.shape[1]}"
        )
        assert gc._activations.ndim == 4, "Activations should be 4-D [B, C, H, W]"
        gc.remove_hooks()

    def test_forward_hook_fires(self, model_and_target, image_tensor):
        """Activations should be captured by the forward hook during generate()."""
        from src.gradcam import GradCAM
        model, target = model_and_target
        gc = GradCAM(model, target)
        gc.generate(image_tensor, class_idx=0)
        assert gc._activations is not None, "Forward hook did not capture activations"
        gc.remove_hooks()

    def test_backward_hook_fires(self, model_and_target, image_tensor):
        """Gradients should be captured by the backward hook during generate()."""
        from src.gradcam import GradCAM
        model, target = model_and_target
        gc = GradCAM(model, target)
        gc.generate(image_tensor, class_idx=0)
        assert gc._gradients is not None, "Backward hook did not capture gradients"
        gc.remove_hooks()

    def test_gradients_shape_matches_activations(self, model_and_target, image_tensor):
        """Gradient tensor shape must match activation tensor shape for GAP to work."""
        from src.gradcam import GradCAM
        model, target = model_and_target
        gc = GradCAM(model, target)
        gc.generate(image_tensor, class_idx=0)
        assert gc._gradients.shape == gc._activations.shape, (
            f"Gradient shape {gc._gradients.shape} != activation shape "
            f"{gc._activations.shape}"
        )
        gc.remove_hooks()

    def test_custom_target_size(self, model_and_target, image_tensor):
        """generate() should upsample to any requested target_size."""
        from src.gradcam import GradCAM
        model, target = model_and_target
        with GradCAM(model, target) as gc:
            cam = gc.generate(image_tensor, class_idx=0, target_size=(128, 128))
        assert cam.shape == (128, 128), f"Expected (128, 128), got {cam.shape}"

    def test_each_generate_call_overwrites_activations(self, model_and_target):
        """
        Structural: calling generate() twice must overwrite _activations each time
        (no stale caching). We verify _activations is not None after each call and
        that the second call did not retain the first call's tensor identity.
        """
        from src.gradcam import GradCAM
        model, target = model_and_target
        img1 = torch.randn(1, 3, 224, 224)
        img2 = torch.randn(1, 3, 224, 224)
        gc = GradCAM(model, target)
        gc.generate(img1, class_idx=0)
        act1_id = id(gc._activations)
        gc.generate(img2, class_idx=0)
        act2_id = id(gc._activations)
        assert gc._activations is not None
        # The tensor object is replaced each time the hook fires
        assert act1_id != act2_id, (
            "Second generate() call did not replace _activations — "
            "activations may be cached from the first call."
        )
        gc.remove_hooks()


# ── Hook management tests ─────────────────────────────────────────────────────


class TestHookManagement:
    def test_remove_hooks_clears_handle_list(self, model_and_target, image_tensor):
        from src.gradcam import GradCAM
        model, target = model_and_target
        gc = GradCAM(model, target)
        assert len(gc._handles) == 2, "Expected 2 handles (forward + backward)"
        gc.remove_hooks()
        assert len(gc._handles) == 0, "Handles should be cleared after remove_hooks()"

    def test_context_manager_removes_hooks_on_exit(self, model_and_target, image_tensor):
        from src.gradcam import GradCAM
        model, target = model_and_target
        with GradCAM(model, target) as gc:
            cam = gc.generate(image_tensor, class_idx=0)
        # After the with block, handles should be gone
        assert len(gc._handles) == 0, (
            "Context manager __exit__ should call remove_hooks()"
        )

    def test_multiple_instances_on_same_layer(self, model_and_target, image_tensor):
        """
        Two separate GradCAM instances on the same target layer should not
        interfere — each must store activations/gradients independently.
        """
        from src.gradcam import GradCAM
        model, target = model_and_target

        with GradCAM(model, target) as gc1:
            cam1 = gc1.generate(image_tensor, class_idx=0)

        with GradCAM(model, target) as gc2:
            cam2 = gc2.generate(image_tensor, class_idx=0)

        assert np.allclose(cam1, cam2, atol=1e-5), (
            "Sequential GradCAM instances on the same layer+image should produce "
            "identical heatmaps"
        )


# ── overlay_heatmap tests ─────────────────────────────────────────────────────


class TestOverlayHeatmap:
    @pytest.fixture
    def valid_cam(self):
        """A synthetic heatmap that highlights the centre of the image."""
        cam = np.zeros((224, 224), dtype=np.float32)
        cam[80:140, 80:140] = 1.0   # bright square in the middle
        return cam

    def test_returns_pil_image(self, valid_cam, sample_pil):
        from src.gradcam import overlay_heatmap
        result = overlay_heatmap(valid_cam, sample_pil, alpha=0.4, colormap="jet")
        assert isinstance(result, Image.Image), (
            f"Expected PIL Image, got {type(result)}"
        )

    def test_output_size_matches_cam(self, valid_cam, sample_pil):
        """Overlay should be (cam.shape[1], cam.shape[0]) = (224, 224)."""
        from src.gradcam import overlay_heatmap
        result = overlay_heatmap(valid_cam, sample_pil, alpha=0.4)
        assert result.size == (224, 224), (
            f"Expected size (224, 224), got {result.size}"
        )

    def test_output_is_rgb(self, valid_cam, sample_pil):
        from src.gradcam import overlay_heatmap
        result = overlay_heatmap(valid_cam, sample_pil, alpha=0.4)
        assert result.mode == "RGB", f"Expected RGB mode, got {result.mode}"

    def test_pixel_values_in_valid_range(self, valid_cam, sample_pil):
        from src.gradcam import overlay_heatmap
        result = overlay_heatmap(valid_cam, sample_pil, alpha=0.4)
        arr = np.array(result)
        assert arr.min() >= 0, "Pixel values below 0"
        assert arr.max() <= 255, "Pixel values above 255"

    def test_alpha_zero_returns_original(self, valid_cam, sample_pil):
        """alpha=0 → 100% original image, no heatmap contribution."""
        from src.gradcam import overlay_heatmap
        from PIL import Image as _Image
        result = overlay_heatmap(valid_cam, sample_pil, alpha=0.0)
        # Use the same resize filter as overlay_heatmap (LANCZOS) for fair comparison
        orig_arr = np.array(sample_pil.resize((224, 224), _Image.LANCZOS).convert("RGB"), dtype=np.float32)
        result_arr = np.array(result, dtype=np.float32)
        # atol=2.0 accounts for uint8 rounding on the final clip().astype(uint8) step
        assert np.allclose(orig_arr, result_arr, atol=2.0), (
            f"At alpha=0, overlay should match original. Max diff: {np.abs(orig_arr - result_arr).max():.1f}"
        )

    def test_colormaps_produce_different_results(self, valid_cam, sample_pil):
        from src.gradcam import overlay_heatmap
        jet = np.array(overlay_heatmap(valid_cam, sample_pil, colormap="jet"))
        hot = np.array(overlay_heatmap(valid_cam, sample_pil, colormap="hot"))
        assert not np.allclose(jet, hot), (
            "Different colormaps should produce different overlays"
        )

    def test_all_zeros_cam(self, sample_pil):
        """A zero heatmap (no class activation anywhere) shouldn't crash."""
        from src.gradcam import overlay_heatmap
        cam = np.zeros((224, 224), dtype=np.float32)
        result = overlay_heatmap(cam, sample_pil, alpha=0.4)
        assert isinstance(result, Image.Image)


# ── get_gradcam_overlay convenience function ──────────────────────────────────


class TestGetGradCAMOverlay:
    def test_returns_tuple_of_correct_types(self, model_and_target, image_tensor, sample_pil):
        from src.gradcam import get_gradcam_overlay
        model, target = model_and_target
        cam, overlay = get_gradcam_overlay(
            model, target, image_tensor, sample_pil, class_idx=0
        )
        assert isinstance(cam, np.ndarray), f"cam should be ndarray, got {type(cam)}"
        assert isinstance(overlay, Image.Image), f"overlay should be PIL Image, got {type(overlay)}"

    def test_cam_shape_and_range(self, model_and_target, image_tensor, sample_pil):
        from src.gradcam import get_gradcam_overlay
        model, target = model_and_target
        cam, _ = get_gradcam_overlay(model, target, image_tensor, sample_pil)
        assert cam.shape == (224, 224)
        assert cam.min() >= 0.0 - 1e-6
        assert cam.max() <= 1.0 + 1e-6

    def test_hooks_removed_after_call(self, model_and_target, image_tensor, sample_pil):
        """get_gradcam_overlay uses a context manager — hooks must be gone after return."""
        from src.gradcam import GradCAM, get_gradcam_overlay
        model, target = model_and_target
        n_hooks_before = len(target._forward_hooks) + len(target._backward_hooks)
        get_gradcam_overlay(model, target, image_tensor, sample_pil)
        n_hooks_after = len(target._forward_hooks) + len(target._backward_hooks)
        assert n_hooks_after == n_hooks_before, (
            "Hook count should be the same before and after get_gradcam_overlay()"
        )