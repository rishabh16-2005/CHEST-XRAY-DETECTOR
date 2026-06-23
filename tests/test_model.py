"""
tests/test_model.py — Verify model construction, freeze/unfreeze logic, and
output shape before any training starts.

All tests use pretrained=False to avoid downloading ImageNet weights from
download.pytorch.org, which isn't reachable in network-restricted CI/sandbox
environments. On Colab, build_model(pretrained=True) works fine — these
tests only verify architecture and behavior, not pretrained weight quality.

Test gates (from blueprint Week 2 + project structure):
  ✓ model(torch.randn(2, 3, 224, 224)).shape == [2, 1]   (binary: 1 logit, not 14)
  ✓ Output is raw logits — sigmoid(output) is in [0, 1]
  ✓ freeze_backbone() leaves only classifier trainable
  ✓ unfreeze_top_layers() unfreezes exactly stages [5, 6, 7, 8] + classifier
  ✓ unfreeze_top_layers() raises ValueError if n_blocks is too large
  ✓ Trainable parameter count increases from Phase 1 → Phase 2
  ✓ get_gradcam_target_layer() returns the head conv (features[-1])
  ✓ Model accepts variable batch sizes
  ✓ Model works in both train() and eval() mode without error

Run with:
    pytest tests/test_model.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model():
    """
    Build one model instance, shared across tests in this module.
    pretrained=False avoids any network call to download.pytorch.org.
    """
    from src.model import build_model
    return build_model(num_classes=1, pretrained=False, dropout=0.4)


@pytest.fixture
def sample_batch():
    """A small random batch — values don't need to be realistic X-rays for shape tests."""
    return torch.randn(2, 3, 224, 224)


# ── Construction & forward pass ─────────────────────────────────────────────────


class TestBuildModel:
    def test_output_shape_binary(self, model, sample_batch):
        """Binary classification: output should be [batch, 1], NOT [batch, 14]."""
        output = model(sample_batch)
        assert output.shape == torch.Size([2, 1]), (
            f"Expected output shape [2, 1] for binary classification, got {output.shape}"
        )

    def test_output_is_raw_logits(self, model, sample_batch):
        """
        Output should NOT already be sigmoid-applied — raw logits can be
        any real number, not bounded to [0, 1]. Sigmoid is applied separately
        at inference/loss time (BCEWithLogitsLoss applies it internally).
        """
        output = model(sample_batch)
        # Logits CAN occasionally fall in [0,1] by chance, so we check that
        # applying sigmoid changes the values (i.e. they weren't already bounded).
        sigmoid_output = torch.sigmoid(output)
        assert not torch.allclose(output, sigmoid_output), (
            "Model output appears to already have sigmoid applied — "
            "build_model() should return raw logits."
        )

    def test_sigmoid_of_output_in_valid_range(self, model, sample_batch):
        """sigmoid(logits) must always be in [0, 1] regardless of logit magnitude."""
        output = model(sample_batch)
        probs = torch.sigmoid(output)
        assert (probs >= 0).all() and (probs <= 1).all(), (
            "sigmoid(output) produced values outside [0, 1]"
        )

    def test_variable_batch_size(self, model):
        """Model should handle any batch size, including batch_size=1."""
        for batch_size in [1, 3, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = model(x)
            assert output.shape == torch.Size([batch_size, 1]), (
                f"Batch size {batch_size} produced output shape {output.shape}"
            )

    def test_multiclass_head_for_future_extension(self):
        """
        build_model(num_classes=14) should work too — this is what the
        14-class NIH extension will use later. Verifying it now ensures
        the head construction logic isn't hardcoded to binary.
        """
        from src.model import build_model
        m = build_model(num_classes=14, pretrained=False)
        output = m(torch.randn(2, 3, 224, 224))
        assert output.shape == torch.Size([2, 14])

    def test_dropout_layer_present(self, model):
        """Classifier head should contain a Dropout layer before the Linear layer."""
        assert isinstance(model.classifier[0], nn.Dropout), (
            "Expected classifier[0] to be a Dropout layer"
        )
        assert model.classifier[0].p == 0.4, (
            f"Expected dropout p=0.4, got {model.classifier[0].p}"
        )

    def test_classifier_in_features(self, model):
        """The Linear layer's in_features must match EfficientNetB3's 1536-dim output."""
        linear = model.classifier[1]
        assert isinstance(linear, nn.Linear)
        assert linear.in_features == 1536, (
            f"Expected in_features=1536, got {linear.in_features}"
        )

    def test_model_works_in_eval_mode(self, model, sample_batch):
        """No errors when switching to eval mode (disables dropout, freezes BN stats)."""
        model.eval()
        with torch.no_grad():
            output = model(sample_batch)
        assert output.shape == torch.Size([2, 1])
        model.train()   # Reset for other tests sharing this fixture

    def test_eval_mode_is_deterministic(self, model, sample_batch):
        """In eval mode (dropout off), the same input must produce the same output."""
        model.eval()
        with torch.no_grad():
            out1 = model(sample_batch)
            out2 = model(sample_batch)
        assert torch.allclose(out1, out2), (
            "eval() mode should be deterministic — dropout must be disabled"
        )
        model.train()


# ── Freeze / unfreeze logic ────────────────────────────────────────────────────


class TestFreezeUnfreeze:
    @pytest.fixture
    def fresh_model(self):
        """A new model per test — freeze state must not leak between tests."""
        from src.model import build_model
        return build_model(num_classes=1, pretrained=False)

    def test_freeze_backbone_freezes_features(self, fresh_model):
        from src.model import freeze_backbone
        freeze_backbone(fresh_model)
        for param in fresh_model.features.parameters():
            assert not param.requires_grad, "features should be frozen after freeze_backbone()"

    def test_freeze_backbone_keeps_classifier_trainable(self, fresh_model):
        from src.model import freeze_backbone
        freeze_backbone(fresh_model)
        for param in fresh_model.classifier.parameters():
            assert param.requires_grad, "classifier should remain trainable after freeze_backbone()"

    def test_unfreeze_top_layers_unfreezes_correct_stages(self, fresh_model):
        """
        With n_blocks=3 on a 9-stage feature extractor, stages [5, 6, 7, 8]
        should become trainable; stages [0, 1, 2, 3, 4] should stay frozen.
        """
        from src.model import freeze_backbone, unfreeze_top_layers
        freeze_backbone(fresh_model)
        unfreeze_top_layers(fresh_model, n_blocks=3)

        feature_stages = list(fresh_model.features.children())
        assert len(feature_stages) == 9, "Expected 9 feature stages in EfficientNetB3"

        for idx in [0, 1, 2, 3, 4]:
            for param in feature_stages[idx].parameters():
                assert not param.requires_grad, f"Stage {idx} should still be frozen"

        for idx in [5, 6, 7, 8]:
            for param in feature_stages[idx].parameters():
                assert param.requires_grad, f"Stage {idx} should be unfrozen"

    def test_unfreeze_keeps_classifier_trainable(self, fresh_model):
        from src.model import freeze_backbone, unfreeze_top_layers
        freeze_backbone(fresh_model)
        unfreeze_top_layers(fresh_model, n_blocks=3)
        for param in fresh_model.classifier.parameters():
            assert param.requires_grad

    def test_unfreeze_raises_for_excessive_n_blocks(self, fresh_model):
        """n_blocks so large it would unfreeze the stem (stage 0) should raise."""
        from src.model import unfreeze_top_layers
        with pytest.raises(ValueError):
            unfreeze_top_layers(fresh_model, n_blocks=10)

    def test_trainable_params_increase_phase1_to_phase2(self, fresh_model):
        """Sanity check: Phase 2 must have strictly more trainable parameters than Phase 1."""
        from src.model import freeze_backbone, unfreeze_top_layers
        from src.utils import count_parameters

        freeze_backbone(fresh_model)
        phase1_counts = count_parameters(fresh_model)

        unfreeze_top_layers(fresh_model, n_blocks=3)
        phase2_counts = count_parameters(fresh_model)

        assert phase2_counts["trainable"] > phase1_counts["trainable"], (
            "Phase 2 should have more trainable parameters than Phase 1"
        )
        assert phase2_counts["total"] == phase1_counts["total"], (
            "Total parameter count should not change between phases"
        )

    def test_different_n_blocks_unfreezes_different_amounts(self, fresh_model):
        """Larger n_blocks should unfreeze more parameters."""
        from src.model import build_model, freeze_backbone, unfreeze_top_layers
        from src.utils import count_parameters

        model_a = build_model(num_classes=1, pretrained=False)
        freeze_backbone(model_a)
        unfreeze_top_layers(model_a, n_blocks=2)
        counts_a = count_parameters(model_a)

        model_b = build_model(num_classes=1, pretrained=False)
        freeze_backbone(model_b)
        unfreeze_top_layers(model_b, n_blocks=4)
        counts_b = count_parameters(model_b)

        assert counts_b["trainable"] > counts_a["trainable"], (
            "n_blocks=4 should unfreeze more parameters than n_blocks=2"
        )


# ── Grad-CAM target layer (used in Week 4) ──────────────────────────────────────


class TestGradCAMTargetLayer:
    def test_returns_last_feature_stage(self, model):
        from src.model import get_gradcam_target_layer
        target = get_gradcam_target_layer(model)
        assert target is model.features[-1], (
            "get_gradcam_target_layer() should return model.features[-1] (head conv)"
        )

    def test_target_layer_produces_expected_channels(self, model, sample_batch):
        """
        Hooking the target layer and running a forward pass should produce
        a feature map with 1536 channels (the head conv's output dimension) —
        this is what Grad-CAM will hook onto in Week 4.
        """
        from src.model import get_gradcam_target_layer

        target_layer = get_gradcam_target_layer(model)
        captured = {}

        def hook(module, input, output):
            captured["activation"] = output

        handle = target_layer.register_forward_hook(hook)
        model.eval()
        with torch.no_grad():
            model(sample_batch)
        handle.remove()
        model.train()

        assert "activation" in captured, "Forward hook did not fire"
        assert captured["activation"].shape[1] == 1536, (
            f"Expected 1536 channels from head conv, got {captured['activation'].shape[1]}"
        )