"""
tests/test_losses.py — Verify loss functions before any training starts.

Test gates (from blueprint Week 2, file 04):
  ✓ Loss on random logits + labels returns a positive scalar
  ✓ WeightedBCELoss with pos_weight matches a manual BCEWithLogitsLoss computation
  ✓ FocalLoss reduces loss contribution from confident-correct predictions
    relative to plain BCE — this is the entire point of focal loss
  ✓ get_loss_fn() factory returns the right class and raises on bad input
  ✓ Gradients flow through both losses (required for backward() to work)
  ✓ pos_weight buffer moves correctly with .to(device) [CPU-only check here,
    since no GPU is available in this environment]

Run with:
    pytest tests/test_losses.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def random_logits_and_labels():
    """16 samples — mix of both classes, realistic batch size for testing."""
    torch.manual_seed(42)
    logits = torch.randn(16, 1)
    labels = torch.randint(0, 2, (16, 1)).float()
    return logits, labels


@pytest.fixture
def pos_weight():
    return torch.tensor([0.35])   # ~ NORMAL/PNEUMONIA ratio from the real dataset


# ── WeightedBCELoss ────────────────────────────────────────────────────────────


class TestWeightedBCELoss:
    def test_returns_positive_scalar(self, random_logits_and_labels):
        from src.losses import WeightedBCELoss
        logits, labels = random_logits_and_labels
        criterion = WeightedBCELoss()
        loss = criterion(logits, labels)
        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert loss.item() > 0, f"BCE loss should be positive, got {loss.item()}"

    def test_matches_manual_bce_with_logits(self, random_logits_and_labels, pos_weight):
        """WeightedBCELoss(pos_weight) should produce identical output to
        calling nn.BCEWithLogitsLoss(pos_weight=...) directly."""
        from src.losses import WeightedBCELoss
        logits, labels = random_logits_and_labels

        wrapped = WeightedBCELoss(pos_weight=pos_weight)
        reference = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        wrapped_loss = wrapped(logits, labels)
        reference_loss = reference(logits, labels)

        assert torch.allclose(wrapped_loss, reference_loss), (
            f"WeightedBCELoss ({wrapped_loss.item():.6f}) should match "
            f"BCEWithLogitsLoss ({reference_loss.item():.6f})"
        )

    def test_no_pos_weight_works(self, random_logits_and_labels):
        """pos_weight=None should default to standard unweighted BCE."""
        from src.losses import WeightedBCELoss
        logits, labels = random_logits_and_labels
        criterion = WeightedBCELoss(pos_weight=None)
        loss = criterion(logits, labels)
        assert loss.item() > 0

    def test_gradients_flow(self, random_logits_and_labels):
        """backward() must populate .grad on a leaf tensor that requires grad."""
        from src.losses import WeightedBCELoss
        logits, labels = random_logits_and_labels
        logits = logits.clone().requires_grad_(True)

        criterion = WeightedBCELoss()
        loss = criterion(logits, labels)
        loss.backward()

        assert logits.grad is not None, "Gradients did not flow back to logits"
        assert not torch.isnan(logits.grad).any(), "Gradients contain NaN"

    def test_pos_weight_moves_with_to_device(self, pos_weight):
        """Calling .to() on the wrapper should move the internal pos_weight buffer too."""
        from src.losses import WeightedBCELoss
        criterion = WeightedBCELoss(pos_weight=pos_weight)
        criterion = criterion.to("cpu")   # No GPU in this sandbox; verifies the call doesn't error
        assert criterion.bce.pos_weight.device.type == "cpu"


# ── FocalLoss ─────────────────────────────────────────────────────────────────


class TestFocalLoss:
    def test_returns_positive_scalar(self, random_logits_and_labels):
        from src.losses import FocalLoss
        logits, labels = random_logits_and_labels
        criterion = FocalLoss()
        loss = criterion(logits, labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_default_hyperparameters(self):
        from src.losses import FocalLoss
        from src.config import CONFIG
        criterion = FocalLoss()
        assert criterion.alpha == CONFIG.FOCAL_ALPHA
        assert criterion.gamma == CONFIG.FOCAL_GAMMA

    def test_down_weights_confident_correct_predictions(self):
        """
        The defining property of focal loss: for a confidently-correct
        prediction (high logit, label=1), focal loss should be smaller than
        plain BCE loss on the same input — that's the "easy example
        down-weighting" focal loss is designed to do.
        """
        from src.losses import FocalLoss

        confident_correct_logit = torch.tensor([[5.0]])   # sigmoid(5) ≈ 0.993
        label = torch.tensor([[1.0]])

        focal = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        bce = nn.BCEWithLogitsLoss(reduction="none")

        focal_loss = focal(confident_correct_logit, label).item()
        bce_loss = bce(confident_correct_logit, label).item()

        assert focal_loss < bce_loss, (
            f"Focal loss ({focal_loss:.6f}) should be smaller than plain BCE "
            f"({bce_loss:.6f}) for a confident, correct prediction"
        )

    def test_does_not_down_weight_confident_wrong_predictions(self):
        """
        For a confidently-WRONG prediction (high logit, but label=0), focal
        loss's (1-p_t)^gamma term should be close to 1 — i.e. it should NOT
        down-weight hard/wrong examples, only easy/correct ones.
        """
        from src.losses import FocalLoss

        confident_wrong_logit = torch.tensor([[5.0]])   # predicts PNEUMONIA strongly
        label = torch.tensor([[0.0]])                   # but truth is NORMAL

        focal = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        focal_loss = focal(confident_wrong_logit, label).item()

        # A hard/wrong example should still produce a substantial loss
        assert focal_loss > 0.5, (
            f"Focal loss for a confident wrong prediction should remain high, "
            f"got {focal_loss:.4f}"
        )

    def test_gamma_zero_approximates_alpha_weighted_bce(self):
        """With gamma=0, the focal term (1-p_t)^0 == 1 for all samples, so
        FocalLoss should reduce to alpha-weighted BCE."""
        from src.losses import FocalLoss

        logits = torch.tensor([[1.5], [-2.0], [0.3]])
        labels = torch.tensor([[1.0], [0.0], [1.0]])

        focal = FocalLoss(alpha=0.5, gamma=0.0, reduction="none")
        bce = nn.BCEWithLogitsLoss(reduction="none")

        focal_loss = focal(logits, labels)
        bce_loss = bce(logits, labels) * 0.5   # alpha=0.5 applied uniformly

        assert torch.allclose(focal_loss, bce_loss, atol=1e-5), (
            "With gamma=0 and alpha=0.5, FocalLoss should match 0.5 * BCE"
        )

    def test_reduction_modes(self, random_logits_and_labels):
        from src.losses import FocalLoss
        logits, labels = random_logits_and_labels

        mean_loss = FocalLoss(reduction="mean")(logits, labels)
        sum_loss = FocalLoss(reduction="sum")(logits, labels)
        none_loss = FocalLoss(reduction="none")(logits, labels)

        assert mean_loss.dim() == 0
        assert sum_loss.dim() == 0
        assert none_loss.shape == logits.shape
        assert torch.allclose(none_loss.mean(), mean_loss, atol=1e-5)
        assert torch.allclose(none_loss.sum(), sum_loss, atol=1e-5)

    def test_gradients_flow(self, random_logits_and_labels):
        from src.losses import FocalLoss
        logits, labels = random_logits_and_labels
        logits = logits.clone().requires_grad_(True)

        loss = FocalLoss()(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_with_pos_weight(self, random_logits_and_labels, pos_weight):
        """FocalLoss should accept an optional pos_weight without erroring."""
        from src.losses import FocalLoss
        logits, labels = random_logits_and_labels
        criterion = FocalLoss(pos_weight=pos_weight)
        loss = criterion(logits, labels)
        assert loss.item() > 0


# ── get_loss_fn factory ──────────────────────────────────────────────────────


class TestGetLossFn:
    def test_weighted_bce_string(self, pos_weight):
        from src.losses import WeightedBCELoss, get_loss_fn
        criterion = get_loss_fn("weighted_bce", pos_weight=pos_weight)
        assert isinstance(criterion, WeightedBCELoss)

    def test_bce_alias(self, pos_weight):
        from src.losses import WeightedBCELoss, get_loss_fn
        criterion = get_loss_fn("bce", pos_weight=pos_weight)
        assert isinstance(criterion, WeightedBCELoss)

    def test_focal_string(self, pos_weight):
        from src.losses import FocalLoss, get_loss_fn
        criterion = get_loss_fn("focal", pos_weight=pos_weight)
        assert isinstance(criterion, FocalLoss)

    def test_case_insensitive(self, pos_weight):
        from src.losses import FocalLoss, get_loss_fn
        criterion = get_loss_fn("FOCAL", pos_weight=pos_weight)
        assert isinstance(criterion, FocalLoss)

    def test_invalid_name_raises(self):
        from src.losses import get_loss_fn
        with pytest.raises(ValueError):
            get_loss_fn("not_a_real_loss")

    def test_factory_output_is_usable(self, random_logits_and_labels, pos_weight):
        """End-to-end: factory output should compute a valid loss like any other."""
        from src.losses import get_loss_fn
        logits, labels = random_logits_and_labels
        for name in ["weighted_bce", "focal"]:
            criterion = get_loss_fn(name, pos_weight=pos_weight)
            loss = criterion(logits, labels)
            assert loss.item() > 0, f"{name} loss should be positive"