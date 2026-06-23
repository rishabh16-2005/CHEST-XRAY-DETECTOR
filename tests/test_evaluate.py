"""
tests/test_evaluate.py — Verify the evaluation pipeline.

Tests are split into metric-computation tests (no matplotlib, fast) and
visualization tests (use matplotlib with non-interactive Agg backend, save
to temp files). No real model inference is run — we use synthetic logits
to verify the metric math is correct.

Run with:
    pytest tests/test_evaluate.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def perfect_predictions():
    """Probabilities that perfectly separate the two classes."""
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    probs = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95], dtype=np.float32)
    return probs, labels


@pytest.fixture
def random_predictions():
    """Random predictions — AUC should be ≈ 0.5."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=200).astype(np.float32)
    probs = rng.uniform(0, 1, size=200).astype(np.float32)
    return probs, labels


@pytest.fixture
def imbalanced_predictions():
    """Realistic class imbalance: 75% PNEUMONIA. Model is decent but not perfect."""
    rng = np.random.default_rng(42)
    n = 200
    labels = (rng.uniform(size=n) < 0.75).astype(np.float32)
    # Predictions are correlated with labels but noisy
    probs = np.clip(labels * 0.7 + rng.normal(0, 0.2, n), 0, 1).astype(np.float32)
    return probs, labels


# ── compute_all_metrics tests ─────────────────────────────────────────────────


class TestComputeAllMetrics:
    def test_returns_required_keys(self, perfect_predictions):
        from src.evaluate import compute_all_metrics
        probs, labels = perfect_predictions
        metrics = compute_all_metrics(probs, labels)
        required_keys = {
            "auc", "fpr", "tpr", "threshold_used", "best_threshold",
            "f1", "precision", "recall", "specificity",
            "confusion_matrix", "TP", "TN", "FP", "FN",
            "n_positive", "n_negative", "class_names",
        }
        assert required_keys.issubset(metrics.keys()), (
            f"Missing keys: {required_keys - metrics.keys()}"
        )

    def test_perfect_predictions_auc_is_one(self, perfect_predictions):
        from src.evaluate import compute_all_metrics
        probs, labels = perfect_predictions
        metrics = compute_all_metrics(probs, labels)
        assert metrics["auc"] == pytest.approx(1.0, abs=1e-6)

    def test_random_predictions_auc_near_half(self, random_predictions):
        """200 samples of random predictions should give AUC close to 0.5."""
        from src.evaluate import compute_all_metrics
        probs, labels = random_predictions
        metrics = compute_all_metrics(probs, labels)
        assert 0.4 < metrics["auc"] < 0.6, (
            f"Random predictions should give AUC near 0.5, got {metrics['auc']:.4f}"
        )

    def test_auc_matches_sklearn(self, imbalanced_predictions):
        """compute_all_metrics AUC should match sklearn.roc_auc_score exactly."""
        from src.evaluate import compute_all_metrics
        probs, labels = imbalanced_predictions
        expected_auc = roc_auc_score(labels, probs)
        metrics = compute_all_metrics(probs, labels)
        assert metrics["auc"] == pytest.approx(expected_auc, abs=1e-6)

    def test_confusion_matrix_shape(self, imbalanced_predictions):
        from src.evaluate import compute_all_metrics
        probs, labels = imbalanced_predictions
        metrics = compute_all_metrics(probs, labels)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (2, 2), f"Expected 2×2 confusion matrix, got {cm.shape}"

    def test_confusion_matrix_sums_to_n_samples(self, imbalanced_predictions):
        from src.evaluate import compute_all_metrics
        probs, labels = imbalanced_predictions
        metrics = compute_all_metrics(probs, labels)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.sum() == len(probs), "Confusion matrix entries should sum to n_samples"

    def test_tp_tn_fp_fn_consistent(self, imbalanced_predictions):
        """TP + TN + FP + FN must equal n_samples, and match the confusion matrix."""
        from src.evaluate import compute_all_metrics
        probs, labels = imbalanced_predictions
        m = compute_all_metrics(probs, labels)
        assert m["TP"] + m["TN"] + m["FP"] + m["FN"] == len(probs)
        cm = np.array(m["confusion_matrix"])
        assert cm[1, 1] == m["TP"], "TP mismatch"
        assert cm[0, 0] == m["TN"], "TN mismatch"
        assert cm[0, 1] == m["FP"], "FP mismatch"
        assert cm[1, 0] == m["FN"], "FN mismatch"

    def test_recall_at_high_threshold_is_low(self, perfect_predictions):
        """At threshold=0.99, no positive is predicted → recall ≈ 0."""
        from src.evaluate import compute_all_metrics
        probs, labels = perfect_predictions
        metrics = compute_all_metrics(probs, labels, threshold=0.99)
        assert metrics["recall"] < 0.1, (
            f"At threshold=0.99, recall should be near 0, got {metrics['recall']:.4f}"
        )

    def test_recall_at_low_threshold_is_high(self, perfect_predictions):
        """At threshold=0.01, everything predicted positive → recall ≈ 1."""
        from src.evaluate import compute_all_metrics
        probs, labels = perfect_predictions
        metrics = compute_all_metrics(probs, labels, threshold=0.01)
        assert metrics["recall"] > 0.9, (
            f"At threshold=0.01, recall should be near 1, got {metrics['recall']:.4f}"
        )

    def test_best_threshold_in_valid_range(self, imbalanced_predictions):
        from src.evaluate import compute_all_metrics
        probs, labels = imbalanced_predictions
        metrics = compute_all_metrics(probs, labels)
        assert 0.0 <= metrics["best_threshold"] <= 1.0

    def test_n_positive_n_negative_correct(self, imbalanced_predictions):
        from src.evaluate import compute_all_metrics
        probs, labels = imbalanced_predictions
        metrics = compute_all_metrics(probs, labels)
        assert metrics["n_positive"] == int(labels.sum())
        assert metrics["n_negative"] == int((1 - labels).sum())
        assert metrics["n_positive"] + metrics["n_negative"] == len(labels)

    def test_class_names_default(self, perfect_predictions):
        from src.evaluate import compute_all_metrics
        from src.config import CONFIG
        probs, labels = perfect_predictions
        metrics = compute_all_metrics(probs, labels)
        assert metrics["class_names"] == CONFIG.CLASS_NAMES


# ── evaluate_model tests (uses synthetic DataLoader + tiny model) ──────────────


class TestEvaluateModel:
    """
    Uses a tiny Linear(3*224*224, 1) model whose weights are fixed by seed,
    so the same logits are produced for the same images on every call.
    Function-scoped fixture: a fresh model + loader per test → no shared
    mutable state between tests (avoids the class-scope / call_count trap).
    """

    @pytest.fixture
    def loader_and_model(self):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(99)
        n, bs = 32, 8
        images = torch.randn(n, 3, 224, 224)
        labels = torch.randint(0, 2, (n, 1)).float()

        # Stateless deterministic model: fixed Linear weights → same output every call
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 224 * 224, 1, bias=False)
                nn.init.constant_(self.fc.weight, 0.001)  # predictable small logits

            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))

        model = TinyModel().eval()
        loader = DataLoader(TensorDataset(images, labels), batch_size=bs, shuffle=False)
        device = torch.device("cpu")
        return loader, model, device

    def test_returns_dict(self, loader_and_model):
        from src.evaluate import evaluate_model
        loader, model, device = loader_and_model
        metrics = evaluate_model(model, loader, device)
        assert isinstance(metrics, dict)

    def test_n_samples_correct(self, loader_and_model):
        from src.evaluate import evaluate_model
        loader, model, device = loader_and_model
        metrics = evaluate_model(model, loader, device)
        assert metrics["n_samples"] == len(loader.dataset)

    def test_auc_in_valid_range(self, loader_and_model):
        from src.evaluate import evaluate_model
        loader, model, device = loader_and_model
        metrics = evaluate_model(model, loader, device)
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_probs_in_zero_one(self, loader_and_model):
        from src.evaluate import evaluate_model
        loader, model, device = loader_and_model
        metrics = evaluate_model(model, loader, device)
        probs = np.array(metrics["probs"])
        assert (probs >= 0).all() and (probs <= 1).all(), (
            f"Probabilities out of [0,1] range: min={probs.min():.4f}, max={probs.max():.4f}"
        )

    def test_labels_are_binary(self, loader_and_model):
        from src.evaluate import evaluate_model
        loader, model, device = loader_and_model
        metrics = evaluate_model(model, loader, device)
        labels = np.array(metrics["labels"])
        assert set(labels.astype(int).tolist()).issubset({0, 1})


# ── Visualization tests ───────────────────────────────────────────────────────


class TestVisualization:
    """
    Use the Agg (non-interactive) matplotlib backend so tests run without
    a display (CI, Colab headless mode). Close figures after each test to
    prevent memory accumulation.
    """

    @pytest.fixture(autouse=True)
    def use_agg_backend(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def test_plot_confusion_matrix_saves_png(self, tmp_path):
        from src.evaluate import plot_confusion_matrix
        cm = np.array([[120, 10], [5, 200]])
        fig = plot_confusion_matrix(
            cm=cm,
            classes=["NORMAL", "PNEUMONIA"],
            save_path=tmp_path / "test_cm.png",
        )
        assert (tmp_path / "test_cm.png").exists()
        assert fig is not None

    def test_plot_roc_curve_saves_png(self, tmp_path):
        from src.evaluate import plot_roc_curve
        probs = np.array([0.1, 0.4, 0.6, 0.9])
        labels = np.array([0, 0, 1, 1])
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probs)
        fig = plot_roc_curve(
            fpr=fpr.tolist(),
            tpr=tpr.tolist(),
            roc_auc=1.0,
            save_path=tmp_path / "test_roc.png",
        )
        assert (tmp_path / "test_roc.png").exists()
        assert fig is not None

    def test_plot_confusion_matrix_no_save_path(self):
        from src.evaluate import plot_confusion_matrix
        cm = np.array([[50, 5], [3, 80]])
        fig = plot_confusion_matrix(cm=cm, classes=["NORMAL", "PNEUMONIA"])
        assert fig is not None   # Returns figure even without save_path


# ── save_evaluation_results tests ────────────────────────────────────────────


class TestSaveResults:
    def test_saves_valid_json(self, perfect_predictions, tmp_path):
        from src.evaluate import compute_all_metrics, save_evaluation_results
        probs, labels = perfect_predictions
        metrics = compute_all_metrics(probs, labels)
        out_path = tmp_path / "eval.json"
        save_evaluation_results(metrics, out_path)
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert "auc" in data
        assert "f1" in data

    def test_excludes_large_list_fields(self, imbalanced_predictions, tmp_path):
        """probs, labels, fpr, tpr should NOT be in the saved JSON (they'd be huge)."""
        from src.evaluate import compute_all_metrics, save_evaluation_results
        probs, labels = imbalanced_predictions
        metrics = compute_all_metrics(probs, labels)
        metrics["probs"] = probs.tolist()
        metrics["labels"] = labels.tolist()
        out_path = tmp_path / "eval.json"
        save_evaluation_results(metrics, out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert "probs" not in data, "probs should not be saved to JSON"
        assert "labels" not in data, "labels should not be saved to JSON"


# ── print_metrics_report smoke test ──────────────────────────────────────────


def test_print_metrics_report_runs(capsys, imbalanced_predictions):
    """print_metrics_report() should print to stdout without raising."""
    from src.evaluate import compute_all_metrics, print_metrics_report
    probs, labels = imbalanced_predictions
    metrics = compute_all_metrics(probs, labels)
    metrics["n_samples"] = len(probs)
    print_metrics_report(metrics)
    captured = capsys.readouterr()
    assert "AUC" in captured.out
    assert "Recall" in captured.out
    assert "Confusion" in captured.out
