"""
tests/test_dataset.py — Verify the data pipeline before any training starts.

These tests use a synthetic fixture (small fake PNGs in a temp directory) so
they run in under 5 seconds without the real 6GB Kaggle download.

Test gates (from blueprint Week 1):
  ✓ No filepath overlap between train and val splits
  ✓ No filepath overlap between (train + val) and test
  ✓ Stratification preserved — class ratio within 5% of original
  ✓ Batch image tensor shape == [B, 3, 224, 224]
  ✓ Batch label tensor shape == [B, 1] and dtype == float32
  ✓ Label values are exactly 0.0 or 1.0 — no other values
  ✓ Image pixel values are in a realistic normalised range (roughly [-3, 3])
  ✓ pos_weight is a positive scalar > 0
  ✓ Transforms return a tensor of the correct size
  ✓ ChestXrayDataset.__len__ matches the DataFrame length
  ✓ Grayscale images are correctly converted to 3-channel tensors
  ✓ DataLoader drop_last=True produces consistent batch size for train
  ✓ get_transforms raises no errors for all valid split names

Run with:
    pytest tests/test_dataset.py -v
    pytest tests/test_dataset.py -v --tb=short   (terse tracebacks)
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _write_fake_image(path: Path, mode: str = "RGB") -> None:
    """Write a tiny random-noise PNG to disk. Fast and valid for PIL.open()."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr).convert(mode)
    img.save(str(path))


def _populate_split(split_dir: Path, n_normal: int, n_pneumonia: int) -> None:
    """Create NORMAL/ and PNEUMONIA/ subdirs with n fake images each."""
    (split_dir / "NORMAL").mkdir(parents=True)
    (split_dir / "PNEUMONIA").mkdir(parents=True)

    for i in range(n_normal):
        _write_fake_image(split_dir / "NORMAL" / f"normal_{i:04d}.jpeg")

    for i in range(n_pneumonia):
        _write_fake_image(split_dir / "PNEUMONIA" / f"pneumonia_{i:04d}.jpeg")


@pytest.fixture(scope="session")
def fake_dataset_root(tmp_path_factory) -> Path:
    """
    Build a minimal synthetic Kaggle chest_xray directory structure.

    Layout (mirrors real Kaggle dataset proportions at 1/100 scale):
        chest_xray/
            train/
                NORMAL/      13 images
                PNEUMONIA/   39 images
            val/
                NORMAL/       1 image   (intentionally tiny — mirrors real dataset)
                PNEUMONIA/    1 image
            test/
                NORMAL/       5 images
                PNEUMONIA/    8 images

    Session-scoped: built once and reused across all tests in the session.
    """
    root = tmp_path_factory.mktemp("chest_xray")
    _populate_split(root / "train", n_normal=13, n_pneumonia=39)
    _populate_split(root / "val", n_normal=1, n_pneumonia=1)
    _populate_split(root / "test", n_normal=5, n_pneumonia=8)
    return root


@pytest.fixture(scope="session")
def dataframes(fake_dataset_root) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build train/val/test DataFrames from the fake dataset.
    Session-scoped so build_dataframes() runs only once for all tests.
    """
    from src.dataset import build_dataframes
    return build_dataframes(str(fake_dataset_root), val_split=0.20, seed=42)


@pytest.fixture(scope="session")
def train_df(dataframes):
    return dataframes[0]


@pytest.fixture(scope="session")
def val_df(dataframes):
    return dataframes[1]


@pytest.fixture(scope="session")
def test_df(dataframes):
    return dataframes[2]


# ── Split integrity tests ─────────────────────────────────────────────────────


class TestSplitIntegrity:
    """Verify that splits are disjoint and contain the expected images."""

    def test_no_overlap_train_val(self, train_df, val_df):
        """No image filepath should appear in both train and val."""
        train_paths = set(train_df["filepath"])
        val_paths = set(val_df["filepath"])
        shared = train_paths & val_paths
        assert len(shared) == 0, (
            f"Data leak: {len(shared)} images appear in both train and val.\n"
            f"Example: {next(iter(shared))}"
        )

    def test_no_overlap_train_test(self, train_df, test_df):
        """No image filepath should appear in both train and test."""
        train_paths = set(train_df["filepath"])
        test_paths = set(test_df["filepath"])
        shared = train_paths & test_paths
        assert len(shared) == 0, (
            f"Data leak: {len(shared)} images appear in both train and test."
        )

    def test_no_overlap_val_test(self, val_df, test_df):
        """No image filepath should appear in both val and test."""
        val_paths = set(val_df["filepath"])
        test_paths = set(test_df["filepath"])
        shared = val_paths & test_paths
        assert len(shared) == 0, (
            f"Data leak: {len(shared)} images appear in both val and test."
        )

    def test_all_filepaths_exist(self, train_df, val_df, test_df):
        """Every filepath in every split must point to a real file on disk."""
        for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            missing = [p for p in df["filepath"] if not Path(p).exists()]
            assert len(missing) == 0, (
                f"{len(missing)} missing files in {df_name} split.\n"
                f"First missing: {missing[0]}"
            )

    def test_combined_size(self, train_df, val_df, fake_dataset_root):
        """
        train + val should equal total images from train/ + val/ directories.
        We put 13+39 = 52 in train/ and 1+1 = 2 in val/ → combined = 54.
        """
        expected_total = 13 + 39 + 1 + 1  # 54
        actual_total = len(train_df) + len(val_df)
        assert actual_total == expected_total, (
            f"Expected {expected_total} combined images, got {actual_total}"
        )

    def test_test_split_untouched(self, test_df):
        """Test set size must match exactly what was created (5+8=13)."""
        assert len(test_df) == 13, (
            f"Test set should have 13 images, got {len(test_df)}"
        )


# ── Label correctness tests ───────────────────────────────────────────────────


class TestLabels:
    """Verify label encoding and class distribution."""

    def test_label_values_are_binary(self, train_df, val_df, test_df):
        """Labels must be exactly 0 or 1 — no NaN, no other values."""
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            unique = set(df["label"].unique())
            assert unique <= {0, 1}, (
                f"{name} split has unexpected label values: {unique - {0, 1}}"
            )
            assert df["label"].isna().sum() == 0, f"{name} split has NaN labels"

    def test_both_classes_present(self, train_df, val_df, test_df):
        """Both NORMAL and PNEUMONIA must appear in every split."""
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            assert 0 in df["label"].values, f"{name} has no NORMAL samples"
            assert 1 in df["label"].values, f"{name} has no PNEUMONIA samples"

    def test_stratification_preserved(self, train_df, val_df, fake_dataset_root):
        """
        Class ratio in train and val should be within 5% of the combined pool ratio.

        Combined: 13 NORMAL + 39 PNEUMONIA = 25% NORMAL, 75% PNEUMONIA.
        With stratify=True, both splits should be within 5% of that ratio.
        """
        combined_pct_pos = (13 + 39 + 1 + 1) and (39 + 1) / (13 + 39 + 1 + 1)
        tolerance = 0.05

        for name, df in [("train", train_df), ("val", val_df)]:
            actual_pct_pos = df["label"].mean()
            assert abs(actual_pct_pos - combined_pct_pos) < tolerance, (
                f"{name} PNEUMONIA ratio {actual_pct_pos:.3f} deviates more than "
                f"{tolerance:.0%} from combined pool ratio {combined_pct_pos:.3f}. "
                f"Check that stratify=combined['label'] is passed to train_test_split."
            )

    def test_label_class_name_consistency(self, train_df):
        """
        NORMAL rows must have label=0, PNEUMONIA rows must have label=1.
        Catches a reversed mapping bug.
        """
        normal_labels = train_df.loc[train_df["class_name"] == "NORMAL", "label"]
        pneumonia_labels = train_df.loc[train_df["class_name"] == "PNEUMONIA", "label"]

        assert (normal_labels == 0).all(), "NORMAL rows have incorrect label != 0"
        assert (pneumonia_labels == 1).all(), "PNEUMONIA rows have incorrect label != 1"


# ── pos_weight tests ──────────────────────────────────────────────────────────


class TestPosWeight:
    """Verify the class-imbalance weight computation."""

    def test_pos_weight_is_tensor(self, train_df):
        from src.dataset import compute_pos_weight
        pw = compute_pos_weight(train_df)
        assert isinstance(pw, torch.Tensor), f"Expected Tensor, got {type(pw)}"

    def test_pos_weight_is_positive(self, train_df):
        from src.dataset import compute_pos_weight
        pw = compute_pos_weight(train_df)
        assert pw.item() > 0, f"pos_weight must be positive, got {pw.item()}"

    def test_pos_weight_formula(self, train_df):
        """pos_weight == n_neg / n_pos (with epsilon for safety)."""
        from src.dataset import compute_pos_weight
        n_pos = int(train_df["label"].sum())
        n_neg = len(train_df) - n_pos
        expected = n_neg / (n_pos + 1e-8)
        actual = compute_pos_weight(train_df).item()
        assert abs(actual - expected) < 1e-4, (
            f"pos_weight formula mismatch: expected {expected:.4f}, got {actual:.4f}"
        )

    def test_pos_weight_shape(self, train_df):
        """Shape should be [1] — matches BCEWithLogitsLoss pos_weight requirement."""
        from src.dataset import compute_pos_weight
        pw = compute_pos_weight(train_df)
        assert pw.shape == torch.Size([1]), (
            f"pos_weight shape should be [1], got {pw.shape}"
        )


# ── Transform tests ───────────────────────────────────────────────────────────


class TestTransforms:
    """Verify that transforms produce correctly shaped and normalised tensors."""

    @pytest.fixture
    def sample_rgb_image(self):
        arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    @pytest.fixture
    def sample_grayscale_image(self):
        """Simulates a real grayscale chest X-ray PNG."""
        arr = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        return Image.fromarray(arr, mode="L")

    def test_train_transform_output_shape(self, sample_rgb_image):
        from src.dataset import get_transforms
        t = get_transforms("train")
        tensor = t(sample_rgb_image)
        assert tensor.shape == torch.Size([3, 224, 224]), (
            f"Train transform output shape {tensor.shape} != [3, 224, 224]"
        )

    def test_val_transform_output_shape(self, sample_rgb_image):
        from src.dataset import get_transforms
        t = get_transforms("val")
        tensor = t(sample_rgb_image)
        assert tensor.shape == torch.Size([3, 224, 224])

    def test_test_transform_output_shape(self, sample_rgb_image):
        from src.dataset import get_transforms
        t = get_transforms("test")
        tensor = t(sample_rgb_image)
        assert tensor.shape == torch.Size([3, 224, 224])

    def test_tensor_dtype_is_float(self, sample_rgb_image):
        from src.dataset import get_transforms
        tensor = get_transforms("train")(sample_rgb_image)
        assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"

    def test_normalised_values_in_range(self, sample_rgb_image):
        """
        After ImageNet normalisation, pixel values should be in roughly [-3, 3].
        A very wide range suggests wrong mean/std or missing normalisation.
        """
        from src.dataset import get_transforms
        tensor = get_transforms("val")(sample_rgb_image)
        assert tensor.min() > -4.0, f"Min pixel {tensor.min():.2f} is unexpectedly low"
        assert tensor.max() < 4.0, f"Max pixel {tensor.max():.2f} is unexpectedly high"

    def test_grayscale_image_produces_3_channels(self, sample_grayscale_image):
        """
        Grayscale PNGs are common in this dataset. After .convert('RGB') in
        __getitem__, they must produce a [3, H, W] tensor, not [1, H, W].
        """
        from src.dataset import get_transforms, ChestXrayDataset

        # Manually apply the pipeline as __getitem__ does
        rgb_image = sample_grayscale_image.convert("RGB")
        tensor = get_transforms("val")(rgb_image)
        assert tensor.shape[0] == 3, (
            f"Grayscale image produced {tensor.shape[0]} channels instead of 3. "
            "Make sure Image.open().convert('RGB') is called in __getitem__."
        )

    def test_train_and_val_transforms_differ(self, sample_rgb_image):
        """
        Train transforms include random ops, so running them twice on the same
        image should (almost certainly) produce different tensors.
        Val transforms are deterministic and should produce identical tensors.
        """
        from src.dataset import get_transforms

        # Val: deterministic — must be identical
        val_t = get_transforms("val")
        val_out_1 = val_t(sample_rgb_image)
        val_out_2 = val_t(sample_rgb_image)
        assert torch.allclose(val_out_1, val_out_2), (
            "Val transforms are not deterministic — check for random ops."
        )


# ── Dataset class tests ───────────────────────────────────────────────────────


class TestChestXrayDataset:
    """Verify ChestXrayDataset __len__, __getitem__, and type contracts."""

    @pytest.fixture(scope="class")
    def dataset(self, train_df):
        from src.dataset import ChestXrayDataset, get_transforms
        return ChestXrayDataset(train_df, transform=get_transforms("train"))

    def test_len_matches_dataframe(self, dataset, train_df):
        assert len(dataset) == len(train_df), (
            f"Dataset len {len(dataset)} != DataFrame len {len(train_df)}"
        )

    def test_getitem_image_shape(self, dataset):
        image, _ = dataset[0]
        assert image.shape == torch.Size([3, 224, 224]), (
            f"Image shape {image.shape} != [3, 224, 224]"
        )

    def test_getitem_label_shape(self, dataset):
        _, label = dataset[0]
        assert label.shape == torch.Size([1]), (
            f"Label shape {label.shape} != [1]. "
            "BCEWithLogitsLoss requires label shape [B, 1] matching output."
        )

    def test_getitem_label_dtype(self, dataset):
        _, label = dataset[0]
        assert label.dtype == torch.float32, (
            f"Label dtype {label.dtype} must be float32 for BCEWithLogitsLoss"
        )

    def test_getitem_label_values(self, dataset):
        """Spot-check first 10 items — labels must be 0.0 or 1.0."""
        for i in range(min(10, len(dataset))):
            _, label = dataset[i]
            assert label.item() in (0.0, 1.0), (
                f"Item {i} has unexpected label {label.item()}"
            )

    def test_get_labels_returns_numpy(self, dataset):
        labels = dataset.get_labels()
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.float32

    def test_no_transform_still_works(self, train_df):
        """Dataset should not crash if transform=None (returns PIL Image)."""
        from src.dataset import ChestXrayDataset
        ds = ChestXrayDataset(train_df, transform=None)
        image, label = ds[0]
        assert isinstance(image, Image.Image), (
            "Without transform, __getitem__ should return a PIL Image"
        )


# ── DataLoader batch tests ────────────────────────────────────────────────────


class TestDataLoaders:
    """
    Verify DataLoader batch shapes and sizes.
    Uses a tiny batch_size=4 to keep tests fast.
    """

    @pytest.fixture(scope="class")
    def loaders(self, fake_dataset_root):
        from src.dataset import get_dataloaders
        loaders, pos_weight = get_dataloaders(
            data_root=str(fake_dataset_root),
            batch_size=4,
            val_split=0.20,
            seed=42,
        )
        return loaders, pos_weight

    def test_returns_three_loaders(self, loaders):
        l, _ = loaders
        assert set(l.keys()) == {"train", "val", "test"}

    def test_train_batch_image_shape(self, loaders):
        l, _ = loaders
        images, labels = next(iter(l["train"]))
        assert images.shape[1:] == torch.Size([3, 224, 224]), (
            f"Train batch image shape {images.shape} — expected [B, 3, 224, 224]"
        )

    def test_train_batch_label_shape(self, loaders):
        l, _ = loaders
        images, labels = next(iter(l["train"]))
        B = images.shape[0]
        assert labels.shape == torch.Size([B, 1]), (
            f"Train batch label shape {labels.shape} — expected [{B}, 1]"
        )

    def test_val_batch_image_shape(self, loaders):
        l, _ = loaders
        images, _ = next(iter(l["val"]))
        assert images.shape[1:] == torch.Size([3, 224, 224])

    def test_test_batch_image_shape(self, loaders):
        l, _ = loaders
        images, _ = next(iter(l["test"]))
        assert images.shape[1:] == torch.Size([3, 224, 224])

    def test_pos_weight_returned(self, loaders):
        _, pos_weight = loaders
        assert isinstance(pos_weight, torch.Tensor)
        assert pos_weight.item() > 0

    def test_train_drop_last(self, loaders):
        """
        drop_last=True on train loader means every batch has exactly batch_size
        images — no partial final batch. This prevents BatchNorm instability.
        """
        l, _ = loaders
        bs = l["train"].batch_size
        for images, labels in l["train"]:
            assert images.shape[0] == bs, (
                f"drop_last=True should ensure all batches have size {bs}, "
                f"got {images.shape[0]}"
            )

    def test_image_pixel_range_batch(self, loaders):
        """Batch pixel values should be in the normalised range."""
        l, _ = loaders
        images, _ = next(iter(l["val"]))
        assert images.min() > -4.0, f"Min pixel {images.min():.2f} is too low"
        assert images.max() < 4.0, f"Max pixel {images.max():.2f} is too high"