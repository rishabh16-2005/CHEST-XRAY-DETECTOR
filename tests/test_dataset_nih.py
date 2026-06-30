"""
tests/test_dataset_nih.py — Verify NIH dataset pipeline without downloading real data.

Uses a synthetic fixture that mimics the NIH CSV + image structure so tests
run in seconds with no network access. The real CSV has 112,120 rows; the
fixture has 60 rows across 20 fake patients.

Critical test gates:
  ✓ parse_finding_labels produces correct multi-hot vectors
  ✓ "No Finding" → all-zero vector
  ✓ Unknown findings are silently ignored
  ✓ No patient overlap between train and val (patient-level split)
  ✓ No image overlap between (train+val) and test
  ✓ Batch shape == [B, 3, 224, 224] images, [B, 14] labels
  ✓ pos_weight shape == [14], all values positive
  ✓ get_transforms_nih returns correct image sizes

Run with:
    pytest tests/test_dataset_nih.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.config_nih import NIH_CLASSES
from src.dataset_nih import parse_finding_labels


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_png(path: Path) -> None:
    arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(path))


@pytest.fixture(scope="session")
def fake_nih_root(tmp_path_factory) -> Path:
    """
    Build a minimal synthetic NIH dataset directory:
        root/
            images/       ← 60 tiny PNG files
            Data_Entry_2017_v2020.csv
            train_val_list.txt   ← 48 image names
            test_list.txt        ← 12 image names
    """
    root = tmp_path_factory.mktemp("nih_chestxray")
    images_dir = root / "images"
    images_dir.mkdir()

    # 20 fake patients, 3 images each = 60 total
    rows = []
    n_patients = 20
    images_per_patient = 3
    all_image_names = []

    findings_cycle = [
        "No Finding",
        "Atelectasis",
        "Effusion|Infiltration",
        "Pneumonia",
        "Cardiomegaly|Effusion",
        "No Finding",
    ]

    for p_idx in range(n_patients):
        patient_id = 1000 + p_idx
        for img_idx in range(images_per_patient):
            img_name = f"{patient_id:08d}_{img_idx:03d}.png"
            _write_png(images_dir / img_name)
            finding = findings_cycle[(p_idx * images_per_patient + img_idx) % len(findings_cycle)]
            rows.append({
                "Image Index": img_name,
                "Finding Labels": finding,
                "Patient ID": patient_id,
            })
            all_image_names.append(img_name)

    df = pd.DataFrame(rows)
    df.to_csv(root / "Data_Entry_2017_v2020.csv", index=False)

    # 80% train_val, 20% test (by image count)
    n_test = 12
    train_val_names = all_image_names[:-n_test]
    test_names = all_image_names[-n_test:]

    (root / "train_val_list.txt").write_text("\n".join(train_val_names))
    (root / "test_list.txt").write_text("\n".join(test_names))

    return root


@pytest.fixture(scope="session")
def dataframes_nih(fake_nih_root):
    from src.dataset_nih import build_dataframes_nih
    return build_dataframes_nih(str(fake_nih_root), val_split=0.25, seed=42)


# ── parse_finding_labels tests ─────────────────────────────────────────────────

class TestParseFindingLabels:
    def test_no_finding_returns_all_zeros(self):
        vec = parse_finding_labels("No Finding")
        assert vec.shape == (14,)
        assert vec.sum() == 0.0, "No Finding should map to all-zero vector"

    def test_single_class(self):
        vec = parse_finding_labels("Atelectasis")
        assert vec[NIH_CLASSES.index("Atelectasis")] == 1.0
        assert vec.sum() == 1.0

    def test_multiple_classes(self):
        vec = parse_finding_labels("Atelectasis|Effusion")
        assert vec[NIH_CLASSES.index("Atelectasis")] == 1.0
        assert vec[NIH_CLASSES.index("Effusion")] == 1.0
        assert vec.sum() == 2.0

    def test_all_14_classes(self):
        label_str = "|".join(NIH_CLASSES)
        vec = parse_finding_labels(label_str)
        assert vec.sum() == 14.0
        assert (vec == 1.0).all()

    def test_unknown_finding_ignored(self):
        vec = parse_finding_labels("Atelectasis|UnknownDisease123")
        assert vec[NIH_CLASSES.index("Atelectasis")] == 1.0
        assert vec.sum() == 1.0, "Unknown findings should be silently ignored"

    def test_output_dtype_float32(self):
        vec = parse_finding_labels("No Finding")
        assert vec.dtype == np.float32

    def test_output_shape(self):
        vec = parse_finding_labels("Pneumonia")
        assert vec.shape == (14,)

    def test_values_are_binary(self):
        vec = parse_finding_labels("Effusion|Infiltration")
        assert set(vec.tolist()).issubset({0.0, 1.0})


# ── Patient-level split tests ──────────────────────────────────────────────────

class TestPatientSplit:
    def test_no_patient_overlap_train_val(self, dataframes_nih):
        train_df, val_df, _ = dataframes_nih
        train_patients = set(train_df["patient_id"].unique())
        val_patients = set(val_df["patient_id"].unique())
        overlap = train_patients & val_patients
        assert len(overlap) == 0, (
            f"Patient-level split leaks: {len(overlap)} patients in both train and val.\n"
            f"Overlapping patient IDs: {overlap}"
        )

    def test_no_patient_overlap_trainval_test(self, dataframes_nih):
        train_df, val_df, test_df = dataframes_nih
        trainval_patients = set(train_df["patient_id"]) | set(val_df["patient_id"])
        test_patients = set(test_df["patient_id"].unique())
        overlap = trainval_patients & test_patients
        assert len(overlap) == 0, (
            f"Test patients leak into train/val: {overlap}"
        )

    def test_no_image_overlap_trainval_test(self, dataframes_nih):
        train_df, val_df, test_df = dataframes_nih
        trainval_images = set(train_df["image_name"]) | set(val_df["image_name"])
        test_images = set(test_df["image_name"])
        assert len(trainval_images & test_images) == 0

    def test_all_images_assigned(self, dataframes_nih, fake_nih_root):
        train_df, val_df, test_df = dataframes_nih
        total = len(train_df) + len(val_df) + len(test_df)
        # 60 images total: 48 in train_val, 12 in test
        assert total == 60, f"Expected 60 total images, got {total}"

    def test_test_size_matches_list_file(self, dataframes_nih, fake_nih_root):
        _, _, test_df = dataframes_nih
        n_in_file = len((fake_nih_root / "test_list.txt").read_text().splitlines())
        assert len(test_df) == n_in_file


# ── Label correctness tests ────────────────────────────────────────────────────

class TestNIHLabels:
    def test_labels_column_shape(self, dataframes_nih):
        train_df, _, _ = dataframes_nih
        for _, row in train_df.head(5).iterrows():
            assert row["labels"].shape == (14,)

    def test_labels_are_float32(self, dataframes_nih):
        train_df, _, _ = dataframes_nih
        for _, row in train_df.head(5).iterrows():
            assert row["labels"].dtype == np.float32

    def test_labels_are_binary(self, dataframes_nih):
        train_df, _, _ = dataframes_nih
        matrix = np.stack(train_df["labels"].values)
        assert set(matrix.ravel().tolist()).issubset({0.0, 1.0})


# ── pos_weight tests ──────────────────────────────────────────────────────────

class TestPosWeightNIH:
    def test_shape_is_14(self, dataframes_nih):
        from src.dataset_nih import compute_pos_weight_nih
        train_df, _, _ = dataframes_nih
        pw = compute_pos_weight_nih(train_df)
        assert pw.shape == torch.Size([14]), f"Expected [14], got {pw.shape}"

    def test_all_positive(self, dataframes_nih):
        from src.dataset_nih import compute_pos_weight_nih
        train_df, _, _ = dataframes_nih
        pw = compute_pos_weight_nih(train_df)
        assert (pw > 0).all(), "All pos_weight values should be positive"

    def test_is_float_tensor(self, dataframes_nih):
        from src.dataset_nih import compute_pos_weight_nih
        train_df, _, _ = dataframes_nih
        pw = compute_pos_weight_nih(train_df)
        assert pw.dtype == torch.float32


# ── Transform tests ───────────────────────────────────────────────────────────

class TestTransformsNIH:
    def test_train_output_shape(self):
        from src.dataset_nih import get_transforms_nih
        t = get_transforms_nih("train")
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == torch.Size([3, 224, 224])

    def test_val_output_shape(self):
        from src.dataset_nih import get_transforms_nih
        t = get_transforms_nih("val")
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == torch.Size([3, 224, 224])

    def test_val_is_deterministic(self):
        from src.dataset_nih import get_transforms_nih
        t = get_transforms_nih("val")
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        assert torch.allclose(t(img), t(img)), "Val transform must be deterministic"


# ── Dataset class tests ───────────────────────────────────────────────────────

class TestNIHChestXrayDataset:
    @pytest.fixture
    def dataset(self, dataframes_nih):
        from src.dataset_nih import NIHChestXrayDataset, get_transforms_nih
        train_df, _, _ = dataframes_nih
        return NIHChestXrayDataset(train_df, transform=get_transforms_nih("val"))

    def test_len(self, dataset, dataframes_nih):
        train_df, _, _ = dataframes_nih
        assert len(dataset) == len(train_df)

    def test_getitem_image_shape(self, dataset):
        image, labels = dataset[0]
        assert image.shape == torch.Size([3, 224, 224])

    def test_getitem_labels_shape(self, dataset):
        _, labels = dataset[0]
        assert labels.shape == torch.Size([14]), (
            f"Expected [14], got {labels.shape}. "
            "BCEWithLogitsLoss needs [B, 14] matching logits."
        )

    def test_getitem_labels_dtype(self, dataset):
        _, labels = dataset[0]
        assert labels.dtype == torch.float32

    def test_getitem_labels_binary(self, dataset):
        for i in range(min(5, len(dataset))):
            _, labels = dataset[i]
            assert set(labels.tolist()).issubset({0.0, 1.0})

    def test_get_labels_matrix_shape(self, dataset, dataframes_nih):
        train_df, _, _ = dataframes_nih
        matrix = dataset.get_labels_matrix()
        assert matrix.shape == (len(train_df), 14)


# ── DataLoader batch tests ────────────────────────────────────────────────────

class TestNIHDataLoaders:
    @pytest.fixture(scope="class")
    def loaders_and_weight(self, fake_nih_root):
        from src.dataset_nih import get_dataloaders_nih
        return get_dataloaders_nih(
            data_root=str(fake_nih_root), batch_size=4, val_split=0.25, seed=42
        )

    def test_keys(self, loaders_and_weight):
        loaders, _ = loaders_and_weight
        assert set(loaders.keys()) == {"train", "val", "test"}

    def test_train_image_batch_shape(self, loaders_and_weight):
        loaders, _ = loaders_and_weight
        images, labels = next(iter(loaders["train"]))
        assert images.shape[1:] == torch.Size([3, 224, 224])

    def test_train_label_batch_shape(self, loaders_and_weight):
        loaders, _ = loaders_and_weight
        images, labels = next(iter(loaders["train"]))
        B = images.shape[0]
        assert labels.shape == torch.Size([B, 14]), (
            f"Expected [{B}, 14], got {labels.shape}"
        )

    def test_pos_weight_shape(self, loaders_and_weight):
        _, pos_weight = loaders_and_weight
        assert pos_weight.shape == torch.Size([14])
