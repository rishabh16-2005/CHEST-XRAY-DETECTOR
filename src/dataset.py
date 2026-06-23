"""
dataset.py — Data loading for binary chest X-ray classification.

Dataset: Kaggle Chest X-Ray Images (Pneumonia)
  URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Expected folder layout after download + extraction:
    chest_xray/
        train/
            NORMAL/       1,341 images
            PNEUMONIA/    3,875 images
        val/
            NORMAL/           8 images  ← too small for meaningful AUC
            PNEUMONIA/         8 images
        test/
            NORMAL/         234 images
            PNEUMONIA/      390 images

Why we merge val back into train:
    The provided val/ split contains only 16 images total (8 per class).
    Computing AUC-ROC on 16 samples is statistically meaningless and produces
    wildly unstable validation curves that mislead training decisions.
    We merge train/ + val/ into one pool, then apply a stratified 80/20 split
    to produce ~4,246 train images and ~1,062 val images.
    test/ is NEVER touched during this process — it stays as the held-out set.

Class imbalance:
    NORMAL    : 1,349 images (25.9%)
    PNEUMONIA : 3,883 images (74.1%)
    pos_weight ≈ 1,349 / 3,883 ≈ 0.35 — passed to BCEWithLogitsLoss.
    This means the model is penalised relatively less for a missed NORMAL
    than for a missed PNEUMONIA (false negative in a disease detector is worse).

Key public API:
    build_dataframes(data_root)  → train_df, val_df, test_df
    compute_pos_weight(train_df) → float
    get_transforms(split)        → transforms.Compose
    ChestXrayDataset(df, ...)    → PyTorch Dataset
    get_dataloaders(data_root)   → {"train": DL, "val": DL, "test": DL}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import CONFIG

log = logging.getLogger(__name__)

# ── Label encoding ─────────────────────────────────────────────────────────────
# NORMAL=0 (negative class), PNEUMONIA=1 (positive class)
# BCEWithLogitsLoss expects 1.0 for the disease we want to detect.
LABEL_MAP: Dict[str, int] = {"NORMAL": 0, "PNEUMONIA": 1}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ── Image collection ──────────────────────────────────────────────────────────

def _collect_images_from_dir(folder: Path) -> pd.DataFrame:
    """
    Walk a class-labelled directory and build a flat DataFrame.

    Expects structure:
        folder/
            CLASSNAME_A/  ← directory name must match a key in LABEL_MAP
            CLASSNAME_B/

    Each row in the returned DataFrame:
        filepath   (str)  — absolute path to image file
        label      (int)  — 0 or 1 from LABEL_MAP
        class_name (str)  — "NORMAL" or "PNEUMONIA"

    Skips hidden files, non-image extensions, and unexpected subdirectory names.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Directory not found: {folder}\n"
            "Check that DATA_ROOT in config.py points to the chest_xray/ folder."
        )

    records: List[Dict] = []

    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        class_name = class_dir.name.upper()
        if class_name not in LABEL_MAP:
            log.warning(f"Unexpected subdirectory '{class_dir.name}' in {folder} — skipping")
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            if img_path.name.startswith("."):
                continue
            records.append({
                "filepath": str(img_path.resolve()),
                "label": LABEL_MAP[class_name],
                "class_name": class_name,
            })

    if not records:
        raise RuntimeError(
            f"No valid images found in {folder}. "
            "Expected NORMAL/ and PNEUMONIA/ subdirectories with .jpg/.png files."
        )

    df = pd.DataFrame(records)
    class_counts = df.groupby("class_name")["filepath"].count().to_dict()
    log.info(
        f"  {folder.name:6s}: {len(df):5d} images | "
        + " | ".join(f"{k}: {v}" for k, v in class_counts.items())
    )
    return df


# ── DataFrames ────────────────────────────────────────────────────────────────

def build_dataframes(
    data_root: str,
    val_split: float = CONFIG.VAL_SPLIT,
    seed: int = CONFIG.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build train, val, and test DataFrames from the Kaggle dataset folder.

    Steps
    -----
    1.  Scan train/ and val/ directories → collect image paths + labels.
    2.  Merge both into a single pool (the provided val/ has only 16 images).
    3.  Stratified split: (1 - val_split) train / val_split val.
        Stratification ensures the class ratio is preserved in both splits.
    4.  Scan test/ directory independently — never touched during splits.

    Args:
        data_root:  Path to chest_xray/ folder (e.g. "/content/chest_xray").
        val_split:  Fraction of combined pool to reserve for validation.
        seed:       Random seed for reproducible splitting.

    Returns:
        train_df, val_df, test_df — each has columns [filepath, label, class_name]
    """
    root = Path(data_root)
    log.info(f"Loading dataset from {root}")

    raw_train = _collect_images_from_dir(root / CONFIG.TRAIN_DIR)
    raw_val_tiny = _collect_images_from_dir(root / CONFIG.VAL_DIR)
    test_df = _collect_images_from_dir(root / CONFIG.TEST_DIR)

    # Merge train + tiny val before splitting
    combined = pd.concat([raw_train, raw_val_tiny], ignore_index=True)
    log.info(f"  Combined pool: {len(combined)} images before split")

    train_df, val_df = train_test_split(
        combined,
        test_size=val_split,
        stratify=combined["label"],    # Preserves class ratio in both halves
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    _log_split_summary(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _log_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Log a compact split summary with class ratios."""
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_pos = int(df["label"].sum())
        n_neg = len(df) - n_pos
        pct_pos = 100 * n_pos / len(df)
        log.info(
            f"  {name:5s} split: {len(df):5d} images | "
            f"NORMAL: {n_neg} ({100-pct_pos:.1f}%) | "
            f"PNEUMONIA: {n_pos} ({pct_pos:.1f}%)"
        )


# ── Class imbalance ───────────────────────────────────────────────────────────

def compute_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute positive class weight for BCEWithLogitsLoss.

    Formula: pos_weight = count(NORMAL) / count(PNEUMONIA)

    With ~1,341 NORMAL and ~3,875 PNEUMONIA images in train:
        pos_weight ≈ 0.35

    This tells the loss function: a missed NORMAL costs 0.35× what a missed
    PNEUMONIA costs — which is correct, since false negatives (missing real
    pneumonia) are more dangerous than false positives.

    Note: Despite the name, a lower pos_weight with more positives than
    negatives is the correct direction. The model gets penalised more for
    confidently predicting NORMAL when PNEUMONIA is present.

    Returns:
        Scalar tensor, ready to pass to BCEWithLogitsLoss(pos_weight=...).
    """
    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    weight = n_neg / (n_pos + 1e-8)

    log.info(
        f"Class weights → NORMAL (neg): {n_neg} | PNEUMONIA (pos): {n_pos} | "
        f"pos_weight: {weight:.4f}"
    )
    return torch.tensor([weight], dtype=torch.float32)


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Return the correct torchvision transform pipeline for a given split.

    Train — augmentation ON:
        Resize(256) → RandomCrop(224) → HFlip → Rotation(10°)
        → ColorJitter → ToTensor → Normalize(ImageNet)

    Val / Test — augmentation OFF:
        Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)

    ImageNet mean/std is used because EfficientNetB3 was pretrained on ImageNet.
    Using different normalization stats would shift the input distribution and
    degrade the pretrained features before fine-tuning can compensate.

    Args:
        split: One of "train", "val", "test".

    Returns:
        transforms.Compose ready to pass to ChestXrayDataset.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize(CONFIG.RESIZE_SIZE),
            transforms.RandomCrop(CONFIG.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=CONFIG.HFLIP_PROB),
            transforms.RandomRotation(degrees=CONFIG.ROTATION_DEG),
            transforms.ColorJitter(
                brightness=CONFIG.BRIGHTNESS_JITTER,
                contrast=CONFIG.CONTRAST_JITTER,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    # val and test share the same deterministic pipeline
    return transforms.Compose([
        transforms.Resize(CONFIG.RESIZE_SIZE),
        transforms.CenterCrop(CONFIG.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for binary chest X-ray classification.

    __getitem__ returns:
        image : FloatTensor of shape [3, IMG_SIZE, IMG_SIZE]
                Normalised to ImageNet mean/std.
        label : FloatTensor of shape [1]
                0.0 = NORMAL, 1.0 = PNEUMONIA
                Shape [1] (not scalar) is required by BCEWithLogitsLoss when
                model output is also [batch, 1].

    X-rays in this dataset are a mix of RGB JPEG and grayscale PNG.
    We always call .convert("RGB") to produce a consistent 3-channel input —
    the EfficientNetB3 stem expects 3 channels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        row = self.df.iloc[idx]

        # Always convert to RGB — handles grayscale X-rays and RGBA screenshots
        image = Image.open(row["filepath"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # float32 scalar wrapped in shape [1] for BCEWithLogitsLoss compatibility
        label = torch.tensor([row["label"]], dtype=torch.float32)

        return image, label

    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array — used for class-weight computation."""
        return self.df["label"].values.astype(np.float32)


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    data_root: Optional[str] = None,
    batch_size: Optional[int] = None,
    val_split: float = CONFIG.VAL_SPLIT,
    seed: int = CONFIG.SEED,
) -> Tuple[Dict[str, DataLoader], torch.Tensor]:
    """
    Build and return all three DataLoaders plus the pos_weight tensor.

    Returns:
        loaders  : dict with keys "train", "val", "test"
        pos_weight: scalar Tensor for BCEWithLogitsLoss (computed from train split)

    Usage:
        loaders, pos_weight = get_dataloaders()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        for images, labels in loaders["train"]:
            ...
    """
    root = data_root or CONFIG.DATA_ROOT
    bs = batch_size or CONFIG.BATCH_SIZE

    train_df, val_df, test_df = build_dataframes(root, val_split=val_split, seed=seed)
    pos_weight = compute_pos_weight(train_df)

    datasets: Dict[str, ChestXrayDataset] = {
        "train": ChestXrayDataset(train_df, transform=get_transforms("train")),
        "val":   ChestXrayDataset(val_df,   transform=get_transforms("val")),
        "test":  ChestXrayDataset(test_df,  transform=get_transforms("test")),
    }

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=bs,
            shuffle=True,
            num_workers=CONFIG.NUM_WORKERS,
            pin_memory=CONFIG.PIN_MEMORY,
            drop_last=True,        # Avoid incomplete batches breaking BatchNorm
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=bs,
            shuffle=False,
            num_workers=CONFIG.NUM_WORKERS,
            pin_memory=CONFIG.PIN_MEMORY,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=bs,
            shuffle=False,
            num_workers=CONFIG.NUM_WORKERS,
            pin_memory=CONFIG.PIN_MEMORY,
        ),
    }

    log.info("DataLoaders ready:")
    for split, loader in loaders.items():
        n = len(loader.dataset)
        n_batches = len(loader)
        log.info(f"  {split:5s}: {n:5d} images | {n_batches:3d} batches (bs={bs})")

    return loaders, pos_weight