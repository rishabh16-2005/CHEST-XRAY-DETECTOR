"""
dataset_nih.py — Data loading for NIH ChestX-ray14 14-class multi-label classification.

Key differences from dataset.py (binary Kaggle):
┌─────────────────────┬──────────────────────────────────┬───────────────────────────────────┐
│                     │ Binary (dataset.py)               │ NIH 14-class (this file)          │
├─────────────────────┼──────────────────────────────────┼───────────────────────────────────┤
│ Image directory     │ Subfolders NORMAL/ PNEUMONIA/     │ Flat images/ folder               │
│ Label source        │ Folder name                       │ Data_Entry_2017_v2020.csv         │
│ Label format        │ 0 or 1 scalar                     │ Multi-hot float32 vector [14]     │
│ Split strategy      │ Stratified image-level            │ Patient-level (no leakage)        │
│ Val source          │ Merged from tiny official val     │ 15% of official train_val patients│
│ pos_weight shape    │ [1]                               │ [14] — one weight per class       │
└─────────────────────┴──────────────────────────────────┴───────────────────────────────────┘

WHY PATIENT-LEVEL SPLIT MATTERS:
    One patient can have 5–10 follow-up X-rays. An image-level random split
    lets the same patient appear in both train and test — the model memorises
    patient anatomy rather than learning pathology features. This inflates
    test AUC by 5–10 points. The NIH dataset provides official patient-level
    splits in train_val_list.txt and test_list.txt. We use those for test,
    and do our own patient-level split within train_val for the val set.

Expected dataset directory layout:
    DATA_ROOT/
        images/                       112,120 .png files (flat, no subfolders)
        Data_Entry_2017_v2020.csv     labels + patient metadata
        train_val_list.txt            86,524 image names for train+val
        test_list.txt                 25,596 image names for test

Public API:
    parse_finding_labels(label_str)   → np.ndarray [14] multi-hot
    build_dataframes_nih(data_root)   → (train_df, val_df, test_df)
    compute_pos_weight_nih(train_df)  → torch.Tensor [14]
    get_transforms_nih(split)         → transforms.Compose
    NIHChestXrayDataset               → PyTorch Dataset
    get_dataloaders_nih(data_root)    → (Dict[str, DataLoader], Tensor[14])
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

from src.config_nih import CONFIG_NIH, NIH_CLASSES

log = logging.getLogger(__name__)


# ── Label parsing ─────────────────────────────────────────────────────────────

# Build a fast lookup: class name → column index
_CLASS_TO_IDX: Dict[str, int] = {cls: i for i, cls in enumerate(NIH_CLASSES)}


def parse_finding_labels(label_str: str) -> np.ndarray:
    """
    Convert a pipe-separated finding string to a [14] multi-hot float32 vector.

    Examples:
        "No Finding"                       → [0, 0, 0, ..., 0]  (all zeros)
        "Atelectasis"                       → [1, 0, 0, ..., 0]
        "Atelectasis|Effusion"              → [1, 0, 1, ..., 0]
        "Cardiomegaly|Effusion|Infiltration"→ [0, 1, 1, 1, ..., 0]

    "No Finding" means the radiologist found no pathological abnormality.
    We encode it as all-zero (the model predicts nothing) rather than as an
    explicit 15th class — this is standard for the CheXNet/NIH literature.

    Unknown findings (not in NIH_CLASSES) are silently ignored. This handles
    minor label-string variations in the CSV without crashing.
    """
    vector = np.zeros(len(NIH_CLASSES), dtype=np.float32)

    if label_str.strip() == "No Finding":
        return vector

    for finding in label_str.split("|"):
        finding = finding.strip()
        idx = _CLASS_TO_IDX.get(finding)
        if idx is not None:
            vector[idx] = 1.0
        else:
            log.debug(f"Unknown finding ignored: '{finding}'")

    return vector


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_csv(data_root: Path) -> pd.DataFrame:
    """
    Load and pre-process Data_Entry_2017_v2020.csv.

    Returns DataFrame with columns:
        image_name   — e.g. "00000001_000.png"
        filepath     — absolute path to the PNG in images/
        patient_id   — int (used for patient-level split)
        labels       — np.ndarray [14] multi-hot float32
        finding_str  — original pipe-separated string (kept for debugging)
    """
    csv_path = data_root / "Data_Entry_2017_v2020.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Label CSV not found: {csv_path}\n"
            "Expected: Data_Entry_2017_v2020.csv in DATA_ROOT"
        )

    df = pd.read_csv(csv_path, usecols=["Image Index", "Finding Labels", "Patient ID"])
    df.columns = ["image_name", "finding_str", "patient_id"]

    images_dir = data_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            "Expected: images/ subfolder containing all 112,120 .png files"
        )

    df["filepath"] = df["image_name"].apply(lambda x: str((images_dir / x).resolve()))
    df["labels"] = df["finding_str"].apply(parse_finding_labels)

    log.info(f"CSV loaded: {len(df)} entries from {csv_path.name}")
    return df


def _load_split_list(path: Path) -> set:
    """Load a newline-separated list of image filenames into a set."""
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Expected: train_val_list.txt and test_list.txt in DATA_ROOT"
        )
    return set(open(path).read().splitlines())


def build_dataframes_nih(
    data_root: str,
    val_split: float = CONFIG_NIH.VAL_SPLIT,
    seed: int = CONFIG_NIH.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build train, val, test DataFrames using the official NIH patient-level splits.

    Strategy
    --------
    1.  Load the master CSV → one row per image, multi-hot labels.
    2.  Apply official test_list.txt → held-out test set (never touched again).
    3.  train_val_list.txt gives 86,524 images across ~24K patients.
    4.  Patient-level val split: randomly assign val_split fraction of PATIENTS
        to validation. All images from a patient land in the same split —
        zero data leakage.

    Args:
        data_root: Path to the directory containing images/, the CSV, and .txt files.
        val_split: Fraction of train_val PATIENTS to reserve for validation.
        seed:      For the patient-level split RNG.

    Returns:
        (train_df, val_df, test_df) — each with columns
        [image_name, filepath, patient_id, labels, finding_str]
    """
    root = Path(data_root)
    df = _load_csv(root)

    train_val_names = _load_split_list(root / "train_val_list.txt")
    test_names = _load_split_list(root / "test_list.txt")

    test_df = df[df["image_name"].isin(test_names)].reset_index(drop=True)
    train_val_df = df[df["image_name"].isin(train_val_names)].reset_index(drop=True)

    log.info(
        f"Official splits — train_val: {len(train_val_df)} images "
        f"| test: {len(test_df)} images"
    )

    # Patient-level split within train_val
    unique_patients = train_val_df["patient_id"].unique()
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=val_split,
        random_state=seed,
    )
    train_patients = set(train_patients)
    val_patients = set(val_patients)

    train_df = train_val_df[train_val_df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = train_val_df[train_val_df["patient_id"].isin(val_patients)].reset_index(drop=True)

    _log_split_summary(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _log_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Log image counts and per-class prevalence for each split."""
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        labels_matrix = np.stack(df["labels"].values)       # [N, 14]
        prevalences = labels_matrix.mean(axis=0) * 100      # % per class
        n_patients = df["patient_id"].nunique()
        log.info(
            f"  {name:5s}: {len(df):6d} images | {n_patients:5d} patients"
        )
        for cls, pct in zip(NIH_CLASSES, prevalences):
            if pct > 0:
                log.debug(f"    {cls:<20s}: {pct:.1f}%")


# ── Class imbalance ───────────────────────────────────────────────────────────

def compute_pos_weight_nih(train_df: pd.DataFrame) -> Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.

    Formula per class i:  pos_weight[i] = n_negative_i / n_positive_i

    For a class like Pneumonia (1.3% of images):
        n_pos ≈ 1,100,  n_neg ≈ 73,000
        pos_weight ≈ 66 — the model is penalised 66× more for a false negative
        than for a false positive on Pneumonia. This prevents the model from
        learning "always predict 0" for rare classes.

    Returns:
        Tensor of shape [14] — pass directly to BCEWithLogitsLoss(pos_weight=...).
        BCEWithLogitsLoss broadcasts [14] to match batch output shape [B, 14].
    """
    labels_matrix = np.stack(train_df["labels"].values)    # [N, 14]
    n_pos = labels_matrix.sum(axis=0)                       # [14]
    n_neg = len(train_df) - n_pos                           # [14]
    weights = n_neg / (n_pos + 1e-8)                        # [14]

    for cls, w, p, n in zip(NIH_CLASSES, weights, n_pos.astype(int), n_neg.astype(int)):
        log.info(f"  {cls:<20s}: pos={p:6d}  neg={n:6d}  pos_weight={w:.2f}")

    return torch.tensor(weights, dtype=torch.float32)


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms_nih(split: str) -> transforms.Compose:
    """
    Transforms for NIH dataset — identical to binary transforms (same ImageNet
    pre-processing) since we're fine-tuning the same EfficientNetB3 backbone.

    X-rays can be horizontally flipped (anatomy is symmetric). We do NOT
    vertically flip because "upside-down lung" is not a valid augmentation.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize(CONFIG_NIH.RESIZE_SIZE),
            transforms.RandomCrop(CONFIG_NIH.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=CONFIG_NIH.HFLIP_PROB),
            transforms.RandomRotation(degrees=CONFIG_NIH.ROTATION_DEG),
            transforms.ColorJitter(
                brightness=CONFIG_NIH.BRIGHTNESS_JITTER,
                contrast=CONFIG_NIH.CONTRAST_JITTER,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    return transforms.Compose([
        transforms.Resize(CONFIG_NIH.RESIZE_SIZE),
        transforms.CenterCrop(CONFIG_NIH.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


# ── Dataset ────────────────────────────────────────────────────────────────────

class NIHChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14 multi-label classification.

    __getitem__ returns:
        image  : FloatTensor [3, 224, 224]  — ImageNet-normalised
        labels : FloatTensor [14]           — multi-hot, 0.0 or 1.0

    Note on label shape: BCEWithLogitsLoss expects targets shape [B, 14]
    matching logits shape [B, 14] — one float per class per sample.
    This is different from CrossEntropyLoss which expects a class index.
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
        image = Image.open(row["filepath"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(row["labels"], dtype=torch.float32)   # [14]
        return image, labels

    def get_labels_matrix(self) -> np.ndarray:
        """Return all labels as [N, 14] numpy array — used for pos_weight."""
        return np.stack(self.df["labels"].values)


# ── DataLoaders ────────────────────────────────────────────────────────────────

def get_dataloaders_nih(
    data_root: Optional[str] = None,
    batch_size: Optional[int] = None,
    val_split: float = CONFIG_NIH.VAL_SPLIT,
    seed: int = CONFIG_NIH.SEED,
) -> Tuple[Dict[str, DataLoader], Tensor]:
    """
    Build and return all three DataLoaders plus the [14] pos_weight tensor.

    Usage:
        loaders, pos_weight = get_dataloaders_nih()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        for images, labels in loaders["train"]:
            # images: [B, 3, 224, 224]
            # labels: [B, 14]  ← multi-hot, multiple 1s per row possible
            ...

    Returns:
        loaders    : dict with "train", "val", "test" DataLoaders
        pos_weight : FloatTensor [14], one weight per pathology class
    """
    root = data_root or CONFIG_NIH.DATA_ROOT
    bs = batch_size or CONFIG_NIH.BATCH_SIZE

    train_df, val_df, test_df = build_dataframes_nih(root, val_split=val_split, seed=seed)
    pos_weight = compute_pos_weight_nih(train_df)

    datasets = {
        "train": NIHChestXrayDataset(train_df, transform=get_transforms_nih("train")),
        "val":   NIHChestXrayDataset(val_df,   transform=get_transforms_nih("val")),
        "test":  NIHChestXrayDataset(test_df,  transform=get_transforms_nih("test")),
    }

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"], batch_size=bs, shuffle=True,
            num_workers=CONFIG_NIH.NUM_WORKERS, pin_memory=CONFIG_NIH.PIN_MEMORY,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=bs, shuffle=False,
            num_workers=CONFIG_NIH.NUM_WORKERS, pin_memory=CONFIG_NIH.PIN_MEMORY,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=bs, shuffle=False,
            num_workers=CONFIG_NIH.NUM_WORKERS, pin_memory=CONFIG_NIH.PIN_MEMORY,
        ),
    }

    log.info("NIH DataLoaders ready:")
    for split, loader in loaders.items():
        log.info(f"  {split:5s}: {len(loader.dataset):6d} images | {len(loader):4d} batches (bs={bs})")

    return loaders, pos_weight
