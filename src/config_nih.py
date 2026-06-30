"""
config_nih.py — Configuration for NIH ChestX-ray14 14-class multi-label training.

Sits alongside config.py (binary). Import whichever you need:
    from src.config     import CONFIG       # binary NORMAL vs PNEUMONIA
    from src.config_nih import CONFIG_NIH   # NIH 14-class multi-label

Nothing in this file imports from config.py — the two configs are independent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── 14 pathology classes in fixed order ───────────────────────────────────────
# Order must never change once training starts — it defines the column index
# of each class in the [B, 14] label and logit tensors.
# "No Finding" is NOT included — images labelled "No Finding" get all-zero vectors.
NIH_CLASSES: List[str] = [
    "Atelectasis",       # 0  — 10.3% prevalence
    "Cardiomegaly",      # 1  —  2.5%
    "Effusion",          # 2  — 11.8%
    "Infiltration",      # 3  — 19.8%  ← most common pathology
    "Mass",              # 4  —  5.1%
    "Nodule",            # 5  —  5.6%
    "Pneumonia",         # 6  —  1.3%  ← rarest, hardest
    "Pneumothorax",      # 7  —  2.7%
    "Consolidation",     # 8  —  4.2%
    "Edema",             # 9  —  2.1%
    "Emphysema",         # 10 —  2.2%
    "Fibrosis",          # 11 —  1.6%
    "Pleural_Thickening",# 12 —  3.0%
    "Hernia",            # 13 —  0.2%  ← rarest overall
]


@dataclass
class ConfigNIH:

    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED: int = 42

    # ── Paths (Colab defaults) ────────────────────────────────────────────────
    # After downloading, the folder should contain:
    #   images/                      ← 112,120 PNG files (flat, no subfolders)
    #   Data_Entry_2017_v2020.csv    ← master label CSV
    #   train_val_list.txt           ← official train+val image names
    #   test_list.txt                ← official test image names
    #   BBox_List_2017.csv           ← bounding boxes for 984 images (bonus)
    DATA_ROOT: str = "/content/data/nih_chestxray"
    CHECKPOINT_DIR: str = "/content/drive/MyDrive/checkpoints/nih_chestxray"

    # ── Classification ────────────────────────────────────────────────────────
    CLASS_NAMES: List[str] = field(default_factory=lambda: NIH_CLASSES)
    NUM_CLASSES: int = 14

    # ── Image ─────────────────────────────────────────────────────────────────
    RESIZE_SIZE: int = 256
    IMG_SIZE: int = 224

    # ── DataLoader ────────────────────────────────────────────────────────────
    # Reduced batch from 32 → 16: 14-class output + AMP still fits T4 at 16
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    VAL_SPLIT: float = 0.15     # 15% of train_val patients → val
                                 # 85% → train (gives ~73K train images)

    # ── Augmentation ──────────────────────────────────────────────────────────
    HFLIP_PROB: float = 0.5
    ROTATION_DEG: int = 10
    BRIGHTNESS_JITTER: float = 0.2
    CONTRAST_JITTER: float = 0.2

    # ── Phase 1 — head only ───────────────────────────────────────────────────
    LR_PHASE1: float = 1e-3
    EPOCHS_PHASE1: int = 5

    # ── Phase 2 — fine-tune top blocks ────────────────────────────────────────
    LR_PHASE2: float = 1e-4
    EPOCHS_PHASE2: int = 10
    UNFREEZE_BLOCKS: int = 3

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    ETA_MIN: float = 1e-6

    # ── Model head ────────────────────────────────────────────────────────────
    DROPOUT: float = 0.4
    BACKBONE_OUT_FEATURES: int = 1536

    # ── Loss ──────────────────────────────────────────────────────────────────
    # pos_weight computed per-class in dataset_nih.compute_pos_weight()
    # Focal loss hyperparams — recommended for extreme imbalance (Pneumonia <2%)
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.25

    # ── Inference ─────────────────────────────────────────────────────────────
    # Per-class threshold — all classes default to 0.5 at start;
    # tune per-class after evaluation using Youden's J from evaluate_nih.py
    DEFAULT_THRESHOLD: float = 0.6

    # ── Experiment tracking ───────────────────────────────────────────────────
    WANDB_PROJECT: str = "nih-chestxray-14class"
    WANDB_ENTITY: str = ""
    USE_WANDB: bool = True

    # ── Mixed precision ───────────────────────────────────────────────────────
    USE_AMP: bool = True

    # ── Checkpointing ─────────────────────────────────────────────────────────
    CHECKPOINT_EVERY: int = 1
    EARLY_STOP_PATIENCE: int = 4   # More patience — multi-label AUC is noisier
    LOG_EVERY_N_BATCHES: int = 100  # 112K images → ~7K batches/epoch at bs=16

    # ── Target AUC (for gate checks) ─────────────────────────────────────────
    PHASE1_AUC_GATE: float = 0.70   # same gate as binary
    PHASE2_AUC_GATE: float = 0.82   # mean AUC across all 14 classes

    # ── Derived paths ─────────────────────────────────────────────────────────
    @property
    def images_dir(self) -> Path:
        return Path(self.DATA_ROOT) / "images"

    @property
    def csv_path(self) -> Path:
        return Path(self.DATA_ROOT) / "Data_Entry_2017_v2020.csv"

    @property
    def train_val_list(self) -> Path:
        return Path(self.DATA_ROOT) / "train_val_list.txt"

    @property
    def test_list(self) -> Path:
        return Path(self.DATA_ROOT) / "test_list.txt"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.CHECKPOINT_DIR)

    @property
    def total_epochs(self) -> int:
        return self.EPOCHS_PHASE1 + self.EPOCHS_PHASE2


CONFIG_NIH = ConfigNIH()
