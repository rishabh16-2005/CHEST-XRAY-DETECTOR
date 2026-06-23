"""
config.py - Single source of truth for all hyperparameters.

Binary Classification variant: NORMAL vs PNEUMONIA.
Tuned for Google Colab T4 GPU (16 GB VRAM).

Usage:
    from src.config import CONFIG
    print(CONFIG.BATCH_SIZE)    # 32
    print(CONFIG.train_path)    # PosixPath('/content/chest_xray/train')

Design rule: nothing is ever hardcoded else in the codebase.
Every numeric constant lives here. Change it here; it changes everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:

    # --- Reproducibility --------------------------------------
    SEED: int = 42

    # --- Paths ------------------------------------------------
    # On Colab, after downloading from kaggle:
    #   DATA_ROOT = "/content/chest_xray"
    # If using Google Drive mount:
    #   DATA_ROOT = "/content/drive/MyDrive/datasets/chest_xray"
    DATA_ROOT: str = "/content/chest_xray"

    # Subdirectory names (match kaggle folder layout exactly)
    TRAIN_DIR: str = "train"
    VAL_DIR: str = "val"
    TEST_DIR: str = "test"

    # Checkpoints saved here every epoch so Colab disconnects don't lose progress
    CHECKPOINT_DIR: str = "/content/drive/MyDrive/checkpoints/chest_xray"

    # --- Classification ---------------------------------------
    # Binary: single sigmoid output, PNEUMONIA is the positive class (label=1)
    CLASS_NAMES: list[str] = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])
    NUM_CLASSES: int = 1    # Single output neuron for binary BCE loss

    # --- Image ------------------------------------------------
    RESIZE_SIZE: int = 256          # Resize shorter edge to this before crop
    IMG_SIZE: int = 224             # Final input size to EfficienetNetB3

    # --- DataLoader -------------------------------------------
    BATCH_SIZE: int = 32           # Safe for T4 16GB + EfficientNetB3 + AMP
    NUM_WORKERS: int = 2           # Colab limit; raises above 2 often causes hangs
    PIN_MEMORY: bool = True        # Faster pageable -> GPU transfer
    VAL_SPLIT: float = 0.20        # 20% of merged ttrain+val used as validation

    # --- Augmentation ------------------------------------------
    HFLIP_PROB: float = 0.5
    ROTATION_DEG: int = 10
    BRIGHTNESS_JITTER: float = 0.2
    CONTRAST_JITTER: float = 0.2
 
    # ── Phase 1 — head training (backbone frozen) ─────────────────────────────
    LR_PHASE1: float = 1e-3
    EPOCHS_PHASE1: int = 5
 
    # ── Phase 2 — fine-tuning (top N backbone blocks unfrozen) ────────────────
    LR_PHASE2: float = 1e-4       # 10× lower than Phase 1 — avoids catastrophic forgetting
    EPOCHS_PHASE2: int = 10
    UNFREEZE_BLOCKS: int = 3      # Unfreeze top 3 MBConv blocks (5, 6, 7)
 
    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0        # Max gradient norm; prevents exploding gradients in Phase 2
    ETA_MIN: float = 1e-6         # CosineAnnealingLR minimum LR
 
    # ── Model head ────────────────────────────────────────────────────────────
    DROPOUT: float = 0.4
    BACKBONE_OUT_FEATURES: int = 1536   # EfficientNetB3 final feature dimension
 
    # ── Loss ──────────────────────────────────────────────────────────────────
    # pos_weight is computed from training data in dataset.py (not hardcoded here)
    # Focal loss hyperparameters (used in losses.py as alternative to weighted BCE)
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.25
 
    # ── Inference ─────────────────────────────────────────────────────────────
    # Sigmoid output >= THRESHOLD → PNEUMONIA detected
    THRESHOLD: float = 0.5
 
    # ── Experiment tracking (Weights & Biases) ────────────────────────────────
    WANDB_PROJECT: str = "chest-xray-binary"
    WANDB_ENTITY: str = ""        # Your W&B username, or leave "" for default
    USE_WANDB: bool = True
 
    # ── Mixed precision ───────────────────────────────────────────────────────
    USE_AMP: bool = True          # torch.cuda.amp — ~2× faster on T4, ~50% less VRAM
 
    # ── Checkpointing ─────────────────────────────────────────────────────────
    CHECKPOINT_EVERY: int = 1     # Save every epoch (critical on Colab)
 
    # ── Early stopping & training logs ──────────────────────────────────────
    EARLY_STOP_PATIENCE: int = 3       # Stop a phase early if val AUC plateaus this many epochs
    LOG_EVERY_N_BATCHES: int = 20      # Console progress logging frequency within an epoch
 
    # ── Derived paths (computed properties, not stored fields) ─────────────────
    @property
    def train_path(self) -> Path:
        return Path(self.DATA_ROOT) / self.TRAIN_DIR
 
    @property
    def val_path(self) -> Path:
        return Path(self.DATA_ROOT) / self.VAL_DIR
 
    @property
    def test_path(self) -> Path:
        return Path(self.DATA_ROOT) / self.TEST_DIR
 
    @property
    def checkpoint_path(self) -> Path:
        return Path(self.CHECKPOINT_DIR)
 
    @property
    def total_epochs(self) -> int:
        return self.EPOCHS_PHASE1 + self.EPOCHS_PHASE2
 
    def __post_init__(self) -> None:
        assert 0 < self.VAL_SPLIT < 1, "VAL_SPLIT must be between 0 and 1"
        assert self.IMG_SIZE <= self.RESIZE_SIZE, "IMG_SIZE must be <= RESIZE_SIZE"
        assert self.LR_PHASE2 < self.LR_PHASE1, "Phase 2 LR must be lower than Phase 1"
 
 
# ── Module-level singleton ─────────────────────────────────────────────────────
# Import this object everywhere. Never instantiate Config() again.
#
#   from src.config import CONFIG
#
CONFIG = Config()