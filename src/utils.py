"""
utils.py — Reproducibility, checkpointing, logging, and device helpers.

These are pure utilities with no ML logic. Every other module depends on this one,
so it is written first and has zero internal imports from src/.

Key functions:
    seed_everything(seed)        — deterministic runs across CPU + GPU + Python
    get_device()                 — returns torch.device, logs GPU name + VRAM
    save_checkpoint(...)         — serialises model + optimizer + metadata
    load_checkpoint(...)         — restores state for resume or inference
    setup_logging(level)         — configures root logger
    print_auc_table(...)         — formatted AUC summary after each val epoch
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA (single + multi-GPU),
    PYTHONHASHSEED, and cuDNN determinism.

    Trade-off: deterministic=True disables some non-deterministic CUDA kernels,
    which can slow training by ~5–10%. Set benchmark=False to prevent cuDNN from
    picking a faster-but-non-deterministic algorithm between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # Harmless if only one GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     # Must be False for determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info(f"All random seeds set to {seed}")


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available device and log its details.

    On Colab T4:
        Device: Tesla T4 | VRAM: 16.0 GB | CUDA: 12.x
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        cuda_ver = torch.version.cuda
        log.info(
            f"Device: {props.name} | VRAM: {vram_gb:.1f} GB | CUDA: {cuda_ver}"
        )
    else:
        log.warning(
            "No CUDA GPU found. Training EfficientNetB3 on CPU will take hours "
            "per epoch. Use Google Colab or Kaggle Notebooks for GPU access."
        )

    return device


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_auc: float,
    config: Any,
    save_dir: Path,
    is_best: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    """
    Serialise model + optimizer + metadata to disk.

    Always writes:
        <save_dir>/last_model.pth   — latest epoch; overwritten every call

    Writes only when is_best=True:
        <save_dir>/best_model.pth   — best val AUC so far

    The scaler state is included so AMP training resumes cleanly after a
    Colab disconnect. Without it, the first resumed epoch can have
    incorrect gradient scaling and produce a NaN loss spike.

    Args:
        model:      Trained model (DataParallel-safe via .module fallback).
        optimizer:  AdamW instance with current state.
        epoch:      Current epoch index (0-based).
        best_auc:   Best validation AUC seen so far.
        config:     CONFIG dataclass instance — stored for reproducibility.
        save_dir:   Directory to write .pth files into.
        is_best:    If True, also overwrites best_model.pth.
        scaler:     Optional GradScaler for AMP resume.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Handle DataParallel wrapping (not needed on Colab single-GPU, but safe)
    model_state = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )

    state: Dict[str, Any] = {
        "epoch": epoch,
        "best_auc": best_auc,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }

    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()

    last_path = save_dir / "last_model.pth"
    torch.save(state, last_path)
    log.info(f"Checkpoint saved → {last_path}  (epoch={epoch+1}, AUC={best_auc:.4f})")

    if is_best:
        best_path = save_dir / "best_model.pth"
        torch.save(state, best_path)
        log.info(f"Best model updated → {best_path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Restore model (and optionally optimizer + scaler) from a checkpoint.

    Returns a metadata dict so the training loop knows where to resume:
        {
            "epoch":    int,    # last completed epoch (resume from epoch+1)
            "best_auc": float,
            "config":   Config | None,
        }

    Args:
        path:       Full path to the .pth file.
        model:      Model instance — weights loaded in-place.
        optimizer:  If provided, optimizer state is restored (set None for
                    inference-only loading).
        scaler:     If provided, GradScaler state is restored.
        device:     Map location for weights. Defaults to current CUDA/CPU.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Did you mean last_model.pth or best_model.pth?"
        )

    state = torch.load(path, map_location=device)

    # Handle DataParallel wrapping
    if hasattr(model, "module"):
        model.module.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state["model_state"])

    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])

    if scaler is not None and "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])

    log.info(
        f"Loaded checkpoint from {path} "
        f"(epoch={state['epoch']+1}, AUC={state['best_auc']:.4f})"
    )

    return {
        "epoch": state["epoch"],
        "best_auc": state["best_auc"],
        "config": state.get("config"),
    }


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a timestamped format.

    Call this once at the top of any entry-point script (run_training.py, etc.)
    before importing other modules. Subsequent getLogger() calls in other
    modules inherit this configuration.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,    # Override any earlier basicConfig calls (e.g., from libraries)
    )


# ── AUC reporting ─────────────────────────────────────────────────────────────

def print_auc_table(
    classes: list[str],
    auc_scores: list[float],
    mean_auc: float,
    epoch: Optional[int] = None,
) -> None:
    """
    Print a formatted per-class AUC table to stdout after each validation epoch.

    Example output:
        Epoch 3 — Validation AUC
        ┌───────────────┬────────┐
        │ Class         │  AUC   │
        ├───────────────┼────────┤
        │ NORMAL        │ 0.8921 │
        │ PNEUMONIA     │ 0.8921 │
        ├───────────────┼────────┤
        │ MEAN          │ 0.8921 │
        └───────────────┴────────┘
    """
    if epoch is not None:
        print(f"\nEpoch {epoch + 1} — Validation AUC")

    col_w = max(len(c) for c in classes) + 2
    inner_w = col_w + 1      # account for leading space in cell

    top    = f"┌{'─' * inner_w}┬{'─' * 8}┐"
    header = f"│ {'Class':<{col_w - 1}}│  AUC   │"
    sep    = f"├{'─' * inner_w}┼{'─' * 8}┤"
    bottom = f"└{'─' * inner_w}┴{'─' * 8}┘"

    print(top)
    print(header)
    print(sep)
    for cls, auc in zip(classes, auc_scores):
        print(f"│ {cls:<{col_w - 1}}│ {auc:.4f} │")
    print(sep)
    print(f"│ {'MEAN':<{col_w - 1}}│ {mean_auc:.4f} │")
    print(bottom)


# ── Training utilities ────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and frozen parameters separately.

    Useful for confirming Phase 1 (only head trained) vs Phase 2
    (head + top backbone blocks trained).

    Returns:
        {"trainable": N, "frozen": M, "total": N+M}
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(
        f"Parameters — trainable: {trainable:,} | frozen: {frozen:,} | "
        f"total: {trainable + frozen:,}"
    )
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}