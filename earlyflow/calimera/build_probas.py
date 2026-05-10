#!/usr/bin/env python3
"""
EarlyFlow — CALIMERA bridge
============================
Loads the trained + calibrated deep model and produces:
  calimera/data/trig_probas_deep.npy   (n_trig, T_MAX, N_CLASSES)  float32
  calimera/data/val_probas_deep.npy    (n_val,  T_MAX, N_CLASSES)  float32
  calimera/data/y_trig.npy             (n_trig,)                   int32

These drop-in replace the MiniRocket-derived probas in phase45_sweep_eval.py
and phase6b_f1_eval.py (3-line edit each, already done).

Usage:
    cd /path/to/earlyflow
    python calimera/build_probas.py                    # uses MODEL from config.py
    python calimera/build_probas.py --model transformer
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import calimera.config as cfg
from calimera.models import get_model, CalibratedModel

torch.manual_seed(cfg.SEED)

CHUNK = 512   # flows per GPU batch to avoid OOM


def build_probas_for_split(cal: CalibratedModel, X_np: np.ndarray) -> np.ndarray:
    """Returns (n, T_MAX, N_CLASSES) float32 probabilities."""
    n      = len(X_np)
    probas = np.zeros((n, cfg.T_MAX, cfg.N_CLASSES), dtype=np.float32)
    X_t    = torch.from_numpy(X_np).float()

    for t in range(1, cfg.T_MAX + 1):
        chunks = []
        for i in range(0, n, CHUNK):
            chunks.append(cal.predict_proba(X_t[i:i + CHUNK], t))
        probas[:, t - 1, :] = np.concatenate(chunks, axis=0)
        print(f"  t={t:02d}/{cfg.T_MAX}", end="\r", flush=True)
    print()
    return probas


def main():
    args = argparse.ArgumentParser(description="Build CALIMERA probas from deep model")
    args.add_argument("--model", default=None,
                      choices=["lstm", "gru", "transformer", "tcn"],
                      help="Model to load (default: from config.py)")
    args = args.parse_args()
    if args.model:
        cfg.MODEL = args.model

    data_dir  = Path(cfg.DATA_DIR)
    model_dir = Path(cfg.MODEL_DIR)

    # ── Load calibrated model ──────────────────────────────────────
    print(f"[0] Loading calibrated {cfg.MODEL} model …")
    model    = get_model().to(cfg.DEVICE)
    cal      = CalibratedModel(model)
    cal_path = str(model_dir / f"cal_{cfg.MODEL}")
    cal.load(cal_path)
    print(f"  Loaded {cal_path}.pt + .platt.pkl")

    # ── Re-derive trigger set (same seed as training) ──────────────
    print("\n[1] Deriving trigger set …")
    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy").astype(np.int32)

    idx = np.arange(len(X_train))
    _, idx_trig = train_test_split(
        idx, test_size=cfg.TRIGGER_FRAC, stratify=y_train, random_state=cfg.SEED
    )
    X_trig = X_train[idx_trig]
    y_trig = y_train[idx_trig]
    print(f"  Trigger set: {len(idx_trig):,} flows")

    # ── Build probas ───────────────────────────────────────────────
    print(f"\n[2] Building trig_probas ({len(X_trig):,} × {cfg.T_MAX} timestamps) …")
    trig_probas = build_probas_for_split(cal, X_trig)

    X_val = np.load(data_dir / "X_val.npy").astype(np.float32)
    print(f"\n[3] Building val_probas ({len(X_val):,} × {cfg.T_MAX} timestamps) …")
    val_probas = build_probas_for_split(cal, X_val)

    # ── Save ───────────────────────────────────────────────────────
    print("\n[4] Saving …")
    np.save(data_dir / "trig_probas_deep.npy", trig_probas)
    np.save(data_dir / "val_probas_deep.npy",  val_probas)
    np.save(data_dir / "y_trig.npy",           y_trig)

    print(f"  trig_probas_deep.npy  {trig_probas.shape}")
    print(f"  val_probas_deep.npy   {val_probas.shape}")
    print(f"  y_trig.npy            {y_trig.shape}")
    print(f"\nNext: python calimera/phase45_sweep_eval.py")


if __name__ == "__main__":
    main()
