#!/usr/bin/env python3
"""
Phase 3 — CALIMERA Core Training
==================================
Architecture:
  1 MiniRocket transformer  — 1k kernels, fitted once on full T_MAX series
  T_MAX classifiers         — calibrated Ridge, trained on clf_set (70%)
  CALIMERA trigger          — KernelRidge backward loop on trigger_set (30%)

Memory strategy (avoids OOM):
  - 1k kernels  → ~6.5k features  → ~0.8 GB feature matrix (vs 70k @ 10k kernels)
  - 70/30 split like ml_edm internally (no nested cross_val_predict)
  - feature matrices deleted immediately after each timestamp

Inputs  : calimera/data/   (Phase 2 output)
Outputs : calimera/models/
    rocket.pkl              — fitted MiniRocket transformer
    classifiers.pkl         — list[T_MAX] of calibrated classifiers
    trigger.pkl             — fitted CALIMERA trigger model
    cost_matrices.pkl       — CostMatrices at alpha=ALPHA
    training_report.json    — per-timestamp val accuracy + overall metrics

Run:
    cd /path/to/earlyflow
    python calimera/phase3_calimera_train.py
"""

import sys, json, pickle, gc
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("ml_edm/src")))
from ml_edm.trigger._calimera import CALIMERA
from ml_edm.cost_matrices import CostMatrices

from sklearn.linear_model import RidgeClassifierCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import MiniRocket
try:
    from sklearn.frozen import FrozenEstimator
    _FROZEN = True
except ImportError:
    _FROZEN = False

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("calimera/data")
MODEL_DIR = Path("calimera/models")

ALPHA            = 0.5      # delay-cost weight; sweep this in Phase 5
N_KERNELS        = 1_000    # MiniRocket kernels (6.5k features; manageable memory)
TRIGGER_FRAC     = 0.30     # fraction of train used for CALIMERA trigger training
RIDGE_ALPHAS     = np.logspace(-3, 3, 10)
RANDOM_STATE     = 42


def mask_after(X_sktime: np.ndarray, t: int) -> np.ndarray:
    """Zero out channels after position t.  X_sktime: (n, F, T_MAX)."""
    Xm = X_sktime.copy()
    Xm[:, :, t:] = 0.0
    return Xm


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Phase 2 data ────────────────────────────────────────────────────
    print("[0] Loading data …")
    X_train = np.load(DATA_DIR / "X_train.npy")   # (n_train, T_MAX, F)
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")

    with open(DATA_DIR / "meta.json") as fh:
        meta = json.load(fh)

    T_MAX     = meta["T_MAX"]
    n_classes = len(meta["label_map"])
    label_map = meta["label_map"]
    timestamps = np.arange(1, T_MAX + 1)

    print(f"  Train {len(X_train):,} | Val {len(X_val):,}")
    print(f"  T_MAX={T_MAX}  N_FEAT={meta['N_FEAT']}  n_classes={n_classes}")

    # ── 70 / 30 split: classifiers vs CALIMERA trigger ───────────────────────
    # This mirrors ml_edm's trigger_proportion design:
    # classifiers are trained on clf_set; trigger model sees trigger_set
    # so trigger probabilities are unbiased (no data leakage).
    idx = np.arange(len(X_train))
    idx_clf, idx_trig = train_test_split(
        idx, test_size=TRIGGER_FRAC, stratify=y_train, random_state=RANDOM_STATE
    )
    X_clf   = X_train[idx_clf];   y_clf   = y_train[idx_clf]
    X_trig  = X_train[idx_trig];  y_trig  = y_train[idx_trig]
    print(f"  Clf set {len(idx_clf):,} | Trigger set {len(idx_trig):,}")

    # sktime format: (n, F, T)
    X_clf_sk  = X_clf.transpose(0, 2, 1).astype(np.float32)
    X_trig_sk = X_trig.transpose(0, 2, 1).astype(np.float32)
    X_val_sk  = X_val.transpose(0, 2, 1).astype(np.float32)

    # ── Step 1: Fit MiniRocket on full clf set ───────────────────────────────
    print(f"\n[1] Fitting MiniRocket ({N_KERNELS} kernels) …")
    rocket = MiniRocket(num_kernels=N_KERNELS, random_state=RANDOM_STATE)
    rocket.fit(X_clf_sk)
    n_feat = rocket.transform(X_clf_sk[:2]).shape[1]
    print(f"  Feature dim: {n_feat}  |  matrix size ≈ {len(X_clf)*n_feat*4/1e9:.2f} GB")

    # ── Step 2: Train T_MAX classifiers ──────────────────────────────────────
    print(f"\n[2] Training {T_MAX} classifiers …")
    classifiers = []
    trig_probas = np.zeros((len(X_trig), T_MAX, n_classes), dtype=np.float32)
    val_probas  = np.zeros((len(X_val),  T_MAX, n_classes), dtype=np.float32)
    val_accs    = []

    for t in range(1, T_MAX + 1):
        # Feature extraction for this timestamp
        feat_clf  = rocket.transform(mask_after(X_clf_sk,  t))
        feat_trig = rocket.transform(mask_after(X_trig_sk, t))
        feat_val  = rocket.transform(mask_after(X_val_sk,  t))

        # 1. Fit Ridge on clf set
        base = RidgeClassifierCV(alphas=RIDGE_ALPHAS)
        base.fit(feat_clf, y_clf)

        # 2. Calibrate (Platt scaling) on trigger set — calibrator sees trigger labels
        frozen = FrozenEstimator(base) if _FROZEN else base
        clf = CalibratedClassifierCV(frozen, method="sigmoid")
        clf.fit(feat_trig, y_trig)
        classifiers.append(clf)

        # 3. Probabilities on trigger set (for CALIMERA backward loop)
        trig_probas[:, t - 1, :] = clf.predict_proba(feat_trig).astype(np.float32)

        # 4. Val accuracy
        val_p = clf.predict_proba(feat_val).astype(np.float32)
        val_probas[:, t - 1, :] = val_p
        acc = (val_p.argmax(axis=1) == y_val).mean()
        val_accs.append(float(acc))

        print(f"  t={t:02d}/{T_MAX}  val_acc={acc:.4f}")

        # Free large arrays immediately to keep memory low
        del feat_clf, feat_trig, feat_val
        gc.collect()

    # ── Step 3: CALIMERA backward loop ───────────────────────────────────────
    print(f"\n[3] Training CALIMERA trigger (α={ALPHA}) …")

    # CostMatrices: α * misclassification + (1-α) * delay
    cost_matrices = CostMatrices(
        timestamps=timestamps,
        n_classes=n_classes,
        alpha=ALPHA,
        delay_cost=lambda t: (1.0 - ALPHA) * float(t) / float(T_MAX),
    )

    # BaseTriggerModel.fit() reshapes X to 2D internally via check_X_y
    X_trig_2d = X_trig.reshape(len(X_trig), -1)

    trigger = CALIMERA(timestamps=timestamps)
    trigger.fit(X_trig_2d, trig_probas, y_trig, cost_matrices)
    print(f"  Backward loop done — {len(trigger.halters)} halters trained.")

    # ── Step 4: Simulate online classification on val set ────────────────────
    print("\n[4] Evaluating on val set (simulate online flow) …")
    X_val_2d  = X_val.reshape(len(X_val), -1)
    all_preds  = np.full(len(X_val), -1, dtype=int)
    all_t_star = np.full(len(X_val), T_MAX, dtype=int)
    decided    = np.zeros(len(X_val), dtype=bool)

    for t_idx, t in enumerate(timestamps):
        probas_t  = val_probas[:, t_idx, :]       # (n_val, n_classes)
        cur_preds = probas_t.argmax(axis=1)

        if t == T_MAX:
            mask = ~decided                        # force classify remaining flows
        else:
            triggers = trigger.predict(X_val_2d, probas_t, cost_matrices)
            mask = triggers & ~decided

        all_preds[mask]  = cur_preds[mask]
        all_t_star[mask] = t
        decided         |= mask
        if decided.all():
            break

    # safety net
    still_open = ~decided
    if still_open.any():
        all_preds[still_open]  = val_probas[still_open, -1, :].argmax(axis=1)
        all_t_star[still_open] = T_MAX

    accuracy  = (all_preds == y_val).mean()
    earliness = all_t_star.mean() / T_MAX
    hm        = 2 * accuracy * (1 - earliness) / (accuracy + (1 - earliness) + 1e-9)

    print(f"\n  Accuracy  : {accuracy:.4f}")
    print(f"  Earliness : {earliness:.4f}  (lower = earlier)")
    print(f"  HM        : {hm:.4f}  (higher = better)")
    print(f"  Baseline (classify at t=T_MAX): acc={val_accs[-1]:.4f}  earliness=1.0  HM={val_accs[-1]:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\n[5] Saving …")
    with open(MODEL_DIR / "rocket.pkl", "wb") as fh:
        pickle.dump(rocket, fh)
    with open(MODEL_DIR / "classifiers.pkl", "wb") as fh:
        pickle.dump(classifiers, fh)
    with open(MODEL_DIR / "trigger.pkl", "wb") as fh:
        pickle.dump(trigger, fh)
    with open(MODEL_DIR / "cost_matrices.pkl", "wb") as fh:
        pickle.dump(cost_matrices, fh)

    report = {
        "alpha": ALPHA,
        "n_kernels": N_KERNELS,
        "trigger_frac": TRIGGER_FRAC,
        "val_accuracy":  round(accuracy, 5),
        "val_earliness": round(earliness, 5),
        "val_hm":        round(hm, 5),
        "baseline_acc":  round(val_accs[-1], 5),
        "per_timestamp_val_acc": {
            str(t): round(a, 5) for t, a in zip(timestamps, val_accs)
        },
        "label_map": label_map,
    }
    with open(MODEL_DIR / "training_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n✓  Saved to {MODEL_DIR}/")
    print(f"   acc={accuracy:.4f}  earliness={earliness:.4f}  HM={hm:.4f}")


if __name__ == "__main__":
    main()
