#!/usr/bin/env python3
"""
Triggered F1 evaluation for ALL models
=======================================
For each model (MiniRocket, LSTM, GRU, Transformer, TCN):
  1. Build test and trigger-set probas at every t
  2. Fit CALIMERA trigger on the trigger set
  3. Simulate online trigger on test set
  4. Compute classification_report on triggered predictions only

Outputs -> calimera/eval/
  triggered_f1_all_models.txt    -- per-model classification reports
  triggered_summary_table.txt    -- comparison table (macro F1, earliness, accuracy)

Run:
    cd /home/sai/Downloads/early_ml_final
    python calimera/eval_triggered_all_models.py
"""

import sys, json, pickle, gc
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path("ml_edm/src")))

import calimera.config as cfg
from calimera.models import get_model, CalibratedModel
from ml_edm.trigger._calimera import CALIMERA
from ml_edm.cost_matrices import CostMatrices

DATA_DIR  = Path(cfg.DATA_DIR)
MODEL_DIR = Path(cfg.MODEL_DIR)
EVAL_DIR  = Path(cfg.EVAL_DIR)

BEST_ALPHA   = cfg.ALPHA_DEFAULT
TRIGGER_FRAC = cfg.TRIGGER_FRAC
SEED         = cfg.SEED
CHUNK        = 512

ALL_MODELS = ["minirocket", "lstm", "gru", "transformer", "tcn"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def mask_after(X_sk: np.ndarray, t: int) -> np.ndarray:
    Xm = X_sk.copy()
    Xm[:, :, t:] = 0.0
    return Xm


def build_probas_minirocket(rocket, classifiers, X_np, T_MAX, n_classes):
    X_sk = X_np.transpose(0, 2, 1).astype(np.float32)
    n = X_sk.shape[0]
    probas = np.zeros((n, T_MAX, n_classes), dtype=np.float32)
    for t in range(1, T_MAX + 1):
        feat = rocket.transform(mask_after(X_sk, t))
        probas[:, t - 1, :] = classifiers[t - 1].predict_proba(feat).astype(np.float32)
        del feat; gc.collect()
        print(f"    t={t:02d}/{T_MAX}", end="\r", flush=True)
    print()
    return probas


def build_probas_deep(cal, X_np, T_MAX, n_classes):
    n = len(X_np)
    probas = np.zeros((n, T_MAX, n_classes), dtype=np.float32)
    X_t = torch.from_numpy(X_np).float()
    for t in range(1, T_MAX + 1):
        chunks = [cal.predict_proba(X_t[i:i + CHUNK], t) for i in range(0, n, CHUNK)]
        probas[:, t - 1, :] = np.concatenate(chunks, axis=0)
        print(f"    t={t:02d}/{T_MAX}", end="\r", flush=True)
    print()
    return probas


def simulate_trigger(trig_probas, test_probas, y_trig, y_test, timestamps, T_MAX, n_classes):
    cost_matrices = CostMatrices(
        timestamps=timestamps, n_classes=n_classes, alpha=BEST_ALPHA,
        delay_cost=lambda t, a=BEST_ALPHA: (1.0 - a) * float(t) / float(T_MAX),
    )
    X_trig_dummy = np.zeros((len(y_trig), T_MAX))
    trigger = CALIMERA(timestamps=timestamps)
    trigger.fit(X_trig_dummy, trig_probas, y_trig, cost_matrices)

    n = len(y_test)
    all_preds  = np.full(n, -1, dtype=int)
    all_t_star = np.full(n, T_MAX, dtype=int)
    decided    = np.zeros(n, dtype=bool)

    for t_idx, t in enumerate(timestamps):
        probas_t  = test_probas[:, t_idx, :]
        cur_preds = probas_t.argmax(axis=1)
        if t == T_MAX:
            mask = ~decided
        else:
            X_at_t   = np.zeros((n, t), dtype=np.float32)
            triggers = trigger.predict(X_at_t, probas_t, cost_matrices)
            mask = triggers & ~decided
        all_preds[mask]  = cur_preds[mask]
        all_t_star[mask] = t
        decided         |= mask
        if decided.all():
            break

    still_open = ~decided
    if still_open.any():
        all_preds[still_open]  = test_probas[still_open, -1, :].argmax(axis=1)
        all_t_star[still_open] = T_MAX

    return all_preds, all_t_star


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("[0] Loading data …")
    X_test  = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
    y_test  = np.load(DATA_DIR / "y_test.npy").astype(np.int32)
    X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(DATA_DIR / "y_train.npy").astype(np.int32)

    with open(DATA_DIR / "meta.json") as fh:
        meta = json.load(fh)

    label_map   = meta["label_map"]
    class_names = [label_map[str(i)] for i in range(len(label_map))]
    T_MAX       = meta["T_MAX"]
    n_classes   = len(label_map)
    timestamps  = np.arange(1, T_MAX + 1)
    print(f"  Test {len(X_test):,} | T_MAX={T_MAX} | classes={class_names}")

    # Derive trigger set (same split as training)
    idx = np.arange(len(X_train))
    _, idx_trig = train_test_split(
        idx, test_size=TRIGGER_FRAC, stratify=y_train, random_state=SEED
    )
    X_trig = X_train[idx_trig]
    y_trig = y_train[idx_trig]
    print(f"  Trigger set: {len(y_trig):,} flows\n")

    summary_rows = []
    all_reports  = []

    for model_name in ALL_MODELS:
        print(f"{'='*60}")
        print(f"  Model: {model_name.upper()}")
        print(f"{'='*60}")

        if model_name == "minirocket":
            with open(MODEL_DIR / "rocket.pkl", "rb") as fh:
                rocket = pickle.load(fh)
            with open(MODEL_DIR / "classifiers.pkl", "rb") as fh:
                classifiers = pickle.load(fh)

            print("  Building trig probas …")
            trig_probas = build_probas_minirocket(rocket, classifiers, X_trig, T_MAX, n_classes)
            print("  Building test probas …")
            test_probas = build_probas_minirocket(rocket, classifiers, X_test, T_MAX, n_classes)
            del rocket, classifiers
        else:
            model = get_model(model_name).to(cfg.DEVICE)
            cal   = CalibratedModel(model)
            cal.load(str(MODEL_DIR / f"cal_{model_name}"))
            print(f"  Loaded cal_{model_name}.pt + .platt.pkl")

            print("  Building trig probas …")
            trig_probas = build_probas_deep(cal, X_trig, T_MAX, n_classes)
            print("  Building test probas …")
            test_probas = build_probas_deep(cal, X_test, T_MAX, n_classes)
            del model, cal
        gc.collect()

        print("  Fitting CALIMERA trigger & simulating on test set …")
        trig_preds, t_star = simulate_trigger(
            trig_probas, test_probas, y_trig, y_test, timestamps, T_MAX, n_classes
        )
        del trig_probas, test_probas; gc.collect()

        avg_t_star = float(t_star.mean())
        earliness  = avg_t_star / T_MAX
        macro_f1   = f1_score(y_test, trig_preds, average="macro", zero_division=0)
        accuracy   = float((trig_preds == y_test).mean())

        report_header = (
            f"Model: {model_name.upper()}   (alpha={BEST_ALPHA})\n"
            f"Avg t* = {avg_t_star:.2f} / {T_MAX}  |  "
            f"earliness = {earliness:.4f}  |  "
            f"macro-F1 = {macro_f1:.4f}  |  "
            f"accuracy = {accuracy:.4f}\n"
            f"{'='*60}\n"
        )
        report_body = classification_report(
            y_test, trig_preds, target_names=class_names, zero_division=0
        )
        full_report = report_header + report_body
        all_reports.append(full_report)
        print(full_report)

        summary_rows.append({
            "model":      model_name,
            "avg_t_star": avg_t_star,
            "earliness":  earliness,
            "accuracy":   accuracy,
            "macro_f1":   macro_f1,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    header = (
        f"{'Model':<14} {'Avg t*':>8} {'Earliness':>10} "
        f"{'Accuracy':>10} {'Macro F1':>10}"
    )
    sep  = "-" * len(header)
    rows = [
        f"{r['model']:<14} {r['avg_t_star']:>8.2f} {r['earliness']:>10.4f} "
        f"{r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}"
        for r in summary_rows
    ]
    summary = (
        f"\nCALIMERA-Triggered Classification Summary — TEST SET  (alpha={BEST_ALPHA})\n"
        f"{'='*len(header)}\n"
        f"{header}\n{sep}\n"
        + "\n".join(rows)
        + f"\n{sep}\n"
    )
    print(summary)

    # ── Save outputs ──────────────────────────────────────────────────────────
    combined = "\n\n".join(all_reports) + "\n" + summary
    with open(EVAL_DIR / "triggered_f1_all_models.txt", "w") as fh:
        fh.write(combined)
    with open(EVAL_DIR / "triggered_summary_table.txt", "w") as fh:
        fh.write(summary)

    print(f"Saved: {EVAL_DIR}/triggered_f1_all_models.txt")
    print(f"Saved: {EVAL_DIR}/triggered_summary_table.txt")


if __name__ == "__main__":
    main()
