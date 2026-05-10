#!/usr/bin/env python3
"""
Phase 6b — Per-timestamp F1 / Precision / Recall Evaluation
=============================================================
For every packet timestamp t=1..T_MAX:
  - run saved classifiers on val set
  - compute per-class and macro P / R / F1

Also evaluates CALIMERA at the best-alpha trigger point.

Outputs  →  calimera/eval/
  f1_per_timestamp.json        — full metrics dict
  plot_macro_prf1.png          — macro P / R / F1 vs packets
  plot_perclass_f1.png         — per-class F1 vs packets
  plot_perclass_precision.png  — per-class precision vs packets
  plot_perclass_recall.png     — per-class recall vs packets
  report_f1_full.txt           — sklearn classification_report at every t
  report_f1_triggered.txt      — classification_report on CALIMERA-triggered val set

Run:
    cd /path/to/earlyflow
    python calimera/phase6b_f1_eval.py
"""

import sys, json, pickle, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support

sys.path.insert(0, str(Path("ml_edm/src")))
from ml_edm.trigger._calimera import CALIMERA
from ml_edm.cost_matrices import CostMatrices

DATA_DIR  = Path("calimera/data")
MODEL_DIR = Path("calimera/models")
EVAL_DIR  = Path("calimera/eval")

BEST_ALPHA   = 0.5     # from sweep — highest HM
TRIGGER_FRAC = 0.30
RANDOM_STATE = 42

# ── Switch: set True to use deep model probas from build_probas.py ────────────
USE_DEEP = True    # False → MiniRocket (phase3); True → deep model (train_deep + build_probas)

# ── Helpers ───────────────────────────────────────────────────────────────────
def mask_after(X_sk: np.ndarray, t: int) -> np.ndarray:
    Xm = X_sk.copy()
    Xm[:, :, t:] = 0.0
    return Xm


def load_data():
    print("[0] Loading data and models …")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    with open(DATA_DIR / "meta.json") as fh:
        meta = json.load(fh)
    with open(MODEL_DIR / "rocket.pkl", "rb") as fh:
        rocket = pickle.load(fh)
    with open(MODEL_DIR / "classifiers.pkl", "rb") as fh:
        classifiers = pickle.load(fh)

    label_map  = meta["label_map"]           # {"0": "cloud", ...}
    class_names = [label_map[str(i)] for i in range(len(label_map))]
    T_MAX      = meta["T_MAX"]
    n_classes  = len(label_map)
    timestamps = np.arange(1, T_MAX + 1)
    print(f"  Val {len(X_val):,} | T_MAX={T_MAX} | classes={class_names}")
    return (X_val, y_val, X_train, y_train,
            meta, rocket, classifiers,
            class_names, T_MAX, n_classes, timestamps)


# ── Step 1: compute val probas at every timestamp ─────────────────────────────
def build_val_probas(rocket, classifiers, X_val_sk, T_MAX, n_classes):
    print(f"\n[1] Computing val probabilities at each of {T_MAX} timestamps …")
    n = X_val_sk.shape[0]
    val_probas = np.zeros((n, T_MAX, n_classes), dtype=np.float32)
    for t in range(1, T_MAX + 1):
        feat = rocket.transform(mask_after(X_val_sk, t))
        val_probas[:, t - 1, :] = classifiers[t - 1].predict_proba(feat).astype(np.float32)
        del feat; gc.collect()
        print(f"  t={t:02d}/{T_MAX}", end="\r", flush=True)
    print()
    return val_probas


# ── Step 2: per-timestamp metrics ─────────────────────────────────────────────
def compute_per_timestamp_metrics(val_probas, y_val, class_names, T_MAX):
    print("\n[2] Computing per-timestamp P / R / F1 …")
    records = {}     # t -> {macro_p, macro_r, macro_f1, per_class: {cls: {p,r,f1}}}
    full_report_lines = []

    for t in range(1, T_MAX + 1):
        preds = val_probas[:, t - 1, :].argmax(axis=1)

        p, r, f, _ = precision_recall_fscore_support(
            y_val, preds, average=None, labels=list(range(len(class_names))),
            zero_division=0
        )
        mp, mr, mf, _ = precision_recall_fscore_support(
            y_val, preds, average="macro", zero_division=0
        )

        per_class = {
            cls: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i, cls in enumerate(class_names)
        }
        records[t] = {
            "macro_precision": float(mp),
            "macro_recall":    float(mr),
            "macro_f1":        float(mf),
            "per_class":       per_class,
        }

        rpt = classification_report(y_val, preds, target_names=class_names,
                                    zero_division=0)
        full_report_lines.append(f"{'='*55}\n  t = {t:02d} packets\n{'='*55}\n{rpt}")

    return records, "\n".join(full_report_lines)


# ── Step 3: CALIMERA-triggered predictions ────────────────────────────────────
def triggered_predictions(val_probas, y_train, y_val, timestamps, T_MAX, n_classes):
    print(f"\n[3] Simulating CALIMERA online trigger (α={BEST_ALPHA}) …")
    from sklearn.model_selection import train_test_split

    # rebuild trigger-set probas from trig classifiers via saved classifiers
    # (we don't have trig_probas on disk, so use a small workaround:
    #  re-split train, derive trig idx, load X_train, rebuild trig probas)
    X_train = np.load(DATA_DIR / "X_train.npy")
    with open(MODEL_DIR / "rocket.pkl", "rb") as fh:
        rocket = pickle.load(fh)
    with open(MODEL_DIR / "classifiers.pkl", "rb") as fh:
        classifiers = pickle.load(fh)

    idx = np.arange(len(X_train))
    idx_clf, idx_trig = train_test_split(
        idx, test_size=TRIGGER_FRAC, stratify=y_train, random_state=RANDOM_STATE
    )
    X_trig    = X_train[idx_trig]
    y_trig    = y_train[idx_trig]
    X_trig_sk = X_trig.transpose(0, 2, 1).astype(np.float32)

    n_trig = len(y_trig)
    trig_probas = np.zeros((n_trig, T_MAX, n_classes), dtype=np.float32)
    print("  Building trigger-set probas …")
    for t in range(1, T_MAX + 1):
        feat = rocket.transform(mask_after(X_trig_sk, t))
        trig_probas[:, t - 1, :] = classifiers[t - 1].predict_proba(feat).astype(np.float32)
        del feat; gc.collect()

    cost_matrices = CostMatrices(
        timestamps=timestamps, n_classes=n_classes, alpha=BEST_ALPHA,
        delay_cost=lambda t, a=BEST_ALPHA: (1.0 - a) * float(t) / float(T_MAX),
    )
    X_trig_dummy = np.zeros((n_trig, T_MAX))
    trigger = CALIMERA(timestamps=timestamps)
    trigger.fit(X_trig_dummy, trig_probas, y_trig, cost_matrices)

    # simulate online
    n = len(y_val)
    all_preds  = np.full(n, -1, dtype=int)
    all_t_star = np.full(n, T_MAX, dtype=int)
    decided    = np.zeros(n, dtype=bool)

    for t_idx, t in enumerate(timestamps):
        probas_t  = val_probas[:, t_idx, :]
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
        all_preds[still_open]  = val_probas[still_open, -1, :].argmax(axis=1)
        all_t_star[still_open] = T_MAX

    return all_preds, all_t_star


# ── Step 4: plots ─────────────────────────────────────────────────────────────
def plot_macro(records, T_MAX, avg_t_star):
    ts  = list(range(1, T_MAX + 1))
    mp  = [records[t]["macro_precision"] for t in ts]
    mr  = [records[t]["macro_recall"]    for t in ts]
    mf  = [records[t]["macro_f1"]        for t in ts]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, mp, "o-", lw=2, label="Macro Precision", color="steelblue")
    ax.plot(ts, mr, "s-", lw=2, label="Macro Recall",    color="darkorange")
    ax.plot(ts, mf, "D-", lw=2, label="Macro F1",        color="seagreen")
    ax.axvline(avg_t_star, color="red", ls="--", lw=1.5,
               label=f"Avg CALIMERA trigger t★={avg_t_star:.1f} (α={BEST_ALPHA})")
    ax.set_xlabel("Packets seen (t)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Macro-averaged Precision / Recall / F1 vs Packets seen\n(val set)", fontsize=12)
    ax.set_xlim(0.5, T_MAX + 0.5)
    ax.set_xticks(ts)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = EVAL_DIR / "plot_macro_prf1.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_per_class(records, class_names, T_MAX, avg_t_star, metric, cmap_name="tab10"):
    ts  = list(range(1, T_MAX + 1))
    cmap = plt.get_cmap(cmap_name)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cls in enumerate(class_names):
        vals = [records[t]["per_class"][cls][metric] for t in ts]
        ax.plot(ts, vals, "o-", lw=2, label=cls, color=cmap(i))
    ax.axvline(avg_t_star, color="red", ls="--", lw=1.5,
               label=f"Avg CALIMERA trigger t★={avg_t_star:.1f}")
    ax.set_xlabel("Packets seen (t)", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"Per-class {metric.capitalize()} vs Packets seen\n(val set)", fontsize=12)
    ax.set_xlim(0.5, T_MAX + 0.5)
    ax.set_xticks(ts)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"plot_perclass_{metric}.png"
    p = EVAL_DIR / fname
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    (X_val, y_val, X_train, y_train,
     meta, rocket, classifiers,
     class_names, T_MAX, n_classes, timestamps) = load_data()

    # Step 1
    if USE_DEEP:
        print("\n[1] Loading deep-model val probas from build_probas.py output …")
        val_probas = np.load(DATA_DIR / "val_probas_deep.npy")
        print(f"  val_probas {val_probas.shape}")
    else:
        X_val_sk   = X_val.transpose(0, 2, 1).astype(np.float32)
        val_probas = build_val_probas(rocket, classifiers, X_val_sk, T_MAX, n_classes)

    # Step 2
    records, full_report_text = compute_per_timestamp_metrics(
        val_probas, y_val, class_names, T_MAX)

    # Step 3
    trig_preds, t_star = triggered_predictions(
        val_probas, y_train, y_val, timestamps, T_MAX, n_classes)
    avg_t_star = float(t_star.mean())

    triggered_report = (
        f"CALIMERA-triggered classification report  (α={BEST_ALPHA})\n"
        f"avg t★ = {avg_t_star:.2f} / {T_MAX}   "
        f"(earliness = {avg_t_star/T_MAX:.4f})\n"
        f"{'='*55}\n"
        + classification_report(y_val, trig_preds, target_names=class_names, zero_division=0)
    )

    # Step 4: plots
    print("\n[4] Generating plots …")
    plot_macro(records, T_MAX, avg_t_star)
    plot_per_class(records, class_names, T_MAX, avg_t_star, "f1")
    plot_per_class(records, class_names, T_MAX, avg_t_star, "precision")
    plot_per_class(records, class_names, T_MAX, avg_t_star, "recall")

    # Save JSON
    out_json = {str(t): v for t, v in records.items()}
    with open(EVAL_DIR / "f1_per_timestamp.json", "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"  Saved {EVAL_DIR}/f1_per_timestamp.json")

    # Save text reports
    with open(EVAL_DIR / "report_f1_full.txt", "w") as fh:
        fh.write(full_report_text)
    with open(EVAL_DIR / "report_f1_triggered.txt", "w") as fh:
        fh.write(triggered_report)
    print(f"  Saved report_f1_full.txt  +  report_f1_triggered.txt")

    # Console: print t=1, t=T_MAX, triggered
    print(f"\n{'='*55}")
    print(f"  Classification report at t=1 (first packet only)")
    print(f"{'='*55}")
    p1 = val_probas[:, 0, :].argmax(axis=1)
    print(classification_report(y_val, p1, target_names=class_names, zero_division=0))

    print(f"\n{'='*55}")
    print(f"  Classification report at t={T_MAX} (all packets = baseline)")
    print(f"{'='*55}")
    pT = val_probas[:, -1, :].argmax(axis=1)
    print(classification_report(y_val, pT, target_names=class_names, zero_division=0))

    print(f"\n{'='*55}")
    print(f"  CALIMERA-triggered  (α={BEST_ALPHA}, avg t★={avg_t_star:.2f})")
    print(f"{'='*55}")
    print(triggered_report)


if __name__ == "__main__":
    main()
