#!/usr/bin/env python3
"""
Phase 4/5 — Alpha sweep + Evaluation
=====================================
For each alpha value:
  1. Rebuild CostMatrices with that alpha
  2. Re-train CALIMERA trigger (KernelRidge backward loop) on saved trig_probas
  3. Simulate online classification on val set
  4. Record accuracy, earliness, HM, C_G

Outputs  →  calimera/eval/
  sweep_results.json   — per-alpha metrics
  pareto_plot.png      — accuracy vs earliness scatter with Pareto frontier

Run:
    cd /path/to/earlyflow
    python calimera/phase45_sweep_eval.py
"""

import sys, json, pickle, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path("ml_edm/src")))
from ml_edm.trigger._calimera import CALIMERA
from ml_edm.cost_matrices import CostMatrices

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("calimera/data")
MODEL_DIR = Path("calimera/models")
EVAL_DIR  = Path("calimera/eval")

ALPHA_VALUES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]

# ── Switch: set True to use deep model probas from build_probas.py ────────────
USE_DEEP = True    # False → MiniRocket (phase3); True → deep model (train_deep + build_probas)

# ── Load ──────────────────────────────────────────────────────────────────────
def load_everything():
    print("[0] Loading data and models …")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")
    X_trig  = np.load(DATA_DIR / "X_train.npy")   # full train — we'll re-derive trig below
    y_train = np.load(DATA_DIR / "y_train.npy")

    with open(DATA_DIR / "meta.json") as fh:
        meta = json.load(fh)

    with open(MODEL_DIR / "classifiers.pkl", "rb") as fh:
        classifiers = pickle.load(fh)
    with open(MODEL_DIR / "rocket.pkl", "rb") as fh:
        rocket = pickle.load(fh)

    T_MAX     = meta["T_MAX"]
    n_classes = len(meta["label_map"])
    timestamps = np.arange(1, T_MAX + 1)

    print(f"  Val {len(X_val):,} | T_MAX={T_MAX} | n_classes={n_classes}")
    return X_val, y_val, X_trig, y_train, meta, classifiers, rocket, T_MAX, n_classes, timestamps


def mask_after(X_sktime: np.ndarray, t: int) -> np.ndarray:
    Xm = X_sktime.copy()
    Xm[:, :, t:] = 0.0
    return Xm


def build_probas(rocket, X_sk, T_MAX, classifiers):
    """Return probas array (n, T_MAX, n_classes) using saved classifiers."""
    n_classes = classifiers[0].classes_.shape[0]
    probas = np.zeros((len(X_sk), T_MAX, n_classes), dtype=np.float32)
    for t in range(1, T_MAX + 1):
        feat = rocket.transform(mask_after(X_sk, t))
        probas[:, t - 1, :] = classifiers[t - 1].predict_proba(feat).astype(np.float32)
        del feat; gc.collect()
    return probas


def simulate_online(val_probas, trigger, cost_matrices, timestamps, y_val, T_MAX):
    """Online simulation — returns preds, t_star arrays.

    trigger.predict() infers the current timestamp from len(ts) for each
    series in X. We pass np.zeros((n, t)) so that len(ts)==t matches the
    correct entry in trigger.timestamps and the right halter is used.
    """
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
            # len(ts) == t  →  BaseTriggerModel maps to timestamp t
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


def harmonic_mean(acc, earliness):
    return 2 * acc * (1 - earliness) / (acc + (1 - earliness) + 1e-9)


# ── Pareto frontier ───────────────────────────────────────────────────────────
def pareto_front(points):
    """
    Returns mask of Pareto-optimal points in (accuracy, 1-earliness) space
    (maximise both axes).
    """
    pts = np.array(points)   # (n, 2): [accuracy, 1-earliness]
    dominated = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            if (pts[j] >= pts[i]).all() and (pts[j] > pts[i]).any():
                dominated[i] = True
                break
    return ~dominated


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    X_val, y_val, X_train, y_train, meta, classifiers, rocket, T_MAX, n_classes, timestamps = \
        load_everything()

    # ── Build trig/val probas once — they don't change across alphas ──────────
    from sklearn.model_selection import train_test_split
    TRIGGER_FRAC  = 0.30
    RANDOM_STATE  = 42

    idx = np.arange(len(X_train))
    idx_clf, idx_trig = train_test_split(
        idx, test_size=TRIGGER_FRAC, stratify=y_train, random_state=RANDOM_STATE
    )
    X_trig  = X_train[idx_trig]
    y_trig  = y_train[idx_trig]

    if USE_DEEP:
        print("\n[1] Loading deep-model probas from build_probas.py output …")
        trig_probas = np.load(DATA_DIR / "trig_probas_deep.npy")
        val_probas  = np.load(DATA_DIR / "val_probas_deep.npy")
        y_trig      = np.load(DATA_DIR / "y_trig.npy")
        print(f"  trig_probas {trig_probas.shape}  val_probas {val_probas.shape}")
    else:
        X_trig_sk = X_trig.transpose(0, 2, 1).astype(np.float32)
        X_val_sk  = X_val.transpose(0, 2, 1).astype(np.float32)
        print("\n[1] Building trigger-set probas …")
        trig_probas = build_probas(rocket, X_trig_sk, T_MAX, classifiers)
        print("[2] Building val-set probas …")
        val_probas  = build_probas(rocket, X_val_sk,  T_MAX, classifiers)

    # CALIMERA.fit uses X only for X.shape[1] (max_length); T_MAX suffices.
    X_trig_dummy = np.zeros((len(y_trig), T_MAX))

    # ── Baseline: classify at T_MAX ───────────────────────────────────────────
    baseline_acc = (val_probas[:, -1, :].argmax(axis=1) == y_val).mean()
    print(f"\n  Baseline (t=T_MAX): acc={baseline_acc:.4f}  earliness=1.0  HM={baseline_acc:.4f}")

    # ── Alpha sweep ───────────────────────────────────────────────────────────
    print(f"\n[3] Sweeping {len(ALPHA_VALUES)} alpha values …")
    results = []

    for alpha in ALPHA_VALUES:
        print(f"\n  α={alpha} …", end=" ", flush=True)

        cost_matrices = CostMatrices(
            timestamps=timestamps,
            n_classes=n_classes,
            alpha=alpha,
            delay_cost=lambda t, a=alpha: (1.0 - a) * float(t) / float(T_MAX),
        )

        trigger = CALIMERA(timestamps=timestamps)
        trigger.fit(X_trig_dummy, trig_probas, y_trig, cost_matrices)

        preds, t_star = simulate_online(val_probas, trigger, cost_matrices, timestamps, y_val, T_MAX)

        acc      = float((preds == y_val).mean())
        earl     = float(t_star.mean() / T_MAX)
        hm       = harmonic_mean(acc, earl)

        # C_G = alpha * misclf_rate + (1-alpha) * earliness
        misclf   = 1.0 - acc
        c_g      = alpha * misclf + (1.0 - alpha) * earl

        print(f"acc={acc:.4f}  earliness={earl:.4f}  HM={hm:.4f}  C_G={c_g:.4f}")

        results.append({
            "alpha":      alpha,
            "accuracy":   round(acc, 5),
            "earliness":  round(earl, 5),
            "hm":         round(hm, 5),
            "c_g":        round(c_g, 5),
        })

    # ── Save JSON ─────────────────────────────────────────────────────────────
    sweep_out = {
        "baseline": {
            "accuracy":  round(float(baseline_acc), 5),
            "earliness": 1.0,
            "hm":        round(float(baseline_acc), 5),
        },
        "sweep": results,
    }
    with open(EVAL_DIR / "sweep_results.json", "w") as fh:
        json.dump(sweep_out, fh, indent=2)
    print(f"\n  Saved sweep_results.json")

    # ── Pareto plot ───────────────────────────────────────────────────────────
    print("[4] Plotting …")

    accs  = [r["accuracy"]  for r in results]
    earls = [r["earliness"] for r in results]
    hms   = [r["hm"]        for r in results]
    alphas = [r["alpha"]    for r in results]

    # Pareto in (accuracy, 1-earliness) space
    pts_2d = [(a, 1 - e) for a, e in zip(accs, earls)]
    is_pareto = pareto_front(pts_2d)

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(earls, accs, c=alphas, cmap="plasma_r",
                    s=80, zorder=5, label="CALIMERA (α sweep)")
    plt.colorbar(sc, ax=ax, label="α (delay-cost weight)")

    # Connect Pareto-optimal points
    pareto_pts = sorted([(earls[i], accs[i]) for i in range(len(results)) if is_pareto[i]])
    if pareto_pts:
        px, py = zip(*pareto_pts)
        ax.plot(px, py, "b--", lw=1.5, alpha=0.7, label="Pareto frontier")

    # Annotate alpha values
    for i, (e, a, al) in enumerate(zip(earls, accs, alphas)):
        ax.annotate(f"α={al}", (e, a), textcoords="offset points",
                    xytext=(6, 2), fontsize=7, color="dimgray")

    # Baseline star
    ax.scatter([1.0], [baseline_acc], marker="*", s=200, color="red",
               zorder=6, label=f"Baseline (t=T_MAX, acc={baseline_acc:.3f})")

    ax.set_xlabel("Earliness  (mean t★/T_MAX, lower = earlier)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("CALIMERA: Accuracy vs Earliness trade-off\n(α sweep, val set)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(max(0, min(accs) - 0.05), min(1.0, max(accs) + 0.05))

    plt.tight_layout()
    plot_path = EVAL_DIR / "pareto_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved {plot_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  {'α':>6}  {'acc':>7}  {'earliness':>9}  {'HM':>7}  {'C_G':>7}")
    print(f"  {'------':>6}  {'-------':>7}  {'---------':>9}  {'-------':>7}  {'-------':>7}")
    for r in results:
        print(f"  {r['alpha']:>6.3f}  {r['accuracy']:>7.4f}  {r['earliness']:>9.4f}"
              f"  {r['hm']:>7.4f}  {r['c_g']:>7.4f}")
    print(f"\n  Baseline: acc={baseline_acc:.4f}  earliness=1.0000  HM={baseline_acc:.4f}")
    best = max(results, key=lambda r: r["hm"])
    print(f"  Best HM : α={best['alpha']}  HM={best['hm']}  acc={best['accuracy']}  earl={best['earliness']}")


if __name__ == "__main__":
    main()
