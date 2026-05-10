#!/usr/bin/env python3
"""
EarlyFlow — Model comparison & ablation
========================================
Compares all trained deep models + MiniRocket baseline.

For each model that has cal_{model}.pt:
  1. Loads calibrated model
  2. Builds probas (or re-uses cached trig/val_probas_deep.npy if model matches)
  3. Runs CALIMERA at ALPHA_DEFAULT
  4. Records val_acc_t1, val_acc_tmax, earliness, HM, n_params, train_time_s

Outputs  →  calimera/eval/
  compare_table.csv
  compare_plot.png

Run:
    cd /path/to/earlyflow
    python calimera/compare_models.py
"""

import sys, json, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, "ml_edm/src")
from ml_edm.trigger._calimera import CALIMERA
from ml_edm.cost_matrices import CostMatrices

import calimera.config as cfg

EVAL_DIR  = Path(cfg.EVAL_DIR)
MODEL_DIR = Path(cfg.MODEL_DIR)
DATA_DIR  = Path(cfg.DATA_DIR)
CHUNK     = 512

ALPHA     = cfg.ALPHA_DEFAULT
TIMESTAMPS = np.arange(1, cfg.T_MAX + 1)


# ── Probas builder ────────────────────────────────────────────────────────────

def build_probas_inline(cal, X_np):
    import torch
    n = len(X_np)
    probas = np.zeros((n, cfg.T_MAX, cfg.N_CLASSES), dtype=np.float32)
    X_t = torch.from_numpy(X_np).float()
    for t in range(1, cfg.T_MAX + 1):
        chunks = []
        for i in range(0, n, CHUNK):
            chunks.append(cal.predict_proba(X_t[i:i + CHUNK], t))
        probas[:, t - 1, :] = np.concatenate(chunks, axis=0)
        print(f"    t={t:02d}/{cfg.T_MAX}", end="\r", flush=True)
    print()
    return probas


# ── CALIMERA simulation ───────────────────────────────────────────────────────

def run_calimera(trig_probas, val_probas, y_trig, y_val, alpha):
    n_classes = trig_probas.shape[2]
    timestamps = TIMESTAMPS

    cost_matrices = CostMatrices(
        timestamps=timestamps, n_classes=n_classes, alpha=alpha,
        delay_cost=lambda t, a=alpha: (1.0 - a) * float(t) / float(cfg.T_MAX),
    )
    X_trig_dummy = np.zeros((len(y_trig), cfg.T_MAX))
    trigger = CALIMERA(timestamps=timestamps)
    trigger.fit(X_trig_dummy, trig_probas, y_trig, cost_matrices)

    n = len(y_val)
    all_preds  = np.full(n, -1, dtype=int)
    all_t_star = np.full(n, cfg.T_MAX, dtype=int)
    decided    = np.zeros(n, dtype=bool)

    for t_idx, t in enumerate(timestamps):
        probas_t  = val_probas[:, t_idx, :]
        cur_preds = probas_t.argmax(axis=1)
        if t == cfg.T_MAX:
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
        all_t_star[still_open] = cfg.T_MAX

    acc  = float((all_preds == y_val).mean())
    earl = float(all_t_star.mean() / cfg.T_MAX)
    hm   = 2 * acc * (1 - earl) / (acc + (1 - earl) + 1e-9)
    return acc, earl, round(hm, 5)


# ── MiniRocket baseline ───────────────────────────────────────────────────────

def load_minirocket_baseline():
    rpt_path = MODEL_DIR / "training_report.json"
    sw_path  = EVAL_DIR  / "sweep_results.json"
    if not rpt_path.exists() or not sw_path.exists():
        return None
    with open(rpt_path) as f:
        rpt = json.load(f)
    with open(sw_path) as f:
        sw = json.load(f)
    # find alpha=0.5 in sweep
    row = next((r for r in sw["sweep"] if r["alpha"] == ALPHA), None)
    if row is None:
        return None
    return {
        "model":       "minirocket",
        "val_acc_t1":  rpt["per_timestamp_val_acc"].get("1", None),
        "val_acc_t20": rpt["baseline_acc"],
        "earliness":   row["earliness"],
        "hm":          row["hm"],
        "n_params":    "—",
        "train_time_s": "—",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load shared data
    X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(DATA_DIR / "y_train.npy").astype(np.int32)
    X_val   = np.load(DATA_DIR / "X_val.npy").astype(np.float32)
    y_val   = np.load(DATA_DIR / "y_val.npy").astype(np.int32)

    idx = np.arange(len(X_train))
    _, idx_trig = train_test_split(
        idx, test_size=cfg.TRIGGER_FRAC, stratify=y_train, random_state=cfg.SEED
    )
    X_trig = X_train[idx_trig]
    y_trig = y_train[idx_trig]

    rows = []

    # ── MiniRocket baseline ────────────────────────────────────────
    baseline = load_minirocket_baseline()
    if baseline:
        rows.append(baseline)
        print(f"  MiniRocket: acc_t1={baseline['val_acc_t1']}  "
              f"acc_t20={baseline['val_acc_t20']}  HM={baseline['hm']}")
    else:
        print("  MiniRocket baseline not found (run phase3 + phase45 first)")

    # ── Deep models ────────────────────────────────────────────────
    from calimera.models import get_model, CalibratedModel

    deep_rpt_path = MODEL_DIR / "training_report_deep.json"

    for model_name in ["lstm", "gru", "transformer", "tcn"]:
        cal_pt = MODEL_DIR / f"cal_{model_name}.pt"
        if not cal_pt.exists():
            print(f"  {model_name}: not trained yet, skipping.")
            continue

        print(f"\n  [{model_name.upper()}]")

        # load training report
        train_info = {}
        if deep_rpt_path.exists():
            with open(deep_rpt_path) as f:
                d = json.load(f)
            if d.get("model") == model_name:
                train_info = d

        # load calibrated model
        orig_model_cfg = cfg.MODEL
        cfg.MODEL = model_name
        model = get_model().to(cfg.DEVICE)
        cal   = CalibratedModel(model)
        cal.load(str(MODEL_DIR / f"cal_{model_name}"))
        cfg.MODEL = orig_model_cfg

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # check if we can reuse cached probas (model must match config)
        cache_trig = DATA_DIR / "trig_probas_deep.npy"
        cache_val  = DATA_DIR / "val_probas_deep.npy"
        cache_ytrig = DATA_DIR / "y_trig.npy"
        if (cache_trig.exists() and cache_val.exists() and
                train_info.get("model") == model_name):
            print("    Reusing cached probas …")
            trig_probas = np.load(cache_trig)
            val_probas  = np.load(cache_val)
            y_trig_cached = np.load(cache_ytrig) if cache_ytrig.exists() else y_trig
        else:
            print("    Building trig probas …")
            trig_probas = build_probas_inline(cal, X_trig)
            print("    Building val probas …")
            val_probas  = build_probas_inline(cal, X_val)
            y_trig_cached = y_trig

        # per-timestamp val acc from report or re-compute
        if train_info.get("per_timestamp_val_acc"):
            acc_t1  = train_info["per_timestamp_val_acc"].get("1",  None)
            acc_t20 = train_info["per_timestamp_val_acc"].get(str(cfg.T_MAX), None)
        else:
            acc_t1  = float((val_probas[:, 0, :].argmax(1) == y_val).mean())
            acc_t20 = float((val_probas[:, -1, :].argmax(1) == y_val).mean())

        print(f"    Running CALIMERA (α={ALPHA}) …")
        acc, earl, hm = run_calimera(trig_probas, val_probas, y_trig_cached, y_val, ALPHA)
        print(f"    acc={acc:.4f}  earl={earl:.4f}  HM={hm:.4f}")

        rows.append({
            "model":        model_name,
            "val_acc_t1":   round(float(acc_t1),  5) if acc_t1  is not None else None,
            "val_acc_t20":  round(float(acc_t20), 5) if acc_t20 is not None else None,
            "earliness":    round(earl, 5),
            "hm":           round(hm, 5),
            "n_params":     n_params,
            "train_time_s": train_info.get("train_time_s", "—"),
        })

    # ── Save CSV ───────────────────────────────────────────────────
    csv_path = EVAL_DIR / "compare_table.csv"
    fields   = ["model", "val_acc_t1", "val_acc_t20", "earliness", "hm",
                 "n_params", "train_time_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {csv_path}")

    # ── Console table ──────────────────────────────────────────────
    print(f"\n{'Model':>12}  {'acc_t1':>7}  {'acc_t20':>7}  {'earl':>6}  {'HM':>7}  {'params':>9}")
    print("-" * 60)
    for r in rows:
        print(f"  {r['model']:>10}  {str(r['val_acc_t1']):>7}  "
              f"{str(r['val_acc_t20']):>7}  {str(r['earliness']):>6}  "
              f"{str(r['hm']):>7}  {str(r['n_params']):>9}")

    if len(rows) < 2:
        print("\nNot enough models to plot — train more and re-run.")
        return

    # ── Plot ───────────────────────────────────────────────────────
    models    = [r["model"]       for r in rows]
    hms       = [float(r["hm"])        if r["hm"]       not in (None, "—") else 0 for r in rows]
    accs_t1   = [float(r["val_acc_t1"]) if r["val_acc_t1"] not in (None, "—") else 0 for r in rows]
    accs_t20  = [float(r["val_acc_t20"]) if r["val_acc_t20"] not in (None, "—") else 0 for r in rows]
    earls     = [float(r["earliness"]) if r["earliness"] not in (None, "—") else 0 for r in rows]

    x    = np.arange(len(models))
    w    = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*w, accs_t1,  w, label="Acc @ t=1",  color="steelblue",  alpha=0.85)
    ax.bar(x - 0.5*w, accs_t20, w, label="Acc @ t=20", color="darkorange", alpha=0.85)
    ax.bar(x + 0.5*w, hms,      w, label="HM",          color="seagreen",   alpha=0.85)
    ax.bar(x + 1.5*w, earls,    w, label="Earliness",   color="tomato",     alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Model comparison (CALIMERA α={ALPHA})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = EVAL_DIR / "compare_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved → {plot_path}")


if __name__ == "__main__":
    main()
