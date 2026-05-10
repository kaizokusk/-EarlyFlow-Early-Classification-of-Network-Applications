#!/usr/bin/env python3
"""
EarlyFlow — Deep model prefix training
=======================================
Trains LSTM / GRU / Transformer / TCN with PREFIX TRAINING.

At each batch a random t ~ Uniform(1, T_MAX) is sampled and only the
first t packets are fed to the model. This trains the model to classify
correctly at every prefix length, replacing the 20 separate MiniRocket
classifiers from Phase 3.

Outputs  →  calimera/models/
  deep_{MODEL}_best.pt          — best checkpoint by val_acc at t=T_MAX
  cal_{MODEL}.pt                — calibrated weights
  cal_{MODEL}.platt.pkl         — Platt scaler
  training_report_deep.json     — per-epoch + per-timestamp metrics

Usage:
    cd /path/to/earlyflow
    python calimera/train_deep.py                        # uses MODEL from config.py
    python calimera/train_deep.py --model transformer    # override model
    python calimera/train_deep.py --model gru --epochs 20
    python calimera/train_deep.py --model tcn --epochs 40 --lr 5e-4
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

import calimera.config as cfg
from calimera.dataset import get_loaders
from calimera.models import get_model, CalibratedModel


def parse_args():
    p = argparse.ArgumentParser(description="EarlyFlow deep model training")
    p.add_argument("--model",  default=None,
                   choices=["lstm", "gru", "transformer", "tcn"],
                   help="Model architecture (default: from config.py)")
    p.add_argument("--epochs", type=int,   default=150, help="Override EPOCHS")
    p.add_argument("--lr",     type=float, default=None, help="Override learning rate")
    p.add_argument("--hidden", type=int,   default=None, help="Override HIDDEN size")
    return p.parse_args()


torch.manual_seed(cfg.SEED)
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        t      = random.randint(1, cfg.T_MAX)       # prefix training
        logits = model.forward_prefix(x, t)
        loss   = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss    += loss.item() * len(y)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n       += len(y)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_at_t(model, loader, t: int) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        preds    = model.forward_prefix(x.to(cfg.DEVICE), t).argmax(1)
        correct += (preds == y.to(cfg.DEVICE)).sum().item()
        total   += len(y)
    return correct / total


@torch.no_grad()
def eval_per_timestamp(model, loader) -> list:
    return [eval_at_t(model, loader, t) for t in range(1, cfg.T_MAX + 1)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # CLI overrides take priority over config.py values
    if args.model:  cfg.MODEL  = args.model
    if args.epochs: cfg.EPOCHS = args.epochs
    if args.lr:     cfg.LR     = args.lr
    if args.hidden: cfg.HIDDEN = args.hidden

    Path(cfg.MODEL_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[config] MODEL={cfg.MODEL}  DEVICE={cfg.DEVICE}  EPOCHS={cfg.EPOCHS}  LR={cfg.LR}")
    loaders  = get_loaders()
    model    = get_model().to(cfg.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model]  {cfg.MODEL}  params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_val_acc = 0.0
    best_path    = Path(cfg.MODEL_DIR) / f"deep_{cfg.MODEL}_best.pt"
    history      = []

    print(f"\nTraining for {cfg.EPOCHS} epochs with prefix sampling …\n")
    t0 = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion)
        scheduler.step()

        # Full per-timestamp eval every 5 epochs; quick t=T_MAX eval otherwise
        if epoch % 5 == 0 or epoch == cfg.EPOCHS:
            val_accs = eval_per_timestamp(model, loaders["val"])
            val_acc  = val_accs[-1]
        else:
            val_accs = []
            val_acc  = eval_at_t(model, loaders["val"], cfg.T_MAX)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state":   model.state_dict(),
                "model_name":    cfg.MODEL,
                "val_acc":       val_acc,
                "val_acc_per_t": val_accs,
                "epoch":         epoch,
                "config": {
                    "N_FEAT":    cfg.N_FEAT,
                    "N_CLASSES": cfg.N_CLASSES,
                    "HIDDEN":    cfg.HIDDEN,
                    "N_LAYERS":  cfg.N_LAYERS,
                    "T_MAX":     cfg.T_MAX,
                },
            }, best_path)

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc, 4),
            "val_acc":    round(val_acc, 4),
        })
        print(f"  epoch {epoch:03d}/{cfg.EPOCHS}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  |  best val_acc={best_val_acc:.4f}")

    # ── Platt calibration on best checkpoint ──────────────────────
    print("\nFitting Platt calibration …")
    ckpt = torch.load(best_path, map_location=cfg.DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    cal = CalibratedModel(model)
    cal.calibrate(loaders["val"])
    cal_path = str(Path(cfg.MODEL_DIR) / f"cal_{cfg.MODEL}")
    cal.save(cal_path)

    # ── Final per-timestamp eval ───────────────────────────────────
    print("\nFinal per-timestamp val accuracy:")
    val_accs_final = eval_per_timestamp(model, loaders["val"])
    for t, acc in enumerate(val_accs_final, 1):
        bar = "█" * int(acc * 40)
        print(f"  t={t:02d}  {acc:.4f}  {bar}")

    # ── Save report ───────────────────────────────────────────────
    report = {
        "model":           cfg.MODEL,
        "best_val_acc":    round(best_val_acc, 5),
        "val_acc_t1":      round(val_accs_final[0], 5),
        "train_time_s":    round(elapsed, 1),
        "n_params":        n_params,
        "per_timestamp_val_acc": {
            str(t): round(a, 5) for t, a in enumerate(val_accs_final, 1)
        },
        "history": history,
    }
    rpt_path = Path(cfg.MODEL_DIR) / "training_report_deep.json"
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport → {rpt_path}")
    print(f"Next: python calimera/build_probas.py")


if __name__ == "__main__":
    main()
