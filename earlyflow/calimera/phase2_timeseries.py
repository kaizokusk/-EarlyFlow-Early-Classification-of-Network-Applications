#!/usr/bin/env python3
"""
Phase 2 — Time-series construction for CALIMERA
================================================
Reads the 4 filtered CSVs (cloud, social_media, streaming, web) from
early_ml/split_file/, builds a padded 3-D array (n_flows, T_MAX, N_FEAT)
and produces stratified train / val / test splits.

Outputs  →  calimera/data/
  X_train.npy, X_val.npy, X_test.npy     float32  (n, T_MAX, N_FEAT)
  y_train.npy, y_val.npy, y_test.npy     int32    (n,)
  flow_ids_{split}.npy                   object   (n,)   — for traceability
  meta.json                              — T_MAX, feature names, label map, scaler

Run:
  cd /path/to/earlyflow
  python calimera/phase2_timeseries.py
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("")
OUT_DIR   = Path("calimera/data")

T_MAX     = 20      # first N packets per flow used as time-series length
MIN_PKTS  = 3       # flows with fewer packets are discarded
VAL_FRAC  = 0.15
TEST_FRAC = 0.15
SEED      = 42

# Numeric features taken directly from each packet row.
# tcp_flags NaN → 0  (means UDP / ICMP, no TCP header present)
NUMERIC_FEATS = [
    "ip_total_len",   # IP payload + header size
    "frame_len",      # L2 frame size
    "iat",            # inter-arrival time (seconds)
    "direction",      # 0 = client→server, 1 = server→client
    "tcp_flags",      # TCP flags byte; 0 for non-TCP flows
    "proto",          # IP protocol number (6=TCP, 17=UDP, …)
]

ALL_FEATS = NUMERIC_FEATS + ["payload_entropy"]
N_FEAT    = len(ALL_FEATS)


# ── Payload entropy ───────────────────────────────────────────────────────────
def _entropy(hex_str) -> float:
    """Shannon entropy (bits) of the bytes encoded in a hex string."""
    if not hex_str or (isinstance(hex_str, float) and math.isnan(hex_str)):
        return 0.0
    s = str(hex_str).strip()
    if len(s) < 2:
        return 0.0
    try:
        data = bytes.fromhex(s)
    except ValueError:
        return 0.0
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    freq = counts[counts > 0] / len(data)
    return float(-np.sum(freq * np.log2(freq)))


# ── Load ──────────────────────────────────────────────────────────────────────
def load_all(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*_filtered.csv"))
    if not files:
        raise FileNotFoundError(f"No *_filtered.csv files found in {data_dir}")

    needed = ["flow_id", "pkt_rank", "payload_hex", "category_label"] + NUMERIC_FEATS
    chunks = []
    for f in files:
        print(f"  Loading {f.name} …")
        df = pd.read_csv(f, usecols=needed)
        chunks.append(df)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df):,} rows | {df['flow_id'].nunique():,} unique flows")

    # tcp_flags: NaN means no TCP header (UDP / ICMP) → fill 0
    df["tcp_flags"] = df["tcp_flags"].fillna(0.0)

    print("  Computing payload entropy …")
    df["payload_entropy"] = df["payload_hex"].apply(_entropy)

    return df


# ── Build per-flow sequences ──────────────────────────────────────────────────
def build_sequences(df: pd.DataFrame):
    """
    Group packets by flow_id, sort by pkt_rank, keep first T_MAX packets.

    Returns
    -------
    flow_ids  : list[str]
    sequences : list[np.ndarray]  shape (actual_len ≤ T_MAX, N_FEAT), float32
    labels    : list[str]
    """
    flow_ids, sequences, labels = [], [], []

    for fid, grp in df.sort_values("pkt_rank").groupby("flow_id", sort=False):
        if len(grp) < MIN_PKTS:
            continue
        grp = grp.head(T_MAX)
        seq = grp[ALL_FEATS].values.astype(np.float32)
        flow_ids.append(fid)
        sequences.append(seq)
        labels.append(grp["category_label"].iloc[0])

    return flow_ids, sequences, labels


# ── Scaler (fitted on training rows only) ────────────────────────────────────
def fit_scaler(train_sequences):
    stacked = np.concatenate(train_sequences, axis=0)   # (total_train_pkts, N_FEAT)
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0) + 1e-8
    return mean, std


def scale_and_pad(sequences, mean, std) -> np.ndarray:
    """
    Normalise every sequence then zero-pad to shape (T_MAX, N_FEAT).
    Padding with zeros is intentional: a zero row signals 'flow ended here'
    to the CALIMERA classifier at each timestamp.
    """
    X = np.zeros((len(sequences), T_MAX, N_FEAT), dtype=np.float32)
    for i, seq in enumerate(sequences):
        normed = (seq - mean) / std
        t = min(len(normed), T_MAX)
        X[i, :t, :] = normed[:t]
    return X


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading CSVs …")
    df = load_all(DATA_DIR)

    print("\n[2/5] Building per-flow sequences …")
    flow_ids, sequences, labels = build_sequences(df)
    print(f"  Kept {len(flow_ids):,} flows  (MIN_PKTS={MIN_PKTS}, T_MAX={T_MAX})")

    le = LabelEncoder()
    y_all = le.fit_transform(labels).astype(np.int32)
    label_map = {int(i): c for i, c in enumerate(le.classes_)}
    print("  Class distribution:")
    for cls_idx, cls_name in label_map.items():
        n = int((y_all == cls_idx).sum())
        print(f"    {cls_name:15s}: {n:,} flows")

    print("\n[3/5] Stratified train / val / test split …")
    idx = np.arange(len(flow_ids))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_FRAC, stratify=y_all, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_FRAC / (1.0 - TEST_FRAC),
        stratify=y_all[idx_trainval],
        random_state=SEED)
    print(f"  Train {len(idx_train):,} | Val {len(idx_val):,} | Test {len(idx_test):,}")

    print("\n[4/5] Fitting z-score scaler on training packets …")
    train_seqs = [sequences[i] for i in idx_train]
    mean, std = fit_scaler(train_seqs)
    print(f"  Feature means : {np.round(mean, 4)}")
    print(f"  Feature stds  : {np.round(std, 4)}")

    print("\n[5/5] Scaling, padding, and saving …")
    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    for split, idx in splits.items():
        seqs  = [sequences[i] for i in idx]
        X     = scale_and_pad(seqs, mean, std)
        y     = y_all[idx]
        fids  = np.array([flow_ids[i] for i in idx])

        np.save(OUT_DIR / f"X_{split}.npy", X)
        np.save(OUT_DIR / f"y_{split}.npy", y)
        np.save(OUT_DIR / f"flow_ids_{split}.npy", fids)
        print(f"  {split:5s}: X={X.shape}  y={y.shape}")

    meta = {
        "T_MAX":      T_MAX,
        "N_FEAT":     N_FEAT,
        "MIN_PKTS":   MIN_PKTS,
        "features":   ALL_FEATS,
        "label_map":  label_map,
        "scaler": {
            "mean": mean.tolist(),
            "std":  std.tolist(),
        },
        "split": {
            "val_frac":  VAL_FRAC,
            "test_frac": TEST_FRAC,
            "seed":      SEED,
        },
    }
    with open(OUT_DIR / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n✓  Saved to {OUT_DIR}/")
    print("   X_{{train,val,test}}.npy       — (n_flows, T_MAX=20, N_FEAT=7)  float32")
    print("   y_{{train,val,test}}.npy       — (n_flows,)  int32")
    print("   flow_ids_{{split}}.npy         — flow_id strings for traceability")
    print("   meta.json                      — config, label map, scaler params")


if __name__ == "__main__":
    main()
