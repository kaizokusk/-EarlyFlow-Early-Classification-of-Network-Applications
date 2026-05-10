# EarlyFlow — Early Network Traffic Classification with CALIMERA

Early classification of mobile network flows using per-packet features. The system identifies whether a flow belongs to **cloud**, **social media**, **streaming**, or **web** traffic — making the decision as early as possible (after seeing only a few packets) without waiting for the entire flow to complete.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Method: CALIMERA](#3-method-calimera)
4. [Project Structure](#4-project-structure)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Phase 2 — Time Series Construction](#6-phase-2--time-series-construction)
7. [Phase 3 — MiniRocket + CALIMERA Training](#7-phase-3--minirocket--calimera-training)
8. [Phase 4/5 — Alpha Sweep & Evaluation](#8-phase-45--alpha-sweep--evaluation)
9. [Phase 6 — Report Generation](#9-phase-6--report-generation)
10. [Phase 6b — F1 / Precision / Recall Analysis](#10-phase-6b--f1--precision--recall-analysis)
11. [Deep Model Extension (EarlyFlow)](#11-deep-model-extension-earlyflow)

13. [How to Run](#13-how-to-run)
14. [Dependencies](#14-dependencies)
15. [Key Design Decisions](#15-key-design-decisions)

---

## 1. Problem Statement

Traditional network traffic classifiers wait until a flow is complete before making a decision. In real-time applications (QoS routing, firewall policy, anomaly detection), this delay is unacceptable.

**Early Time Series Classification (ETSC)** asks: *at what point during a flow can we confidently classify it?*

The trade-off:
- **Too early** → low accuracy (few packets seen)
- **Too late** → high latency (waited unnecessarily)

This project implements CALIMERA, a state-of-the-art ETSC method, and extends it with deep learning architectures (LSTM, GRU, Transformer, TCN) — all trained on real mobile YouTube network captures.

---

## 2. Dataset

**Source:** Raw PCAP captures from a campus network, labeled via DNS-based domain mapping and processed into per-packet CSVs by `pcap_pipeline.py`.

**Input files:** `*_filtered.csv` — one per traffic category, output of `pcap_pipeline.py`. Set `DATA_DIR` in `calimera/phase2_timeseries.py` to the directory containing these files before running the pipeline.

### Traffic Categories (Classes)

| Label | Class | Description |
|-------|-------|-------------|
| 0 | `cloud` | Cloud storage, productivity, dev tools, and update traffic |
| 1 | `conferencing` | Real-time bidirectional audio/video calls |
| 2 | `gaming` | Game servers and gaming platform traffic |
| 3 | `social_media` | Social media and messaging platforms |
| 4 | `streaming` | Video/audio streaming CDNs (YouTube, Netflix, Spotify, etc.) |
| 5 | `web` | Search, AI, news, e-commerce, and general browsing |

### Per-Packet Features

Each row in the raw CSVs represents one packet. The following 7 features are extracted:

| Feature | Description | Notes |
|---------|-------------|-------|
| `ip_total_len` | IP header + payload size (bytes) | Distinguishes large media vs small control packets |
| `frame_len` | Layer-2 frame size (bytes) | Similar to ip_total_len, includes L2 overhead |
| `iat` | Inter-arrival time (seconds) | Captures flow pacing / burstiness |
| `direction` | 0 = client→server, 1 = server→client | Streaming is server-heavy; web is request-driven |
| `tcp_flags` | TCP flags byte; 0 for non-TCP | SYN/ACK patterns differ by application |
| `proto` | IP protocol number (6=TCP, 17=UDP) | Streaming often uses QUIC/UDP |
| `payload_entropy` | Shannon entropy (bits) of payload bytes | Encrypted traffic (TLS) has high entropy (~7–8 bits); plaintext is lower |

> **Why payload entropy?** It is a single float derived from the payload — no byte reconstruction is possible. It distinguishes encrypted protocols (HTTPS/TLS, QUIC) from unencrypted ones without any privacy risk, and is one of the strongest single discriminating features for traffic type.

### Dataset Statistics

After filtering (minimum 3 packets per flow), splitting at flow level:

| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | 70% | Classifier + trigger model training |
| Val | 15% | Hyperparameter selection, early stopping |
| Test | 15% | Final held-out evaluation |

Each flow is represented as a padded 3D array of shape `(T_MAX=20, N_FEAT=7)`. Flows shorter than 20 packets are zero-padded (zero = "flow ended here").

**Z-score normalisation** is applied feature-wise, scaler fitted on training packets only.

---

## 3. Method: CALIMERA

**CALIMERA** (CALIbrated Models for EaRly clAssification) is a stopping-rule algorithm that decides, at each incoming packet, whether to classify now or wait for more data.

### Core Idea

At timestamp `t` (after seeing `t` packets), a classifier produces calibrated class probabilities `P(class | X₁..ₜ)`. CALIMERA asks:

> *Does waiting for one more packet reduce the expected cost, or not?*

If waiting costs more → classify now. Otherwise → wait.

### Cost Function

```
C_G = α · misclassification_cost + (1 - α) · delay_cost
```

- **misclassification_cost**: Fraction of wrong predictions
- **delay_cost**: `t / T_MAX` — how late the decision was made (0 = instant, 1 = full flow)
- **α (alpha)**: Delay-cost weight — controls the accuracy/earliness trade-off
  - Low α → prioritise early decisions (accept lower accuracy)
  - High α → prioritise accuracy (accept later decisions)

### CALIMERA Backward Loop

1. **Forward pass**: Train T_MAX classifiers, one per timestamp. Collect probabilities on a held-out trigger set.
2. **Backward loop**: Starting from `t = T_MAX - 1` down to `t = 1`:
   - Compute features from probabilities at time `t`: `[probas, max_proba, max - 2nd_max]`
   - Compute target: `Δcost = cost(t+1) - cost(t)` — the benefit of waiting one more step
   - Fit a **KernelRidge** (RBF kernel) regressor to predict `Δcost`
   - If predicted `Δcost > 0` (waiting is more expensive) → this halter fires "classify now"
   - Propagate updated costs backward

3. **Online inference**: For each new flow at timestamp `t`, run its probabilities through the halter at `t`. If it fires → classify immediately. Otherwise → wait for `t+1`.

### 70/30 Classifier/Trigger Split

To avoid data leakage (the trigger model must see unbiased probabilities):
- **70% of training data** → fit classifiers (MiniRocket + Ridge / deep model)
- **30% of training data** → generate probabilities from classifiers, train CALIMERA halters on those

This mirrors the `trigger_proportion` design in the `ml_edm` library.

---

## 4. Project Structure

```
earlyflow/
├── domain_map.py                     ← DNS domain → 6-class label map (built from campus DNS data)
├── pcap_pipeline.py                  ← PCAP → labeled parquet/CSV pipeline (Pass 1: DNS; Pass 2: features)
│
├── calimera/                         ← Main project package
│   ├── __init__.py
│   ├── config.py                     ← Central hyperparameter config
│   ├── dataset.py                    ← PyTorch DataLoader for deep models
│   ├── models.py                     ← LSTM / GRU / Transformer / TCN + CalibratedModel
│   ├── train_deep.py                 ← Deep model prefix training loop
│   ├── build_probas.py               ← CALIMERA bridge: deep model → .npy probas
│   ├── compare_models.py             ← Ablation: all models vs MiniRocket baseline
│   ├── eval_triggered_all_models.py  ← CALIMERA triggered F1 evaluation across ALL models
│   │
│   ├── phase2_timeseries.py          ← Build X_train/val/test.npy from CSVs (set DATA_DIR first)
│   ├── phase3_calimera_train.py      ← MiniRocket + Ridge + CALIMERA training
│   ├── phase45_sweep_eval.py         ← Alpha sweep, Pareto plot
│   ├── phase6_report.py              ← Summary report + plots
│   ├── phase6b_f1_eval.py            ← Per-class F1/Precision/Recall vs packets
│   │
│   
│
└── ml_edm/                           ← External CALIMERA library (submodule)
    └── src/ml_edm/
        ├── trigger/_calimera.py      CALIMERA backward loop implementation
        ├── trigger/_base.py          BaseTriggerModel interface
        └── cost_matrices.py          CostMatrices class
```

> **Data files (not committed — large):** Raw PCAP files are processed offline by `pcap_pipeline.py` to produce `*_filtered.csv` files. Set `DATA_DIR` in `calimera/phase2_timeseries.py` to the directory containing those CSVs before running Phase 2.

---

## 5. Pipeline Overview

```
Raw PCAPs (user-configured PCAP_DIR)
         │
         ▼
pcap_pipeline.py  (uses domain_map.py)
  ├─ Pass 1: extract DNS CNAME answers → flow label cache (6 classes)
  ├─ Pass 2: extract per-packet features → labeled parquet / *_filtered.csv
  └─ Output: one *_filtered.csv per traffic category
         │
         ▼
phase2_timeseries.py  (set DATA_DIR = path to *_filtered.csv files)
  ├─ Load & merge 4 category CSVs
  ├─ Compute payload entropy
  ├─ Group by flow_id, sort by pkt_rank
  ├─ Keep first T_MAX=20 packets per flow
  ├─ Z-score normalise (train statistics only)
  ├─ Zero-pad shorter flows
  └─ 70/15/15 stratified split → X_{train,val,test}.npy, meta.json
         │
         ▼
── MiniRocket Path ──────────────────────────────────────────────────────────
phase3_calimera_train.py
  ├─ Fit MiniRocket (1000 kernels) on training data → 6,468 features
  ├─ For t = 1..20: zero-mask X after position t, extract features
  ├─ Fit RidgeClassifierCV on clf-set (70%), calibrate on trig-set (30%)
  ├─ Collect trig_probas (n_trig, 20, 4) and val_probas (n_val, 20, 4)
  ├─ CALIMERA backward loop → 19 KernelRidge halters
  └─ Simulate online val set → accuracy, earliness, HM
         │
── Deep Model Path ──────────────────────────────────────────────────────────
train_deep.py --model [lstm|gru|transformer|tcn]
  ├─ Prefix training: random t per batch → model.forward_prefix(x, t)
  ├─ CosineAnnealingLR + AdamW, gradient clipping
  ├─ Platt calibration on val set
  └─ Save cal_{model}.pt + .platt.pkl
         │
build_probas.py --model [model]
  └─ Run forward_prefix(x, t) for all t → trig_probas_deep.npy, val_probas_deep.npy
         │
         ▼ (both paths produce same interface)
phase45_sweep_eval.py  [set USE_DEEP = True/False]
  ├─ For α in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]:
  │   ├─ Build CostMatrices(α)
  │   ├─ Re-train CALIMERA trigger on trig_probas
  │   ├─ Simulate online classification on val set
  │   └─ Record accuracy, earliness, HM, C_G
  └─ Save sweep_results.json + pareto_plot.png
         │
         ▼
phase6_report.py              → report.md, per_timestamp_acc.png (updated pareto_plot.png)
phase6b_f1_eval.py            → F1/Precision/Recall plots + classification reports
compare_models.py             → compare_table.csv + compare_plot.png
eval_triggered_all_models.py  → triggered_f1_all_models.txt + triggered_summary_table.txt
```

---

## 6. Phase 2 — Time Series Construction

**Script:** `calimera/phase2_timeseries.py`

### What it does

1. **Loads** all `*_filtered.csv` files from the directory set in `DATA_DIR`
2. **Fills** `tcp_flags` NaN → 0 (UDP/ICMP flows have no TCP header)
3. **Computes** payload entropy for each packet (Shannon entropy of hex-decoded bytes)
4. **Groups** packets by `flow_id`, sorts by `pkt_rank` (packet arrival order)
5. **Truncates** each flow to first `T_MAX = 20` packets; discards flows with < 3 packets
6. **Encodes** labels with `LabelEncoder` → integer class indices
7. **Splits** at flow level (not packet level) — stratified 70/15/15
8. **Fits** z-score scaler on training packets only, applies to all splits
9. **Zero-pads** flows shorter than T_MAX (zero row = "no more packets at this step")
10. **Saves** `X_{train,val,test}.npy`, `y_{train,val,test}.npy`, `flow_ids_{split}.npy`, `meta.json`

### Configuration

```python
T_MAX    = 20     # packets per flow used as time-series length
MIN_PKTS = 3      # flows with fewer packets discarded
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED     = 42
```

### Output shapes

```
X_train.npy  →  (n_train, 20, 7)   float32   zero-padded, z-normalised
y_train.npy  →  (n_train,)          int32     class indices 0–3
```

### Run

```bash
cd /path/to/earlyflow
# Edit DATA_DIR in calimera/phase2_timeseries.py to point to *_filtered.csv files first
python calimera/phase2_timeseries.py
```

---

## 7. Phase 3 — MiniRocket + CALIMERA Training

**Script:** `calimera/phase3_calimera_train.py`

### What it does

**Step 1 — Fit MiniRocket**

MiniRocket (Minimal Random Convolutional Kernel Transform) transforms multivariate time series into a fixed-size feature vector using random convolutional kernels. With 1,000 kernels on 7-channel data it produces **6,468 features**.

- Fitted once on the full clf-set (70% of training data)
- At each timestamp `t`, the full T_MAX series is **zero-masked** after position `t` to simulate "only t packets seen", then transformed

```python
def mask_after(X_sktime, t):
    Xm = X_sktime.copy()
    Xm[:, :, t:] = 0.0    # zero out everything after packet t
    return Xm
```

**Step 2 — Train 20 calibrated classifiers**

For each timestamp `t = 1..20`:
- Extract features from masked series
- Fit `RidgeClassifierCV` on clf-set (cross-validates regularisation strength α)
- Apply **Platt scaling** (sigmoid calibration via `CalibratedClassifierCV`) on trig-set to produce well-calibrated probabilities
- Collect trigger-set probabilities → `trig_probas` array of shape `(n_trig, 20, 4)`

> **Why calibration?** Raw classifier scores are not probabilities. CALIMERA's cost function needs `P(class | X)`. Platt scaling maps scores to the [0,1] probability simplex.

**Step 3 — CALIMERA backward loop**

Trains 19 KernelRidge halters in reverse order using `ml_edm.trigger._calimera.CALIMERA`.

**Step 4 — Online validation simulation**

Simulates the online decision process on the val set — at each timestamp `t`, runs the halter for flows not yet decided. Records `t★` (decision time) and final prediction per flow.

### Memory management

10,000 kernels would produce ~70,000 features — too large for RAM. 1,000 kernels keeps the feature matrix at ~0.8 GB. Feature matrices are deleted after each timestamp with `gc.collect()`.

### Outputs

```
calimera/models/rocket.pkl           Fitted MiniRocket transformer
calimera/models/classifiers.pkl      List of 20 calibrated Ridge classifiers
calimera/models/trigger.pkl          CALIMERA trigger model (α=0.5)
calimera/models/cost_matrices.pkl    CostMatrices object
calimera/models/training_report.json Per-timestamp val accuracy
```

### Run

```bash
python calimera/phase3_calimera_train.py
```

---

## 8. Phase 4/5 — Alpha Sweep & Evaluation

**Script:** `calimera/phase45_sweep_eval.py`

### What it does

Sweeps `α` across 10 values: `[0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]`

For each α:
1. Rebuilds `CostMatrices(α)` — lower α = delay is cheap, higher α = delay is expensive
2. Re-trains CALIMERA trigger from scratch on `trig_probas` (fast, ~seconds per α)
3. Simulates online classification on val set
4. Records: **accuracy**, **earliness** (`mean(t★) / T_MAX`), **HM** (harmonic mean), **C_G** (total cost)

### Critical implementation detail

`BaseTriggerModel.predict()` infers the current timestamp from `len(ts)` for each series in `X`. Passing `np.zeros((n, t))` at each timestamp `t` ensures the correct halter (for timestamp `t`) is used:

```python
X_at_t = np.zeros((n, t), dtype=np.float32)   # len(row) == t → correct halter fires
triggers = trigger.predict(X_at_t, probas_t, cost_matrices)
```

### Switching between MiniRocket and deep model

At the top of the script:

```python
USE_DEEP = False   # False → MiniRocket; True → deep model (run build_probas.py first)
```

### Outputs

```
calimera/eval/sweep_results.json    Per-alpha metrics
calimera/eval/pareto_plot.png       Two-panel figure:
                                      Left:  accuracy vs earliness (Pareto frontier)
                                      Right: HM and C_G vs α (log scale)
```

### Run

```bash
python calimera/phase45_sweep_eval.py
```

---

## 9. Phase 6 — Report Generation

**Script:** `calimera/phase6_report.py`

Reads `sweep_results.json` and `training_report.json` — no models loaded.

### Outputs

| File | Description |
|------|-------------|
| `calimera/eval/report.md` | Full markdown report with method, results table, key findings |
| `calimera/eval/pareto_plot.png` | Enhanced two-panel Pareto + HM/C_G chart |
| `calimera/eval/per_timestamp_acc.png` | Accuracy vs packets seen (1→20), showing when accuracy plateaus |

### Run

```bash
python calimera/phase6_report.py
```

---

## 10. Phase 6b — F1 / Precision / Recall Analysis

**Script:** `calimera/phase6b_f1_eval.py`

### What it does

For every timestamp `t = 1..20`:
- Runs the saved classifier on val set
- Computes per-class and macro-averaged Precision / Recall / F1
- Also runs CALIMERA at best α and produces a `classification_report` on triggered predictions

### Outputs

| File | Description |
|------|-------------|
| `plot_macro_prf1.png` | Macro P / R / F1 vs packets (1→20) with CALIMERA avg trigger line |
| `plot_perclass_f1.png` | F1 per class (cloud/social_media/streaming/web) vs packets |
| `plot_perclass_precision.png` | Precision per class vs packets |
| `plot_perclass_recall.png` | Recall per class vs packets |
| `f1_per_timestamp.json` | Full numeric data |
| `report_f1_full.txt` | sklearn `classification_report` at every `t` |
| `report_f1_triggered.txt` | Report on CALIMERA-triggered val predictions |

Console prints three side-by-side reports:
- Classification at `t=1` (first packet only)
- Classification at `t=20` (all packets = baseline)
- CALIMERA-triggered predictions (adaptive, avg `t★ ≈ 3.3 packets`)

### Switch to deep model

```python
USE_DEEP = True   # loads val_probas_deep.npy instead of recomputing from MiniRocket
```

### Run

```bash
python calimera/phase6b_f1_eval.py
```

---

## 11. Deep Model Extension (EarlyFlow)

Instead of training 20 separate MiniRocket classifiers (one per timestamp), a **single shared deep sequence model** is trained with **prefix training**. This is more parameter-efficient and learns temporal patterns jointly across all timestamps.

### Key Concept: Prefix Training

At each training batch, a random timestamp `t ~ Uniform(1, T_MAX)` is sampled. The model only sees `x[:, :t, :]` — the first `t` packets. This forces the model to make correct predictions at every prefix length:

```python
t = random.randint(1, cfg.T_MAX)
logits = model.forward_prefix(x, t)   # only first t packets
loss   = criterion(logits, y)
```

One model replaces 20 separate classifiers.

### Architectures (`calimera/models.py`)

All models share the same interface:
- `forward(x)` — full sequence `(B, T, F)` → logits `(B, n_classes)`
- `forward_prefix(x, t)` — only uses `x[:, :t, :]`

| Model | Architecture | Parameters | Best for |
|-------|-------------|-----------|---------|
| `lstm` | 2-layer LSTM, hidden=128, dropout=0.3, last hidden state | 202,756 | General use, recommended default |
| `gru` | 2-layer GRU, hidden=128, dropout=0.3 | 152,196 | Faster than LSTM, similar accuracy |
| `transformer` | CLS token + 2-layer TransformerEncoder, causal mask, HIDDEN=128 | 266,628 | Long-range dependencies, parallel training |
| `tcn` | 4 causal conv blocks (dilations 1,2,4,8), global avg pool | 153,220 | Fastest training, explicit causal structure |

### Platt Calibration (`CalibratedModel`)

After training, a `LogisticRegression` Platt scaler is fitted on val set logits to produce calibrated probabilities. CALIMERA requires well-calibrated `P(class | X)`.

```python
cal = CalibratedModel(model)
cal.calibrate(val_loader)          # fit Platt scaler
proba = cal.predict_proba(x, t=5)  # (B, 4) calibrated probabilities
```

### CALIMERA Bridge (`calimera/build_probas.py`)

After training, generate the probability arrays that CALIMERA needs:

```python
# For every flow in trig-set and val-set, run forward_prefix at every t
probas[:, t-1, :] = cal.predict_proba(X_tensor, t)   # shape: (n, 20, 4)
```

Saves `trig_probas_deep.npy` and `val_probas_deep.npy` — same shape as MiniRocket output. Everything downstream is unchanged.

### Central Config (`calimera/config.py`)

```python
MODEL    = "lstm"    # "lstm" | "gru" | "transformer" | "tcn"
HIDDEN   = 128
N_LAYERS = 2
EPOCHS   = 60
LR       = 1e-3
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
```

### Training

```bash
# LSTM (default)
python calimera/train_deep.py

# Transformer
python calimera/train_deep.py --model transformer

# Quick 20-epoch test
python calimera/train_deep.py --model gru --epochs 20

# Custom LR and hidden size
python calimera/train_deep.py --model tcn --epochs 40 --lr 5e-4 --hidden 256
```

### Model Comparison (`calimera/compare_models.py`)

After training multiple models, compare them all including the MiniRocket baseline:

```bash
python calimera/compare_models.py
```

Produces `compare_table.csv` and `compare_plot.png` — grouped bar chart of acc@t=1, acc@t=20, HM, and earliness for every trained model.

---



---

## 13. How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn sktime matplotlib
pip install torch torchvision torchaudio   # for deep models
```

> **GPU:** If a CUDA-capable GPU is available, PyTorch detects it automatically (`DEVICE = "cuda"`).

### Full MiniRocket pipeline

```bash
cd /path/to/earlyflow

# Step 0 (once): Process raw PCAPs into labeled CSVs
# Edit PCAP_DIR / OUTPUT_DIR in pcap_pipeline.py, then:
python pcap_pipeline.py

# Step 1: Build time-series arrays (set DATA_DIR in phase2_timeseries.py first)
python calimera/phase2_timeseries.py

# Step 2: Train MiniRocket + 20 classifiers + CALIMERA trigger
python calimera/phase3_calimera_train.py

# Step 3: Alpha sweep evaluation
python calimera/phase45_sweep_eval.py

# Step 4: Generate report
python calimera/phase6_report.py

# Step 5: F1/Precision/Recall analysis
python calimera/phase6b_f1_eval.py
```

### Deep model pipeline (after Phase 2)

```bash
# Train your chosen architecture
python calimera/train_deep.py --model transformer   # or lstm / gru / tcn

# Build CALIMERA probability arrays
python calimera/build_probas.py --model transformer

# In phase45_sweep_eval.py and phase6b_f1_eval.py, set:
#   USE_DEEP = True

# Run sweep and reports
python calimera/phase45_sweep_eval.py
python calimera/phase6_report.py
python calimera/phase6b_f1_eval.py

# Compare all models
python calimera/compare_models.py

# Triggered F1 evaluation across all models
python calimera/eval_triggered_all_models.py
```

### Train all 4 deep models sequentially

```bash
for MODEL in lstm gru transformer tcn; do
    python calimera/train_deep.py --model $MODEL
    python calimera/build_probas.py --model $MODEL
    python calimera/phase45_sweep_eval.py   # USE_DEEP=True
    cp calimera/eval/sweep_results.json calimera/eval/sweep_${MODEL}.json
done
python calimera/compare_models.py
```

---

## 14. Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, `.npy` file I/O |
| `pandas` | CSV loading, packet grouping |
| `scikit-learn` | RidgeClassifierCV, CalibratedClassifierCV, LabelEncoder, train_test_split |
| `sktime` | MiniRocket transformer |
| `torch` | LSTM / GRU / Transformer / TCN model training |
| `matplotlib` | All plots |
| `ml_edm` (local) | CALIMERA trigger, CostMatrices — lives in `ml_edm/src/` |

The `ml_edm` library is a local package (not on PyPI). It is accessed via:
```python
sys.path.insert(0, "ml_edm/src")
from ml_edm.trigger._calimera import CALIMERA
from ml_edm.cost_matrices import CostMatrices
```

---

## 15. Key Design Decisions

### Why zero-masking instead of 20 separate datasets?

MiniRocket is fitted **once** on the full T_MAX series, then the same transformer is applied to masked versions (packets after `t` are set to zero). This is faster and avoids refitting 20 transformers. The zero-padding signal ("flow ended here") is consistent with how training data is zero-padded.

### Why 70/30 clf/trigger split?

If the same data is used to train classifiers **and** fit the CALIMERA trigger, the trigger sees "in-sample" probabilities (overconfident, not representative of test-time behaviour). The 70/30 split ensures trigger probabilities are out-of-sample: the classifier was trained on 70%, so its probabilities on the 30% trigger set are unbiased.

### Why Platt scaling?

`RidgeClassifierCV` is a discriminative classifier — its raw outputs are decision function values, not calibrated probabilities. CALIMERA computes `P(class | X) · misclassification_cost`, which requires true probabilities. Sigmoid (Platt) calibration maps raw scores to well-calibrated `[0,1]` probabilities.

### Why prefix training for deep models?

The alternative is to train 20 separate models (one per timestamp). Prefix training uses one shared model — cheaper, faster, and the shared weights capture cross-timestamp patterns (e.g., "if packets 1–3 look like this, packet 5 will likely be...").

### Why KernelRidge for CALIMERA halters?

The original CALIMERA paper uses RBF KernelRidge for the halters. It is a non-parametric regression method that fits smooth decision boundaries in the probability feature space. Alternatives like linear regression or decision trees would either underfit or overfit the smooth cost-difference surface.

### Why α sweep instead of a single fixed α?

α is a domain-specific hyperparameter: in a QoS system, a network operator might accept lower accuracy for near-instant decisions (low α), while a security analyst needs high accuracy even at the cost of latency (high α). The sweep + Pareto plot exposes the full trade-off frontier so the operator can pick α for their deployment context.

---

## Citation / References

- CALIMERA: *Bilski et al., "CALIMERA: A new early time series classification method"* — [GitHub](https://github.com/JakubBilski/CALIMERA)
- MiniRocket: *Dempster et al., "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification"*, 2021
- ml_edm library: local fork providing `CALIMERA`, `CostMatrices`, `EarlyClassifier`
