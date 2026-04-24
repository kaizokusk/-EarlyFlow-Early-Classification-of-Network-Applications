

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, accuracy_score)
from xgboost import XGBClassifier
from ml_edm.classification.classifiers_collection import ClassifiersCollection
from ml_edm.cost_matrices import CostMatrices
from ml_edm.trigger import EconomyGamma
from ml_edm.early_classifier import EarlyClassifier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
# 0. CONFIG 
CHECKPOINTS   = [3, 5, 10, 20, 50]
ALPHAS        = [0.1, 0.3, 0.5, 0.7, 0.9]
RANDOM_STATE  = 42
TEST_SIZE     = 0.2

FLOW_FEATURES = [
    'duration', 'pkt_count', 'bytes_total',
    'pkt_len_mean', 'pkt_len_min', 'pkt_len_max',
    'src_port', 'dst_port',
    'port_is_443', 'port_is_80', 'port_is_53',
    'dns_query_count', 'has_dns',
    'bytes_per_pkt',
    'proto_-1', 'proto_1', 'proto_17', 'proto_6'
]

EARLY_FEATURES = [
    "pkt_count", "bytes_total",
    "pkt_len_mean", "pkt_len_std", "pkt_len_min", "pkt_len_max",
    "iat_mean", "iat_std", "iat_min", "iat_max",
    "has_dns", "port_is_443", "port_is_80", "port_is_53"
]

# XGBoost params (same for both baseline and early classifiers)
XGB_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 6,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    use_label_encoder= False,
    eval_metric      = 'mlogloss',
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    verbosity        = 0,
)
 
# 1. LOAD DATA 
print("Loading data...")
flow = pd.read_parquet("flow_level.parquet")
pkt  = pd.read_parquet("packet_level.parquet")

print(f"Flow-level rows : {len(flow):,}")
print(f"Packet-level rows: {len(pkt):,}")
print(f"\nTraffic type distribution (flow level):")
print(flow["traffic_type"].value_counts())
 
# 2. BASELINE — XGBoost on full flow-level features 
print("\n" + "="*60)
print("BASELINE: XGBoost on full flow features")
print("="*60)

flow_clean = flow[flow["traffic_type"] != "other"].copy()

# Keep only columns that exist
available_flow_feats = [c for c in FLOW_FEATURES if c in flow_clean.columns]
print(f"Flow features used: {available_flow_feats}")

le_base = LabelEncoder()
X_flow  = flow_clean[available_flow_feats].fillna(0).values
y_flow  = le_base.fit_transform(flow_clean["traffic_type"].values)

X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(
    X_flow, y_flow, test_size=TEST_SIZE, stratify=y_flow, random_state=RANDOM_STATE
)

baseline_xgb = XGBClassifier(**XGB_PARAMS)
baseline_xgb.fit(X_flow_train, y_flow_train)

baseline_preds = baseline_xgb.predict(X_flow_test)
baseline_acc   = accuracy_score(y_flow_test, baseline_preds)

print(f"\nBaseline Accuracy: {baseline_acc:.4f}")
print(classification_report(y_flow_test, baseline_preds,
                             target_names=le_base.classes_, zero_division=0))

base_report = classification_report(y_flow_test, baseline_preds,
                                    target_names=le_base.classes_,
                                    output_dict=True, zero_division=0)
base_cm = confusion_matrix(y_flow_test, baseline_preds)
 
# 3. EARLY FEATURES — Build snapshots from packet-level data 
print("\n" + "="*60)
print("EARLY: Building packet snapshots")
print("="*60)

# Derive flow_id and pkt_rank if missing
if "flow_id" not in pkt.columns:
    proto  = pkt["ip.proto"].astype(str).str.split(',').str[0].replace('nan','-1').fillna('-1')
    sp     = pkt["tcp.srcport"].fillna(pkt["udp.srcport"]).fillna(0).astype(np.int32)
    dp     = pkt["tcp.dstport"].fillna(pkt["udp.dstport"]).fillna(0).astype(np.int32)
    src    = pkt["ip.src"].fillna("0.0.0.0")
    dst    = pkt["ip.dst"].fillna("0.0.0.0")
    swap   = (src > dst) | ((src == dst) & (sp > dp))
    a_ip   = np.where(swap, dst, src);  b_ip = np.where(swap, src, dst)
    a_port = np.where(swap, dp,  sp);   b_port = np.where(swap, sp, dp)
    pkt["flow_key"] = (pd.Series(a_ip) + "_" + pd.Series(b_ip) + "_" +
                       proto.values + "_" + pd.Series(a_port.astype(str)) + "_" +
                       pd.Series(b_port.astype(str))).values
    pkt["flow_id"]  = pd.Categorical(pkt["flow_key"]).codes.astype(np.int32)

if "pkt_rank" not in pkt.columns:
    pkt = pkt.sort_values(["flow_id", "frame.time_epoch"]).reset_index(drop=True)
    pkt["pkt_rank"] = pkt.groupby("flow_id").cumcount()

# Filter flows with enough packets
flow_max_rank  = pkt.groupby("flow_id")["pkt_rank"].max()
valid_flow_ids = flow_max_rank[flow_max_rank >= CHECKPOINTS[0]].index
pkt_valid      = pkt[pkt["flow_id"].isin(valid_flow_ids)]
print(f"Valid flows (≥{CHECKPOINTS[0]} pkts): {len(valid_flow_ids):,}")

# Build snapshots per checkpoint
snapshots = {}
for N in CHECKPOINTS:
    snap = (
        pkt_valid[pkt_valid["pkt_rank"] < N]
        .groupby("flow_id")
        .agg(
            pkt_count    = ("frame.len",   "count"),
            bytes_total  = ("frame.len",   "sum"),
            pkt_len_mean = ("frame.len",   "mean"),
            pkt_len_std  = ("frame.len",   lambda x: x.std(ddof=0) if len(x)>1 else 0.0),
            pkt_len_min  = ("frame.len",   "min"),
            pkt_len_max  = ("frame.len",   "max"),
            iat_mean     = ("iat",         "mean"),
            iat_std      = ("iat",         lambda x: x.std(ddof=0) if len(x)>1 else 0.0),
            iat_min      = ("iat",         "min"),
            iat_max      = ("iat",         "max"),
            has_dns      = ("is_dns",      "max"),
            port_is_443  = ("port_is_443", "max"),
            port_is_80   = ("port_is_80",  "max"),
            port_is_53   = ("port_is_53",  "max"),
        )
    )
    snapshots[N] = snap
    print(f"  N={N:>3}: {len(snap):,} flows")

# Labels
flow_labels = (
    pkt_valid.groupby("flow_id")["traffic_type"]
    .agg(lambda x: x[x != "other"].iloc[0] if (x != "other").any() else "other")
)
flow_labels = flow_labels[flow_labels != "other"]

common_ids = set(flow_labels.index)
for N in CHECKPOINTS:
    common_ids &= set(snapshots[N].index)
common_ids = sorted(common_ids)
print(f"Flows with all checkpoints + label: {len(common_ids):,}")

le = LabelEncoder()
y  = le.fit_transform(flow_labels.loc[common_ids].values)
print(f"Classes: {le.classes_}")

# Build 3D → flatten to 2D for ML-EDM
n_flows = len(common_ids)
n_ckpts = len(CHECKPOINTS)
n_feats = len(EARLY_FEATURES)

X_3d = np.zeros((n_flows, n_ckpts, n_feats), dtype=np.float32)
for c_idx, N in enumerate(CHECKPOINTS):
    snap = snapshots[N].loc[common_ids]
    X_3d[:, c_idx, :] = snap[EARLY_FEATURES].fillna(0).values

# Forward-fill
for i in range(n_flows):
    for c in range(1, n_ckpts):
        if X_3d[i, c, 0] == 0 and X_3d[i, c-1, 0] != 0:
            X_3d[i, c, :] = X_3d[i, c-1, :]

# Normalize
scaler    = StandardScaler()
X_2d_norm = scaler.fit_transform(X_3d.reshape(-1, n_feats))
X_3d_norm = X_2d_norm.reshape(n_flows, n_ckpts, n_feats).astype(np.float32)
X_flat    = X_3d_norm.reshape(n_flows, n_ckpts * n_feats)
timestamps = np.array([(i+1) * n_feats for i in range(n_ckpts)])

print(f"\nX_flat shape      : {X_flat.shape}")
print(f"ML-EDM timestamps : {timestamps}")

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
 
# 4. PER-CHECKPOINT XGBoost (static early classifiers) 
print("\n" + "="*60)
print("Per-checkpoint XGBoost (static early classifiers)")
print("="*60)

ckpt_accs    = []
ckpt_reports = []
ckpt_cms     = []

for c_idx, N in enumerate(CHECKPOINTS):
    col_s = c_idx * n_feats
    col_e = col_s + n_feats
    clf   = XGBClassifier(**XGB_PARAMS)
    clf.fit(X_train[:, col_s:col_e], y_train)
    preds = clf.predict(X_test[:, col_s:col_e])
    acc   = accuracy_score(y_test, preds)
    ckpt_accs.append(acc)
    ckpt_reports.append(classification_report(y_test, preds, target_names=le.classes_,
                                              output_dict=True, zero_division=0))
    ckpt_cms.append(confusion_matrix(y_test, preds))
    print(f"  N={N:>3} packets → accuracy = {acc:.4f}")

per_class_recall = {cls: [r[cls]["recall"] for r in ckpt_reports] for cls in le.classes_}
 
# 5. ML-EDM EARLY CLASSIFIER with XGBoost base 
print("\n" + "="*60)
print("ML-EDM EarlyClassifier with XGBoost base")
print("="*60)

n_cls = len(le.classes_)
#misclf_cost = 1 - np.eye(n_cls)

cost_matrices = CostMatrices(
    timestamps=timestamps,
    n_classes=n_cls,
    alpha=0.5
)
clf_stack = ClassifiersCollection(
    base_classifier=XGBClassifier(**XGB_PARAMS),
    min_length=1,
    timestamps=timestamps
)

ec = EarlyClassifier(
    chronological_classifiers=clf_stack,
    trigger_model=EconomyGamma(timestamps),
    cost_matrices=cost_matrices,
    trigger_proportion=0.3,
    random_state=RANDOM_STATE
)
ec.fit(X_train, y_train)
metrics = ec.score(X_test, y_test, return_metrics=True)

print(f"\nML-EDM EarlyClassifier (XGBoost) Results:")
print(f"  Accuracy      : {metrics['accuracy']:.4f}")
print(f"  Earliness     : {metrics['earliness']:.4f}  (lower = faster)")
print(f"  Harmonic Mean : {metrics['harmonic_mean']:.4f}")
print(f"  Avg Cost      : {metrics['average_cost']:.4f}")
print(f"  Kappa         : {metrics['kappa']:.4f}")
 
# 6. ALPHA SWEEP 
print("\n── Alpha sweep ──")
alpha_results = []

for alpha in ALPHAS:
    cm_a = CostMatrices(timestamps=timestamps, n_classes=n_cls, alpha=alpha)
    ec_a = EarlyClassifier(
        chronological_classifiers=ClassifiersCollection(
            base_classifier=XGBClassifier(**XGB_PARAMS),
            min_length=1, timestamps=timestamps
        ),
        trigger_model=EconomyGamma(timestamps),
        cost_matrices=cm_a,
        trigger_proportion=0.3,
        random_state=RANDOM_STATE
    )
    ec_a.fit(X_train, y_train)
    m = ec_a.score(X_test, y_test, return_metrics=True)
    alpha_results.append({"alpha": alpha, "accuracy": m["accuracy"], "earliness": m["earliness"]})
    print(f"  alpha={alpha} | acc={m['accuracy']:.4f} | earliness={m['earliness']:.4f}")
 
# 7. PLOTS 
print("\nGenerating plots...")
fig    = plt.figure(figsize=(22, 20))
gs     = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
colors = plt.cm.tab10.colors
t_star = np.array(metrics["t_star"])

# ── Plot 1: Packets vs Accuracy (early vs baseline) ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(CHECKPOINTS, ckpt_accs, marker='o', color='steelblue',
         linewidth=2, label='XGBoost Early (N pkts)')
ax1.axhline(baseline_acc, color='red', linestyle='--', linewidth=1.5,
            label=f'XGBoost Baseline (full flow) {baseline_acc:.3f}')
ax1.fill_between(CHECKPOINTS, ckpt_accs, baseline_acc, alpha=0.1, color='red')
ax1.set_xlabel("Packets observed"); ax1.set_ylabel("Accuracy")
ax1.set_title("① Early vs Baseline Accuracy\n(cost of early decision)")
ax1.legend(fontsize=8); ax1.grid(True)

# ── Plot 2: Latency vs Accuracy tradeoff (alpha sweep) ───────────────────────
ax2 = fig.add_subplot(gs[0, 1])
res_df = pd.DataFrame(alpha_results)
ax2.plot(res_df["earliness"], res_df["accuracy"],
         marker='o', color='darkorange', linewidth=2)
for _, row in res_df.iterrows():
    ax2.annotate(f"α={row['alpha']}", (row["earliness"], row["accuracy"]),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)
ax2.set_xlabel("Earliness (lower = faster)"); ax2.set_ylabel("Accuracy")
ax2.set_title("② Latency vs Accuracy Tradeoff\n(α=0 accuracy-first, α=1 speed-first)")
ax2.grid(True)

# ── Plot 3: Trigger distribution ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(t_star, bins=20, color='mediumseagreen', edgecolor='black', rwidth=0.85)
ax3.set_xlabel("Timestamp at decision"); ax3.set_ylabel("Flows")
ax3.set_title("③ When does EarlyClassifier trigger?\n(earlier = more efficient)")
ax3.grid(True)

# ── Plot 4: Per-class recall vs checkpoint ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for i, cls in enumerate(le.classes_):
    ax4.plot(CHECKPOINTS, per_class_recall[cls], marker='o',
             label=cls, color=colors[i % len(colors)], linewidth=2)
ax4.set_xlabel("Packets observed"); ax4.set_ylabel("Recall")
ax4.set_title("④ Per-Class Recall vs Packets\n(which classes need more?)")
ax4.legend(fontsize=7); ax4.grid(True)

# ── Plot 5: Confusion matrix at N=3 ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ConfusionMatrixDisplay(confusion_matrix=ckpt_cms[0], display_labels=le.classes_).plot(
    ax=ax5, colorbar=False, cmap='Blues')
ax5.set_title(f"⑤ Confusion @ N={CHECKPOINTS[0]} pkts (earliest)")
plt.setp(ax5.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax5.get_yticklabels(), fontsize=7)

# ── Plot 6: Confusion matrix at N=50 ────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ConfusionMatrixDisplay(confusion_matrix=ckpt_cms[-1], display_labels=le.classes_).plot(
    ax=ax6, colorbar=False, cmap='Greens')
ax6.set_title(f"⑥ Confusion @ N={CHECKPOINTS[-1]} pkts (most info)")
plt.setp(ax6.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax6.get_yticklabels(), fontsize=7)

# ── Plot 7: Trigger time per class ───────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
for i, cls_id in enumerate(np.unique(y_test)):
    mask  = (y_test == cls_id)
    cls_t = t_star[mask]
    ax7.scatter(np.full(len(cls_t), i), cls_t,
                alpha=0.2, s=8, color=colors[i % len(colors)])
    ax7.scatter(i, cls_t.mean(), marker='D', s=80,
                color=colors[i % len(colors)], edgecolors='black', zorder=5)
ax7.set_xticks(range(len(le.classes_)))
ax7.set_xticklabels(le.classes_, rotation=30, ha='right', fontsize=8)
ax7.set_ylabel("Trigger timestamp"); ax7.grid(True, axis='y')
ax7.set_title("⑦ Trigger Time per Class\n(diamond=mean, dots=flows)")

# ── Plot 8: Early (N=3) vs Baseline recall per class ─────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
early_recall_n3 = [ckpt_reports[0][cls]["recall"] for cls in le.classes_]
# Align baseline labels with early labels
base_recall = []
for cls in le.classes_:
    if cls in base_report:
        base_recall.append(base_report[cls]["recall"])
    else:
        base_recall.append(0.0)

x_pos = np.arange(len(le.classes_))
ax8.bar(x_pos - 0.2, early_recall_n3, 0.4,
        label=f'Early N={CHECKPOINTS[0]}', color='steelblue')
ax8.bar(x_pos + 0.2, base_recall,      0.4,
        label='Full-flow baseline',      color='tomato')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(le.classes_, rotation=30, ha='right', fontsize=8)
ax8.set_ylabel("Recall"); ax8.set_ylim(0, 1.1)
ax8.set_title("⑧ Early (N=3) vs Baseline Recall per Class")
ax8.legend(fontsize=8); ax8.grid(True, axis='y')

# ── Plot 9: Summary table ─────────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
table_data = [
    ["Metric",                  "Value"],
    ["Baseline Accuracy",       f"{baseline_acc:.4f}"],
    ["Early Accuracy (α=0.5)",  f"{metrics['accuracy']:.4f}"],
    ["Accuracy Gap",            f"{baseline_acc - metrics['accuracy']:+.4f}"],
    ["Earliness",               f"{metrics['earliness']:.4f}"],
    ["Harmonic Mean",           f"{metrics['harmonic_mean']:.4f}"],
    ["Avg Cost",                f"{metrics['average_cost']:.4f}"],
    ["Kappa",                   f"{metrics['kappa']:.4f}"],
    ["Classes",                 str(len(le.classes_))],
    ["Test flows (early)",      f"{len(y_test):,}"],
    ["Test flows (baseline)",   f"{len(y_flow_test):,}"],
]
tbl = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.7)
ax9.set_title("⑨ Summary", fontsize=11, pad=10)

plt.suptitle("EarlyFlow — XGBoost Early vs Full-Flow Baseline",
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig("earlyflow_xgboost_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved earlyflow_xgboost_results.png")
 
# 8. PRINT FINAL COMPARISON 
print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)
print(f"{'Method':<35} {'Accuracy':>10} {'Earliness':>12}")
print("-"*60)
print(f"{'XGBoost Baseline (full flow)':<35} {baseline_acc:>10.4f} {'1.0000':>12}")
for c_idx, N in enumerate(CHECKPOINTS):
    print(f"{'XGBoost Early N='+str(N):<35} {ckpt_accs[c_idx]:>10.4f} {N/CHECKPOINTS[-1]:>12.4f}")
print(f"{'ML-EDM EarlyClassifier (α=0.5)':<35} {metrics['accuracy']:>10.4f} {metrics['earliness']:>12.4f}")
print("="*60)
