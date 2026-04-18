import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# CONFIG
# -----------------------------
PARQUET_FILE = "output.parquet"

FEATURES = [
    "frame.len",
    "iat",
    "signed_len",
    "len_diff",
    "relative_time",
]

# 🔥 better K values
K_VALUES     = [5, 10, 20, 30]
MIN_PACKETS  = 5
MAX_PACKETS  = 50

# 🔥 CLASS REDUCTION (CRITICAL FIX)
CLASS_MAP = {
    "streaming": "streaming",
    "messaging": "messaging",
    "web": "web",

    "ads": "web",
    "cloud/api": "web",
    "system": "web",
    "local/network": "web"
}

# -----------------------------
# STREAM LOAD
# -----------------------------
print("Streaming parquet safely...")

parquet_file = pq.ParquetFile(PARQUET_FILE)

dfs = []

for i, batch in enumerate(parquet_file.iter_batches(batch_size=50000)):
    print(f"Reading batch {i}")

    df_batch = batch.to_pandas()

    df_batch = df_batch[
        ["flow_key", "frame.time_epoch", "label"] + FEATURES
    ]

    dfs.append(df_batch)

    if i == 6:   # ~400k rows
        break

df = pd.concat(dfs, ignore_index=True)

print("Loaded subset:", df.shape)

# -----------------------------
# PREPROCESS
# -----------------------------
df = df.sort_values(["flow_key", "frame.time_epoch"])
df[FEATURES] = df[FEATURES].fillna(0)

# 🔥 APPLY CLASS REDUCTION
df["label"] = df["label"].map(CLASS_MAP)

# remove unknown labels (if any)
df = df.dropna(subset=["label"])

# -----------------------------
# BUILD SEQUENCES
# -----------------------------
print("Building sequences...")

sequences = []
labels = []

for _, g in df.groupby("flow_key"):
    if len(g) >= MIN_PACKETS:

        g_vals = g[FEATURES].values

        # 🔥 normalize per flow
        g_vals = (g_vals - g_vals.mean(axis=0)) / (g_vals.std(axis=0) + 1e-6)

        sequences.append(g_vals)
        labels.append(g["label"].iloc[0])

print("Sequences:", len(sequences))

# -----------------------------
# LABEL ENCODING
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(labels)

print("Classes:", list(le.classes_))

# -----------------------------
# SPLIT
# -----------------------------
idx = np.arange(len(sequences))

train_idx, test_idx = train_test_split(
    idx,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# -----------------------------
# SCALING
# -----------------------------
def prepare_X(seq_list):
    X = pad_sequences(seq_list, maxlen=MAX_PACKETS, padding="post", dtype="float32")

    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    return X

# -----------------------------
# TRAIN + EVALUATE
# -----------------------------
results = []

for K in K_VALUES:
    print(f"\n===== K = {K} =====")

    seq_k = [s[:K] for s in sequences]

    X = prepare_X(seq_k)

    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    # 🔥 stronger model
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(MAX_PACKETS, len(FEATURES))),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(len(le.classes_), activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=256,
        class_weight=class_weights,
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    results.append({"K": K, "accuracy": acc})

# -----------------------------
# PLOT
# -----------------------------
res_df = pd.DataFrame(results)

plt.figure(figsize=(8, 4))
plt.plot(res_df["K"], res_df["accuracy"], marker="o")

plt.xlabel("K packets")
plt.ylabel("Accuracy")
plt.title("Early Traffic Classification (Improved)")
plt.grid()

plt.savefig("early_plot.png", dpi=150)
plt.show()

print("✅ DONE")
