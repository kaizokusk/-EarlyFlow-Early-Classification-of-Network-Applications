import torch

# ── Model selection ───────────────────────────────────────────────
MODEL = "lstm"          # "lstm" | "gru" | "transformer" | "tcn"

# ── Data dimensions ───────────────────────────────────────────────
T_MAX     = 20
N_FEAT    = 7
N_CLASSES = 4

# ── Model hyperparams ─────────────────────────────────────────────
HIDDEN    = 128
N_LAYERS  = 2
N_HEADS   = 4           # Transformer only
FFN_DIM   = 256         # Transformer only
DROPOUT   = 0.3

# ── Training ─────────────────────────────────────────────────────
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 60
BATCH        = 256
SEED         = 42

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR  = "calimera/data"
MODEL_DIR = "calimera/models"
EVAL_DIR  = "calimera/eval"

# ── Device ───────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── CALIMERA ──────────────────────────────────────────────────────
TRIGGER_FRAC  = 0.30
ALPHA_DEFAULT = 0.5
ALPHA_SWEEP   = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
