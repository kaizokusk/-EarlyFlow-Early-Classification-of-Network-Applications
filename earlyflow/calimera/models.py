import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression

import calimera.config as cfg


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.N_FEAT,
            hidden_size=cfg.HIDDEN,
            num_layers=cfg.N_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT if cfg.N_LAYERS > 1 else 0.0,
        )
        self.drop = nn.Dropout(cfg.DROPOUT)
        self.fc   = nn.Linear(cfg.HIDDEN, cfg.N_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))

    def forward_prefix(self, x, t: int):
        out, _ = self.lstm(x[:, :t, :])
        return self.fc(self.drop(out[:, -1, :]))


# ── GRU ───────────────────────────────────────────────────────────────────────

class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cfg.N_FEAT,
            hidden_size=cfg.HIDDEN,
            num_layers=cfg.N_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT if cfg.N_LAYERS > 1 else 0.0,
        )
        self.drop = nn.Dropout(cfg.DROPOUT)
        self.fc   = nn.Linear(cfg.HIDDEN, cfg.N_CLASSES)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1, :]))

    def forward_prefix(self, x, t: int):
        out, _ = self.gru(x[:, :t, :])
        return self.fc(self.drop(out[:, -1, :]))


# ── Transformer ───────────────────────────────────────────────────────────────

class TransformerClassifier(nn.Module):
    """
    CLS token prepended; causal mask keeps prefix semantics at inference.
    CLS attends to all positions (first mask row is all-False).
    """
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(cfg.N_FEAT, cfg.HIDDEN)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, cfg.HIDDEN))
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=cfg.HIDDEN,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.FFN_DIM,
            dropout=cfg.DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.N_LAYERS)
        self.drop    = nn.Dropout(cfg.DROPOUT)
        self.fc      = nn.Linear(cfg.HIDDEN, cfg.N_CLASSES)

    def _causal_mask(self, seq_len: int, device):
        sz   = seq_len + 1           # +1 for CLS
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        mask[0, :] = False           # CLS attends to all positions
        return mask

    def _encode(self, x):
        B  = x.size(0)
        h  = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, h], dim=1)
        mask = self._causal_mask(x.size(1), x.device)
        out  = self.encoder(seq, mask=mask)
        return out[:, 0, :]          # CLS output

    def forward(self, x):
        return self.fc(self.drop(self._encode(x)))

    def forward_prefix(self, x, t: int):
        return self.fc(self.drop(self._encode(x[:, :t, :])))


# ── TCN ───────────────────────────────────────────────────────────────────────

class _CausalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv(x)
        h = h[:, :, :x.size(2)]    # trim to causal length
        return self.act(self.norm(h)) + self.skip(x)


class TCNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        dilations = [1, 2, 4, 8]
        layers = [_CausalConvBlock(cfg.N_FEAT, cfg.HIDDEN, dilation=dilations[0])]
        for d in dilations[1:]:
            layers.append(_CausalConvBlock(cfg.HIDDEN, cfg.HIDDEN, dilation=d))
        self.tcn  = nn.Sequential(*layers)
        self.drop = nn.Dropout(cfg.DROPOUT)
        self.fc   = nn.Linear(cfg.HIDDEN, cfg.N_CLASSES)

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))    # (B, HIDDEN, T)
        return self.fc(self.drop(h.mean(dim=2)))

    def forward_prefix(self, x, t: int):
        h = self.tcn(x[:, :t, :].transpose(1, 2))
        return self.fc(self.drop(h.mean(dim=2)))


# ── Factory ───────────────────────────────────────────────────────────────────

def get_model(model_name: str = None) -> nn.Module:
    name = (model_name or cfg.MODEL).lower()
    registry = {
        "lstm":        LSTMClassifier,
        "gru":         GRUClassifier,
        "transformer": TransformerClassifier,
        "tcn":         TCNClassifier,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name]()


# ── Calibration wrapper ───────────────────────────────────────────────────────

class CalibratedModel:
    """
    Wraps any base model with Platt scaling.

    Usage:
        cal = CalibratedModel(model)
        cal.calibrate(val_loader)
        proba = cal.predict_proba(x, t=5)   # (B, n_classes) numpy float32
        cal.save("calimera/models/cal_lstm")
        cal.load("calimera/models/cal_lstm")
    """

    def __init__(self, base_model: nn.Module):
        self.model = base_model.to(cfg.DEVICE)
        self.platt = None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def calibrate(self, val_loader):
        self.model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = self.model(x.to(cfg.DEVICE)).cpu().numpy()
                all_logits.append(logits)
                all_y.append(y.numpy())
        logits_np = np.concatenate(all_logits)
        y_np      = np.concatenate(all_y)
        self.platt = LogisticRegression(max_iter=1000, C=1.0, random_state=cfg.SEED)
        self.platt.fit(self._softmax(logits_np), y_np)
        print(f"  Platt scaler fitted on {len(y_np):,} val samples.")

    def predict_proba(self, x: torch.Tensor, t: int) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward_prefix(x.to(cfg.DEVICE), t).cpu().numpy()
        proba = self._softmax(logits)
        if self.platt is not None:
            proba = self.platt.predict_proba(proba)
        return proba.astype(np.float32)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path + ".pt")
        with open(path + ".platt.pkl", "wb") as f:
            pickle.dump(self.platt, f)
        print(f"  Saved {path}.pt  +  {path}.platt.pkl")

    def load(self, path: str):
        self.model.load_state_dict(
            torch.load(path + ".pt", map_location=cfg.DEVICE, weights_only=True)
        )
        with open(path + ".platt.pkl", "rb") as f:
            self.platt = pickle.load(f)
