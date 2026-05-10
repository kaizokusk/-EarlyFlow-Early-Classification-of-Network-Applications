import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import calimera.config as cfg


class FlowDataset(Dataset):
    def __init__(self, split: str = "train"):
        data_dir = Path(cfg.DATA_DIR)
        self.X = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        self.y = np.load(data_dir / f"y_{split}.npy").astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


def get_loaders(batch_size: int = None) -> dict:
    if batch_size is None:
        batch_size = cfg.BATCH
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = FlowDataset(split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=(cfg.DEVICE == "cuda"),
        )
    return loaders
