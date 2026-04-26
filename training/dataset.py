"""
Memory-mapped token dataset for pre-training.

Expects binary files produced by tokenize_data.py:
    data/train.bin   -- flat uint32 array of token IDs
    data/val.bin     -- flat uint32 array of token IDs
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SyntheticDataset(Dataset):
    """Random token data — for testing."""

    def __init__(self, seq_len: int, vocab_size: int = 128_256, size: int = 100_000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(idx)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=g)
        return tokens[:-1], tokens[1:]


class TokenDataset(Dataset):
    """Training dataset"""

    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.num_sequences = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        # uint32 → int64 for nn.Embedding and F.cross_entropy
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def build_dataloaders(
    data_path: str,
    seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    train_ds = TokenDataset(f"{data_path}/train.bin", seq_len)
    val_ds   = TokenDataset(f"{data_path}/val.bin",   seq_len)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_sampler
