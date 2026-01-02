import os
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
from ..constants import NOTE_ON_OFFSET, DUR_OFFSET, TIME_SHIFT_OFFSET, VELOCITY_OFFSET

class JazzMIDIDataset(Dataset):
    def __init__(self, root, vocab_size, seq_len):
        # self.files = [
        #     os.path.join(root, f)
        #     for f in os.listdir(root)
        #     if f.endswith(".pt")
        # ]
        self.seq_len = seq_len #128 #2048 #10240
        self.files = sorted(Path(root).rglob("*.pt"))
        self.VOCAB_SIZE = vocab_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        seq = torch.load(self.files[idx])  # shape (N,)
        assert seq.dtype == torch.long
        assert seq.min() >= 0
        assert seq.max() < self.VOCAB_SIZE

        if len(seq) >= self.seq_len:
            start = random.randint(0, len(seq) - self.seq_len)
            while (NOTE_ON_OFFSET <= seq[start] < DUR_OFFSET) or (DUR_OFFSET <= seq[start] < TIME_SHIFT_OFFSET):
                start = random.randint(0, len(seq) - self.seq_len)
            seq = seq[start:start+self.seq_len]
        else:
            pad = 0
            padded = torch.full((self.seq_len,), pad, dtype=torch.long)
            padded[: seq.size(0)] = seq
            seq = padded
        
        return seq
