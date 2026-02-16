import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class SpeechDataset(Dataset):
    def __init__(self, csv_path, spec_root):
        self.df = pd.read_csv(csv_path)
        self.spec_root = spec_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        spec_path = os.path.join(
            self.spec_root,
            row["file_path"] + ".pt"
        )

        spec = torch.load(spec_path)
        label = torch.tensor(row["label"], dtype=torch.long)

        return spec, label
