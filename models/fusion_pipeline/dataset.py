import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from transformers import BertTokenizer

class FusionDataset(Dataset):
    def __init__(self, csv_path, spec_root, max_length=16):
        self.df = pd.read_csv(csv_path)
        self.spec_root = spec_root
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        spec_path = os.path.join(
            self.spec_root,
            row["file_path"] + ".pt"
        )
        spec = torch.load(spec_path)

        encoding = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        label = torch.tensor(row["label"], dtype=torch.long)

        return spec, input_ids, attention_mask, label
