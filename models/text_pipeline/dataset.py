import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, csv_path, max_length=16):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = row["text"]
        label = torch.tensor(row["label"], dtype=torch.long)

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return input_ids, attention_mask, label
