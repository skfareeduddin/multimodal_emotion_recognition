import torch
import torch.nn as nn
from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.pooler_output

        x = self.dropout(cls_embedding)
        logits = self.classifier(x)

        return logits, cls_embedding
