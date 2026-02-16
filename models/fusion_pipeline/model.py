import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.speech_pipeline.model import SpeechModel
from models.text_pipeline.model import TextModel

class FusionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.speech_model = SpeechModel(num_classes=num_classes)
        self.text_model = TextModel(num_classes=num_classes)

        self.speech_model.load_state_dict(
            torch.load("models/speech_pipeline/checkpoints/best_model.pt")
        )

        self.text_model.load_state_dict(
            torch.load("models/text_pipeline/checkpoints/best_model.pt")
        )

        for param in self.text_model.bert.parameters():
            param.requires_grad = False

        self.speech_proj = nn.Linear(128, 256)
        self.text_proj = nn.Linear(768, 256)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, spec, input_ids, attention_mask):

        speech_logits, speech_emb = self.speech_model(spec)
        text_logits, text_emb = self.text_model(input_ids, attention_mask)

        s = self.speech_proj(speech_emb)
        t = self.text_proj(text_emb)

        fusion = torch.cat([s, t], dim=1)

        logits = self.classifier(fusion)

        return logits, fusion
