import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.fusion_pipeline.dataset import FusionDataset
from models.fusion_pipeline.model import FusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():

    dataset = FusionDataset(
        csv_path="data/metadata.csv",
        spec_root="data/spectrograms"
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = FusionModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0

    for epoch in range(5):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")

        for spec, input_ids, attention_mask, labels in loop:

            spec = spec.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs, _ = model(spec, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for spec, input_ids, attention_mask, labels in val_loader:

                spec = spec.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs, _ = model(spec, input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"\nValidation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models/fusion_pipeline/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "models/fusion_pipeline/checkpoints/best_model.pt")

    print("\nBest Validation Accuracy:", best_val_acc)


if __name__ == "__main__":
    train()
