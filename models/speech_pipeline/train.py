import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.speech_pipeline.dataset import SpeechDataset
from models.speech_pipeline.model import SpeechModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():

    dataset = SpeechDataset(
        csv_path="data/metadata.csv",
        spec_root="data/spectrograms"
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = SpeechModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0

    for epoch in range(6):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/6")

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs, _ = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs, _ = model(x)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"\nValidation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models/speech_pipeline/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "models/speech_pipeline/checkpoints/best_model.pt")

    print("\nBest Validation Accuracy:", best_val_acc)


if __name__ == "__main__":
    train()
