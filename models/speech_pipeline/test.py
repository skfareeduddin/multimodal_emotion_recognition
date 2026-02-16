import torch
from torch.utils.data import DataLoader
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.speech_pipeline.dataset import SpeechDataset
from models.speech_pipeline.model import SpeechModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():

    dataset = SpeechDataset(
        csv_path="data/metadata.csv",
        spec_root="data/spectrograms"
    )

    loader = DataLoader(dataset, batch_size=64)

    model = SpeechModel().to(device)
    model.load_state_dict(
        torch.load("models/speech_pipeline/checkpoints/best_model.pt")
    )
    model.eval()

    all_preds = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            outputs, embeddings = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_embeddings.extend(embeddings.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print("Speech Model Results")
    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    os.makedirs("Results/plots", exist_ok=True)
    os.makedirs("Results/accuracy_tables", exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Speech Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("Results/plots/speech_confusion_matrix.png")
    plt.close()

    report = classification_report(all_labels, all_preds, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        "Results/accuracy_tables/speech_classification_report.csv"
    )

    metrics_df = pd.DataFrame({
        "Model": ["Speech"],
        "Accuracy": [accuracy],
        "Macro_F1": [macro_f1],
        "Weighted_F1": [weighted_f1]
    })

    metrics_df.to_csv("Results/accuracy_tables/speech_metrics.csv", index=False)

    print("Results saved in Results/ folder.")


if __name__ == "__main__":
    test()
