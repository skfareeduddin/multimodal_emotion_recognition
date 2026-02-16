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

from models.fusion_pipeline.dataset import FusionDataset
from models.fusion_pipeline.model import FusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():

    dataset = FusionDataset(
        csv_path="data/metadata.csv",
        spec_root="data/spectrograms"
    )

    loader = DataLoader(dataset, batch_size=16)

    model = FusionModel().to(device)
    model.load_state_dict(
        torch.load("models/fusion_pipeline/checkpoints/best_model.pt")
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for spec, input_ids, attention_mask, labels in loader:

            spec = spec.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs, _ = model(spec, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print("Fusion Model Results")
    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    os.makedirs("Results/plots", exist_ok=True)
    os.makedirs("Results/accuracy_tables", exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Fusion Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("Results/plots/fusion_confusion_matrix.png")
    plt.close()

    report = classification_report(all_labels, all_preds, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        "Results/accuracy_tables/fusion_classification_report.csv"
    )

    metrics_df = pd.DataFrame({
        "Model": ["Fusion"],
        "Accuracy": [accuracy],
        "Macro_F1": [macro_f1],
        "Weighted_F1": [weighted_f1]
    })

    metrics_df.to_csv("Results/accuracy_tables/fusion_metrics.csv", index=False)

    print("Results saved in Results/ folder.")


if __name__ == "__main__":
    test()
