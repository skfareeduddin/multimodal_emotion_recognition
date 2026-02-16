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

from models.text_pipeline.dataset import TextDataset
from models.text_pipeline.model import TextModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():

    dataset = TextDataset("data/metadata.csv")
    loader = DataLoader(dataset, batch_size=8)

    model = TextModel().to(device)
    model.load_state_dict(
        torch.load("models/text_pipeline/checkpoints/best_model.pt")
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs, _ = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print("Text Model Results")
    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    os.makedirs("Results/plots", exist_ok=True)
    os.makedirs("Results/accuracy_tables", exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Text Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("Results/plots/text_confusion_matrix.png")
    plt.close()

    report = classification_report(all_labels, all_preds, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        "Results/accuracy_tables/text_classification_report.csv"
    )

    metrics_df = pd.DataFrame({
        "Model": ["Text"],
        "Accuracy": [accuracy],
        "Macro_F1": [macro_f1],
        "Weighted_F1": [weighted_f1]
    })

    metrics_df.to_csv("Results/accuracy_tables/text_metrics.csv", index=False)

    print("Results saved in Results/ folder.")


if __name__ == "__main__":
    test()
