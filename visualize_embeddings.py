import torch
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath("."))

from models.speech_pipeline.dataset import SpeechDataset
from models.speech_pipeline.model import SpeechModel

from models.text_pipeline.dataset import TextDataset
from models.text_pipeline.model import TextModel

from models.fusion_pipeline.dataset import FusionDataset
from models.fusion_pipeline.model import FusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_labels = [
    "angry", "disgust", "fear",
    "happy", "neutral", "pleasant_surprise", "sad"
]


def plot_tsne(embeddings, labels, title, save_path):

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    for i in range(len(emotion_labels)):
        idx = np.where(labels == i)
        plt.scatter(
            reduced[idx, 0],
            reduced[idx, 1],
            label=emotion_labels[i],
            alpha=0.6
        )

    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def extract_speech_embeddings():
    dataset = SpeechDataset(
        csv_path="data/metadata.csv",
        spec_root="data/spectrograms"
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    model = SpeechModel().to(device)
    model.load_state_dict(
        torch.load("models/speech_pipeline/checkpoints/best_model.pt")
    )
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for spec, labels in loader:
            spec = spec.to(device)
            _, emb = model(spec)

            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_embeddings), np.hstack(all_labels)


def extract_text_embeddings():
    dataset = TextDataset("data/metadata.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    model = TextModel().to(device)
    model.load_state_dict(
        torch.load("models/text_pipeline/checkpoints/best_model.pt")
    )
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            _, emb = model(input_ids, attention_mask)

            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_embeddings), np.hstack(all_labels)


def extract_fusion_embeddings():
    dataset = FusionDataset(
        csv_path="data/metadata.csv",
        spec_root="data/spectrograms"
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    model = FusionModel().to(device)
    model.load_state_dict(
        torch.load("models/fusion_pipeline/checkpoints/best_model.pt")
    )
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for spec, input_ids, attention_mask, labels in loader:
            spec = spec.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            _, emb = model(spec, input_ids, attention_mask)

            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_embeddings), np.hstack(all_labels)


if __name__ == "__main__":

    os.makedirs("Results/plots", exist_ok=True)

    print("Extracting Speech embeddings...")
    speech_emb, speech_labels = extract_speech_embeddings()
    plot_tsne(
        speech_emb,
        speech_labels,
        "Speech Embedding t-SNE",
        "Results/plots/speech_tsne.png"
    )

    print("Extracting Text embeddings...")
    text_emb, text_labels = extract_text_embeddings()
    plot_tsne(
        text_emb,
        text_labels,
        "Text Embedding t-SNE",
        "Results/plots/text_tsne.png"
    )

    print("Extracting Fusion embeddings...")
    fusion_emb, fusion_labels = extract_fusion_embeddings()
    plot_tsne(
        fusion_emb,
        fusion_labels,
        "Fusion Embedding t-SNE",
        "Results/plots/fusion_tsne.png"
    )

    print("All t-SNE plots saved in Results/plots/")
