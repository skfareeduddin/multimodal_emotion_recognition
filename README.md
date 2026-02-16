# Multimodal Emotion Classification

Multimodal emotion recognition system using **Speech and Text** modalities on the Toronto Emotional Speech Set (TESS).  
This project compares **Speech-only**, **Text-only**, and **Multimodal Fusion** approaches and analyzes learned representations.

---

## Objective

The goal of this project is to:

- Learn emotion representations from speech signals.
- Learn contextual representations from text transcripts.
- Combine both modalities into a unified representation.
- Compare unimodal and multimodal performance.
- Analyze representation separability using t-SNE.

---

## Dataset

**Toronto Emotional Speech Set (TESS)**

- 2800 audio samples
- 7 emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Pleasant Surprise
  - Sad
- Balanced dataset (400 samples per class)

All utterances follow the carrier phrase:

> “Say the word ___”

Emotion is primarily encoded through **acoustic-prosodic features**, not lexical content.

---

## Project Structure
```
multimodal_emotion_classification/
│
├── data/
│ ├── TESS/
│ └── spectrograms/
│
├── models/
│ ├── speech_pipeline/
│ │ ├── dataset.py
│ │ ├── model.py
│ │ ├── train.py
│ │ └── test.py
│ │
│ ├── text_pipeline/
│ │ ├── dataset.py
│ │ ├── model.py
│ │ ├── train.py
│ │ └── test.py
│ │
│ └── fusion_pipeline/
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ └── test.py
│
├── Results/
│ ├── accuracy_tables/
│ └── plots/
│
├── prepare_metadata.py
├── precompute_spectrograms.py
├── visualize_embeddings.py
├── requirements.txt
└── README.md
```

---

## Methodology

### 1️. Speech Pipeline

- Preprocessing:
  - Resampling (16 kHz)
  - Log-Mel Spectrogram (64 mel bins)
  - Fixed-length padding
- Temporal Modelling:
  - Convolutional Neural Network (CNN)
  - 128-dimensional embedding
- Classification:
  - Fully connected layer

### 2️. Text Pipeline

- Tokenization using `bert-base-uncased`
- Contextual modelling using BERT (CLS embedding – 768-d)
- Linear classification layer

### 3️. Fusion Pipeline

- Speech embedding (128-d → 256-d projection)
- Text embedding (768-d → 256-d projection)
- Concatenation → 512-d unified representation
- MLP classifier

BERT is frozen during fusion training for stability and efficiency.

---

## Results

| Model   | Accuracy | Macro F1 | Weighted F1 |
|----------|----------|----------|-------------|
| Speech   | 0.8871   | 0.8797   | 0.8797      |
| Text     | 0.1429   | 0.0357   | 0.0357      |
| Fusion   | 0.9814   | 0.9813   | 0.9813      |

### Key Observations

- Speech-only model performs strongly.
- Text-only model performs at chance level.
- Fusion achieves the highest performance.
- Emotion in TESS is primarily acoustic.

---

## Representation Analysis

t-SNE visualization was applied to embeddings from:

- Temporal Modelling block
- Contextual Modelling block
- Fusion block

Findings:

- Speech embeddings show clear emotion clusters.
- Text embeddings show complete overlap.
- Fusion embeddings show the strongest separation.

This confirms that multimodal fusion improves representation quality.

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Metadata

```
python prepare_metadata.py
```

### 3. Precompute Spectrograms

```
python precompute_spectrograms.py
```

### 4. Train Models

Speech:
```
python models/speech_pipeline/train.py
```

Text:
```
python models/text_pipeline/train.py
```

Fusion:
```
python models/fusion_pipeline/train.py
```

### 5. Test Models
```
python models/speech_pipeline/test.py
python models/text_pipeline/test.py
python models/fusion_pipeline/test.py
```

### 6. Visualize Embeddings
```
python visualize_embeddings.py
```

---

## Conclusion

- Emotion in TESS is primarily encoded in acoustic features.
- Textual modality alone is insufficient for emotion recognition.
- Multimodal fusion enhances representation separability.
- The final system achieves 98.14% accuracy, demonstrating effective multimodal learning.

---

## Requirements
```
Python 3.10
PyTorch
torchaudio
transformers
scikit-learn
matplotlib
seaborn
pandas
```

See requirements.txt for full list.

---
