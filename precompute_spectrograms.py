import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

csv_path = "data/metadata.csv"
audio_root = "data/TESS"
save_root = "data/spectrograms"

os.makedirs(save_root, exist_ok=True)

df = pd.read_csv(csv_path)

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)

db = torchaudio.transforms.AmplitudeToDB()

max_len = 128

for idx, row in tqdm(df.iterrows(), total=len(df)):
    audio_path = os.path.join(audio_root, row["file_path"])
    save_path = os.path.join(save_root, row["file_path"] + ".pt")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    spec = mel(waveform)
    spec = db(spec)

    if spec.shape[-1] > max_len:
        spec = spec[:, :, :max_len]
    else:
        pad = max_len - spec.shape[-1]
        spec = torch.nn.functional.pad(spec, (0, pad))

    torch.save(spec, save_path)

print("All spectrograms saved.")