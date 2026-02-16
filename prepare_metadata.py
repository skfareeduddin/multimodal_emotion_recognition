import os
import pandas as pd

data_root = "data/TESS"

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "pleasant_surprise": 5,
    "sad": 6
}

rows = []

for folder in os.listdir(data_root):
    folder_path = os.path.join(data_root, folder)

    if os.path.isdir(folder_path):

        emotion = "_".join(folder.split("_")[1:]).lower()

        if emotion not in emotion_map:
            print(f"Skipping unknown emotion folder: {folder}")
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):

                parts = file.replace(".wav", "").split("_")
                word = parts[1]

                transcript = f"Say the word {word}"

                rows.append({
                    "file_path": os.path.join(folder, file),
                    "text": transcript,
                    "label": emotion_map[emotion]
                })

df = pd.DataFrame(rows)

df.to_csv("data/metadata.csv", index=False)

print("Metadata created successfully.")
print("Total samples:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts().sort_index())