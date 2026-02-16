import pandas as pd

speech = pd.read_csv("Results/accuracy_tables/speech_metrics.csv")
text = pd.read_csv("Results/accuracy_tables/text_metrics.csv")
fusion = pd.read_csv("Results/accuracy_tables/fusion_metrics.csv")

combined = pd.concat([speech, text, fusion], ignore_index=True)

combined.to_csv("Results/accuracy_tables/combined_metrics.csv", index=False)

print(combined)