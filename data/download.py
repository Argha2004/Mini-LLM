from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

dataset = load_dataset(
    "DKYoon/SlimPajama-6B",
    split="train",
    streaming=True
)

with open("data/raw_train.txt", "w", encoding="utf-8") as f:
    for i, sample in enumerate(dataset):
        f.write(sample["text"].replace("\n", " ") + "\n")
        if i >= 1_000_000:
            break