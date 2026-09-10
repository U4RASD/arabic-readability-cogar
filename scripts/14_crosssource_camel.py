"""
CAMeL readability classifier on the unlabelled cross-source corpus.

The released arabertv02-word-CE classifier predicts on the 19-level BAREC
scale. Predictions are mapped to the five levels of this work with the
thresholds at BAREC levels 10, 12, 14 and 16, which correspond to levels 1
to 4 of the evaluation set.

Inputs
    data/crosssource_chunks.csv       columns: uid, source, text

Outputs
    outputs/crosssource/camel_predictions.csv   uid, source, camel_arabertv02

Usage
    python scripts/14_crosssource_camel.py
"""
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

CORPUS_FILE = "data/crosssource_chunks.csv"
OUT_DIR = "outputs/crosssource"
OUT_FILE = os.path.join(OUT_DIR, "camel_predictions.csv")
MODEL = "CAMeL-Lab/readability-arabertv02-word-CE"
LEVEL_THRESHOLDS = [10, 12, 14, 16]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 0 if torch.cuda.is_available() else -1
    print("device:", "GPU" if device == 0 else "CPU")

    df = pd.read_csv(CORPUS_FILE)
    texts = df["text"].astype(str).tolist()

    clf = pipeline("text-classification", model=MODEL, device=device,
                   truncation=True, max_length=512, batch_size=32)

    levels_19 = []
    for i in tqdm(range(0, len(texts), 200), desc="camel"):
        out = clf(texts[i:i + 200])
        levels_19.extend(int(o["label"].split("_")[-1]) + 1 for o in out)

    df["camel_arabertv02"] = np.digitize(levels_19, bins=LEVEL_THRESHOLDS).astype(int)
    df[["uid", "source", "camel_arabertv02"]].to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"written: {OUT_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
