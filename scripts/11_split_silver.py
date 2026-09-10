"""
Stratified split of the silver-labelled corpus into train, validation and
test sets for distillation.

Gemini labels on the 1 to 5 scale are mapped to 0 to 4. Rows without a label
or without text are dropped. The split is stratified by level so that every
subset keeps the same level proportions.

Inputs
    outputs/silver/silver_chunks_labeled.csv   from 10_label_with_gemini.py

Outputs
    outputs/silver/train.csv, val.csv, test.csv   columns: text, label, source

Usage
    python scripts/11_split_silver.py
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

IN_CSV = "outputs/silver/silver_chunks_labeled.csv"
OUT_DIR = "outputs/silver"
TEXT_COL = "text"
LABEL_COL = "level"
SEED = 42
TEST_FRAC = 0.10
VAL_FRAC = 0.10


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    print(f"read {len(df)} rows")

    before = len(df)
    df = df.dropna(subset=[LABEL_COL, TEXT_COL])
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]
    print(f"dropped {before - len(df)} rows without label or text")

    df[LABEL_COL] = df[LABEL_COL].astype(float).round().astype(int)
    lo, hi = df[LABEL_COL].min(), df[LABEL_COL].max()
    if hi >= 5 or lo >= 1:
        df["label"] = (df[LABEL_COL] - 1).clip(0, 4)
        print("labels mapped from 1-5 to 0-4")
    else:
        df["label"] = df[LABEL_COL].clip(0, 4)
    print("level distribution:", df["label"].value_counts().sort_index().to_dict())

    keep = [TEXT_COL, "label"] + (["source"] if "source" in df.columns else [])
    df = df[keep].reset_index(drop=True)

    train_val, test = train_test_split(df, test_size=TEST_FRAC, random_state=SEED, stratify=df["label"])
    val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
    train, val = train_test_split(train_val, test_size=val_rel, random_state=SEED,
                                  stratify=train_val["label"])

    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        part.to_csv(path, index=False, encoding="utf-8")
        print(f"  {name:5s} {len(part):6d} rows  ->  {path}")


if __name__ == "__main__":
    main()
