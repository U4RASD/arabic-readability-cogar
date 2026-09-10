"""
CAMeL readability classifiers on the evaluation set.

Two publicly released classifiers fine-tuned on BAREC are applied to the
1,000 paragraphs as supervised baselines:

    CAMeL-Lab/readability-arabertv02-word-CE   raw text input, classification
    CAMeL-Lab/readability-arabertv2-d3tok-reg  D3Tok morphological tokens,
                                               regression head (strongest
                                               released variant)

Both models predict on the 19-level BAREC scale. Predictions are mapped to
the five levels of this work through the official 19-to-7 BAREC grouping,
then 7-scale levels 1 to 3 map to level 0 and levels 4 to 7 map to levels 1
to 4. The official grouping is read from the BAREC release files.

Both models were trained on BAREC, from which levels 1 to 4 of this set are
drawn, so their scores reflect an in-domain advantage. The paper states this.

Produces
    Table 1, CAMeL rows

Inputs
    data/barec_5levels.parquet
    data/barec_release/{train,dev,test}.csv   official BAREC sentence files,
                                              needed for the 19-to-7 mapping

Outputs
    outputs/tables/table1_camel_rows.csv
    outputs/barec/camel_predictions.csv       19-level and 5-level predictions
    outputs/barec/d3tok_inputs.csv            cached morphological tokens

Usage
    python scripts/05_camel_baselines.py

Requires torch, transformers and camel-tools. The D3Tok tokenisation of
1,000 paragraphs takes a few minutes and is cached.
"""
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATASET = "data/barec_5levels.parquet"
BAREC_RELEASE_DIR = "data/barec_release"
OUT_DIR = "outputs/tables"
PRED_DIR = "outputs/barec"
LABEL_COL = "difficulty_level"
BATCH = 16
K = 5

MODELS = [
    ("CAMeL arabertv02-word-CE", "CAMeL-Lab/readability-arabertv02-word-CE", "raw"),
    ("CAMeL arabertv2-d3tok-reg", "CAMeL-Lab/readability-arabertv2-d3tok-reg", "d3tok"),
]


def metrics(yt, yp):
    yt = np.asarray(yt).astype(int)
    yp = np.asarray(yp).astype(int)
    diff = np.abs(yp - yt)
    return {
        "exact": round(float(np.mean(diff == 0)), 3),
        "within_1": round(float(np.mean(diff <= 1)), 3),
        "within_2": round(float(np.mean(diff <= 2)), 3),
        "mae": round(float(np.mean(diff)), 3),
        "spearman": round(float(spearmanr(yp, yt).statistic), 3),
        "qwk": round(float(cohen_kappa_score(yt, yp, weights="quadratic",
                                              labels=list(range(K)))), 3),
    }


def load_level_mapping():
    """Official BAREC 19-to-7 grouping, then 7 to the five levels used here."""
    frames = []
    for f in ("train.csv", "dev.csv", "test.csv"):
        p = os.path.join(BAREC_RELEASE_DIR, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        sys.exit(f"BAREC release files not found in {BAREC_RELEASE_DIR}. "
                 "Download train.csv, dev.csv and test.csv from the BAREC release.")
    barec = pd.concat(frames, ignore_index=True)
    map19to7 = (barec.groupby("Readability_Level_19")["Readability_Level_7"]
                .agg(lambda s: int(s.mode().iloc[0])).to_dict())
    seven_to_ours = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4}
    map19 = {k: seven_to_ours[v] for k, v in map19to7.items()}
    assert map19[10] == 1 and map19[12] == 2 and map19[14] == 3 and map19[16] == 4
    return map19


def d3tok_inputs(texts):
    """Morphological D3Tok tokenisation with CAMeL Tools, cached on disk."""
    cache = os.path.join(PRED_DIR, "d3tok_inputs.csv")
    if os.path.exists(cache):
        cached = pd.read_csv(cache)["d3tok"].astype(str).tolist()
        if len(cached) == len(texts):
            print("  D3Tok inputs loaded from cache")
            return cached
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer
    print("  D3Tok tokenisation, this takes a few minutes")
    tk = MorphologicalTokenizer(disambiguator=MLEDisambiguator.pretrained(),
                                scheme="d3tok", split=True, diac=False)
    out = []
    t0 = time.time()
    for i, t in enumerate(texts):
        out.append(" ".join(tk.tokenize(str(t).split())))
        if i % 100 == 0:
            print(f"    {i}/{len(texts)} ({time.time() - t0:.0f}s)")
    pd.DataFrame({"d3tok": out}).to_csv(cache, index=False)
    return out


def predict_19(model_id, inputs):
    """Run a CAMeL classifier and return predictions on the 19-level scale."""
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    is_regression = mdl.config.num_labels == 1
    out = []
    with torch.no_grad():
        for i in range(0, len(inputs), BATCH):
            enc = tok(inputs[i:i + BATCH], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            logits = mdl(**enc).logits
            if is_regression:
                out.extend(np.clip(np.rint(logits.squeeze(-1).numpy()), 1, 19).astype(int).tolist())
            else:
                out.extend((logits.argmax(-1).numpy() + 1).tolist())
    return np.asarray(out, dtype=int)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    ds = pd.read_parquet(DATASET)
    gold = ds[LABEL_COL].astype(int).to_numpy()
    texts = ds["text"].astype(str).tolist()
    map19 = load_level_mapping()
    print(f"dataset: {len(ds)} paragraphs")

    rows, preds = [], {"id": ds["ID"], "gold": gold}
    for label, model_id, input_kind in MODELS:
        print(f"\n{label}")
        inputs = d3tok_inputs(texts) if input_kind == "d3tok" else texts
        p19 = predict_19(model_id, inputs)
        p5 = np.array([map19.get(int(v), 4) for v in p19])
        m = metrics(gold, p5)
        rows.append({"method": label, **m})
        preds[label.replace(" ", "_") + "_19"] = p19
        preds[label.replace(" ", "_") + "_5"] = p5
        print(f"  exact {m['exact']:.3f}  within-1 {m['within_1']:.3f}  "
              f"within-2 {m['within_2']:.3f}  MAE {m['mae']:.3f}  "
              f"Spearman {m['spearman']:.3f}  QWK {m['qwk']:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "table1_camel_rows.csv"), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(PRED_DIR, "camel_predictions.csv"), index=False)
    print(f"\nwritten: {OUT_DIR}/table1_camel_rows.csv")
    print(f"written: {PRED_DIR}/camel_predictions.csv")


if __name__ == "__main__":
    main()
