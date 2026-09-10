"""
Evaluation restricted to the four BAREC levels.

Level 0 comes from an external source and is trivially separable. This
script reports every learned method on levels 1 to 4 alone. CoGAR is
retrained on those levels only, with the same ordinal regressor and
cross-validation as the main pipeline, over ten seeds. QWK is computed on
the four-level scale, and within-1 accuracy is reported alongside it.

Inputs
    outputs/graph_features.csv        from 01_extract_features.py
    outputs/barec/predictions.csv     from 02_evaluate.py
    data/barec_5levels.parquet        gold labels

Outputs
    outputs/tables/level0_excluded.csv

Usage
    python scripts/09_level0_excluded.py
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    import mord
except ImportError:
    sys.exit("mord is required (pip install mord).")

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET = "data/barec_5levels.parquet"
FEATURES_CSV = "outputs/graph_features.csv"
PRED_CSV = "outputs/barec/predictions.csv"
OUT_DIR = "outputs/tables"
LABEL_COL = "difficulty_level"
N_SEEDS = 10

LLM_COLS = {"llm_gpt-4.1": "GPT-4.1", "llm_qwen3.5-35b": "Qwen-35B", "llm_fanar-c-2-27b": "Fanar"}


def make_qwk(k):
    W = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2

    def f(yt, yp, offset=0):
        a = np.asarray(yt).astype(int) - offset
        b = np.clip(np.asarray(yp).astype(int) - offset, 0, k - 1)
        O = np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype(float)
        if O.sum() == 0:
            return np.nan
        E = np.outer(O.sum(1), O.sum(0)) / O.sum()
        d = (W * E).sum()
        return 1 - (W * O).sum() / d if d > 0 else np.nan
    return f


qwk5, qwk4 = make_qwk(5), make_qwk(4)


def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", mord.LogisticAT()),
    ])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in (DATASET, FEATURES_CSV, PRED_CSV):
        if not os.path.exists(f):
            sys.exit(f"missing input: {f}")

    gold = pd.read_parquet(DATASET)[LABEL_COL].astype(int).to_numpy()
    feat = pd.read_csv(FEATURES_CSV)
    pred = pd.read_csv(PRED_CSV)
    mask = gold >= 1
    gold14 = gold[mask]
    print(f"documents kept: {mask.sum()} of {len(gold)}\n")

    rows = []
    for col, label in LLM_COLS.items():
        if col not in pred.columns:
            continue
        v = pred[col].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        m = ok & mask
        q5 = qwk5(gold[ok], v[ok])
        q4 = qwk4(gold[m], np.clip(v[m], 1, 4), offset=1)
        w1 = float(np.mean(np.abs(v[m] - gold[m]) <= 1))
        rows.append({"method": label, "qwk_5_levels": round(float(q5), 3),
                     "qwk_levels_1_4": round(float(q4), 3), "within_1_levels_1_4": round(w1, 3)})
        print(f"  {label:22s} five levels {q5:.3f}   levels 1-4 {q4:.3f}   within-1 {w1:.3f}")

    COOC = [c for c in feat.columns if c.startswith("cooc_")]
    TOPO = [c for c in COOC if c not in ("cooc_n_nodes", "cooc_n_edges")]
    ALL = ([c for c in feat.columns if c.startswith("lex_")] + COOC +
           [c for c in feat.columns if c.startswith("ner_")] +
           [c for c in feat.columns if c.startswith("rel_")])

    for label, cols in [("CoGAR (topology)", TOPO), ("CoGAR (all features)", ALL)]:
        X5, X4 = feat[cols].to_numpy(), feat.loc[mask, cols].to_numpy()
        q5s, q4s, w1s = [], [], []
        for s in range(N_SEEDS):
            cv = StratifiedKFold(5, shuffle=True, random_state=s)
            q5s.append(qwk5(gold, cross_val_predict(make_pipe(), X5, gold, cv=cv)))
            yp = cross_val_predict(make_pipe(), X4, gold14, cv=cv)
            q4s.append(qwk4(gold14, yp, offset=1))
            w1s.append(float(np.mean(np.abs(yp - gold14) <= 1)))
        rows.append({"method": label, "qwk_5_levels": round(float(np.mean(q5s)), 3),
                     "qwk_levels_1_4": round(float(np.mean(q4s)), 3),
                     "within_1_levels_1_4": round(float(np.mean(w1s)), 3)})
        print(f"  {label:22s} five levels {np.mean(q5s):.3f}   levels 1-4 {np.mean(q4s):.3f}   "
              f"within-1 {np.mean(w1s):.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "level0_excluded.csv"), index=False)
    print(f"\nwritten: {OUT_DIR}/level0_excluded.csv")


if __name__ == "__main__":
    main()
