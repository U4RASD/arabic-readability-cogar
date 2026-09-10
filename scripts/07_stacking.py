"""
Combining an LLM rater with the co-occurrence graph.

Two combinations are evaluated. Fusion averages the LLM rating and the
out-of-fold CoGAR prediction. Stacking trains an ordinal meta-learner
(LogisticAT) on the LLM rating and the topology features, evaluated strictly
out-of-fold under stratified five-fold cross-validation and repeated over
five seeds.

Leakage controls. The LLM is used zero-shot and never trained on this data.
The topology inputs are deterministic functions of the text. The
meta-learner has no tuned hyper-parameters, so a single cross-validation
level is unbiased. Missing LLM ratings (Qwen answers about 95% of the
documents) are imputed by the median inside the pipeline.

Produces
    Table 4

Inputs
    outputs/graph_features.csv        from 01_extract_features.py
    outputs/barec/predictions.csv     from 02_evaluate.py
    data/barec_5levels.parquet        gold labels

Outputs
    outputs/tables/table4_stacking.csv

Usage
    python scripts/07_stacking.py
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

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATASET = "data/barec_5levels.parquet"
FEATURES_CSV = "outputs/graph_features.csv"
PRED_CSV = "outputs/barec/predictions.csv"
OUT_DIR = "outputs/tables"
LABEL_COL = "difficulty_level"
GPT_COL = "llm_gpt-4.1"
QWEN_COL = "llm_qwen3.5-35b"
N_SEEDS = 5
K = 5

_W = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2 / (K - 1) ** 2


def qwk(yt, yp):
    yt = np.asarray(yt).astype(int)
    yp = np.clip(np.asarray(yp).astype(int), 0, K - 1)
    O = np.bincount(yt * K + yp, minlength=K * K).reshape(K, K).astype(float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (_W * O).sum() / (_W * E).sum()


def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", mord.LogisticAT()),
    ])


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(make_pipe(), X, y, cv=cv)


def repeated(X, y):
    scores = [qwk(y, oof(X, y, s)) for s in range(N_SEEDS)]
    return float(np.mean(scores)), float(np.std(scores))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in (DATASET, FEATURES_CSV, PRED_CSV):
        if not os.path.exists(f):
            sys.exit(f"missing input: {f}")

    gold = pd.read_parquet(DATASET)[LABEL_COL].astype(int).to_numpy()
    feat = pd.read_csv(FEATURES_CSV)
    pred = pd.read_csv(PRED_CSV)

    COOC = [c for c in feat.columns if c.startswith("cooc_")]
    TOPO = [c for c in COOC if c not in ("cooc_n_nodes", "cooc_n_edges")]
    X_topo = feat[TOPO].to_numpy()

    gpt = pred[GPT_COL].to_numpy(dtype=float)
    qwen = pred[QWEN_COL].to_numpy(dtype=float)
    qwen_ok = ~np.isnan(qwen)
    print(f"documents: {len(gold)}  Qwen coverage: {qwen_ok.mean():.3f}")

    # CoGAR topology alone, out-of-fold, seed 42 as in the main pipeline
    topo_pred = oof(X_topo, gold, 42).astype(float)

    rows = []

    def add(system, m, sd=None):
        rows.append({"system": system, "qwk": round(m, 3),
                     "qwk_sd": None if sd is None else round(sd, 3)})
        print(f"  {system:40s} QWK = {m:.3f}" + (f" +/- {sd:.3f}" if sd is not None else ""))

    print("\ncomponents alone")
    add("CoGAR topology (alone)", qwk(gold, topo_pred))
    add("GPT-4.1 (alone)", qwk(gold, gpt))
    add("Qwen-35B (alone)", qwk(gold[qwen_ok], qwen[qwen_ok]))

    print("\nfusion, average of the two ratings rounded to the nearest level")
    fusion = np.rint((gpt + topo_pred) / 2.0)
    add("Fusion (averaging GPT-4.1 and topology)", qwk(gold, fusion))

    print(f"\nstacking, ordinal meta-learner, {N_SEEDS} seeds x 5 folds")
    m, sd = repeated(np.column_stack([gpt, X_topo]), gold)
    add("Stacking (GPT-4.1 + topology features)", m, sd)

    m, sd = repeated(np.column_stack([gpt, topo_pred]), gold)
    add("Stacking (GPT-4.1 + topology prediction only)", m, sd)

    m, sd = repeated(np.column_stack([qwen, X_topo]), gold)
    add("Stacking (Qwen-35B + topology features)", m, sd)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "table4_stacking.csv"), index=False)
    print(f"\nwritten: {OUT_DIR}/table4_stacking.csv")


if __name__ == "__main__":
    main()
