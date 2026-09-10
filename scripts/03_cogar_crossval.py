"""
CoGAR cross-validation: main-table rows and feature-family ablation.

Every configuration is trained and evaluated with the same ordinal regressor
(LogisticAT) under stratified five-fold cross-validation, repeated over ten
random seeds. Every paragraph receives its prediction from a fold in which it
was not used for training. Results are reported as mean and standard
deviation over seeds.

Produces
    Table 1, CoGAR rows   : topology and all-feature configurations, with
                            exact, within-1, within-2, MAE, Spearman and QWK
    Table 2, ablation     : one row per feature family

Inputs
    outputs/graph_features.csv      from 01_extract_features.py
    data/barec_5levels.parquet      gold labels

Outputs
    outputs/tables/table1_cogar_rows.csv
    outputs/tables/table2_ablation.csv

Usage
    python scripts/03_cogar_crossval.py
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    import mord
except ImportError:
    sys.exit("mord is required (pip install mord). The paper uses mord.LogisticAT.")

from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATASET = "data/barec_5levels.parquet"
FEATURES_CSV = "outputs/graph_features.csv"
OUT_DIR = "outputs/tables"
LABEL_COL = "difficulty_level"
N_SEEDS = 10
N_FOLDS = 5
K = 5


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
_W = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2 / (K - 1) ** 2


def qwk(yt, yp):
    a = np.clip(np.asarray(yt).astype(int), 0, K - 1)
    b = np.clip(np.asarray(yp).astype(int), 0, K - 1)
    O = np.bincount(a * K + b, minlength=K * K).reshape(K, K).astype(float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    d = (_W * E).sum()
    return 1 - (_W * O).sum() / d if d > 0 else np.nan


def all_metrics(yt, yp):
    yt = np.asarray(yt).astype(int)
    yp = np.asarray(yp).astype(int)
    diff = np.abs(yp - yt)
    return {
        "exact": float(np.mean(diff == 0)),
        "within_1": float(np.mean(diff <= 1)),
        "within_2": float(np.mean(diff <= 2)),
        "mae": float(np.mean(diff)),
        "spearman": float(spearmanr(yp, yt).statistic),
        "qwk": float(qwk(yt, yp)),
    }


def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", mord.LogisticAT()),
    ])


def oof_predict(X, y, seed):
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    return cross_val_predict(make_pipe(), X, y, cv=cv)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    feat = pd.read_csv(FEATURES_CSV)
    gold = pd.read_parquet(DATASET)[LABEL_COL].astype(int).to_numpy()
    if len(feat) != len(gold):
        sys.exit(f"feature rows ({len(feat)}) and gold labels ({len(gold)}) differ")

    LEX = [c for c in feat.columns if c.startswith("lex_")]
    COOC = [c for c in feat.columns if c.startswith("cooc_")]
    NER = [c for c in feat.columns if c.startswith("ner_")]
    REL = [c for c in feat.columns if c.startswith("rel_")]
    COOC_VOL = [c for c in ("cooc_n_nodes", "cooc_n_edges") if c in COOC]
    COOC_TOPO = [c for c in COOC if c not in COOC_VOL]
    ALL = LEX + COOC + NER + REL
    print(f"features: lexical {len(LEX)}, co-occurrence {len(COOC)} "
          f"({len(COOC_VOL)} volume, {len(COOC_TOPO)} topology), "
          f"named entities {len(NER)}, entity-relation {len(REL)}, total {len(ALL)}")

    aari = feat["aari"].to_numpy(dtype=float).reshape(-1, 1)

    # ----------------------------------------------------------------------
    # Table 2: ablation, QWK mean and SD over seeds
    # ----------------------------------------------------------------------
    print(f"\nablation, {N_SEEDS} seeds x {N_FOLDS} folds per configuration")
    configs = [
        ("Entity-relation graph", feat[REL].to_numpy()),
        ("Named entities (NER)", feat[NER].to_numpy()),
        ("AARI (length only)", aari),
        ("Co-occurrence volume (size)", feat[COOC_VOL].to_numpy()),
        ("Lexical (morphology, rarity, length)", feat[LEX].to_numpy()),
        ("Graph (cooc+NER+rel)", feat[COOC + NER + REL].to_numpy()),
        ("All features", feat[ALL].to_numpy()),
        ("AARI + all features", np.hstack([feat[ALL].to_numpy(), aari])),
        ("Co-occurrence graph (vol+topo)", feat[COOC].to_numpy()),
        ("Co-occurrence topology", feat[COOC_TOPO].to_numpy()),
    ]
    rows = []
    for label, X in configs:
        scores = [qwk(gold, oof_predict(X, gold, s)) for s in range(N_SEEDS)]
        m, sd = float(np.mean(scores)), float(np.std(scores))
        rows.append({"configuration": label, "n_features": X.shape[1],
                     "qwk_mean": round(m, 3), "qwk_sd": round(sd, 3)})
        print(f"  {label:38s} {X.shape[1]:3d} features   QWK = {m:.3f} +/- {sd:.3f}")
    t2 = pd.DataFrame(rows).sort_values("qwk_mean")
    t2.to_csv(os.path.join(OUT_DIR, "table2_ablation.csv"), index=False)

    # ----------------------------------------------------------------------
    # Table 1: full metrics for the two CoGAR configurations
    # ----------------------------------------------------------------------
    print(f"\nmain-table rows, all metrics, {N_SEEDS} seeds")
    rows = []
    for label, cols in [("CoGAR (topology)", COOC_TOPO), ("CoGAR (all features)", ALL)]:
        X = feat[cols].to_numpy()
        per_seed = pd.DataFrame([all_metrics(gold, oof_predict(X, gold, s)) for s in range(N_SEEDS)])
        r = {"method": label}
        for c in per_seed.columns:
            r[c] = round(float(per_seed[c].mean()), 3)
            r[c + "_sd"] = round(float(per_seed[c].std()), 3)
        rows.append(r)
        print(f"  {label:22s} exact {r['exact']:.3f}  within-1 {r['within_1']:.3f}  "
              f"within-2 {r['within_2']:.3f}  MAE {r['mae']:.3f}  "
              f"Spearman {r['spearman']:.3f}  QWK {r['qwk']:.3f} +/- {r['qwk_sd']:.3f}")
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(OUT_DIR, "table1_cogar_rows.csv"), index=False)

    print(f"\nwritten: {OUT_DIR}/table2_ablation.csv")
    print(f"written: {OUT_DIR}/table1_cogar_rows.csv")


if __name__ == "__main__":
    main()
