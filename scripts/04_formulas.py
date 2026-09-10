"""
Readability formulas on the evaluation set, oriented towards difficulty.

Flesch and OSMAN are reading-ease scores: a higher raw value means an easier
text. The evaluation harness (02_evaluate.py) rank-bins them into five
equal-frequency groups in the direction of the raw score, so its bins run
from hard to easy. This script reverses them so that, like the gold levels,
higher means harder. AARI is a grade-level index and already increases with
difficulty, so its bins are computed here from the raw score and kept as
they are.

Produces
    Table 1, formula rows : OSMAN, Flesch and AARI with all metrics

Inputs
    outputs/barec/predictions.csv   from 02_evaluate.py (columns osman_only,
                                    flesch_only)
    outputs/graph_features.csv      from 01_extract_features.py (column aari)
    data/barec_5levels.parquet      gold labels

Outputs
    outputs/tables/table1_formula_rows.csv
    outputs/barec/formula_levels.csv   oriented levels per document, used
                                       downstream

Usage
    python scripts/04_formulas.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATASET = "data/barec_5levels.parquet"
PRED_CSV = "outputs/barec/predictions.csv"
FEATURES_CSV = "outputs/graph_features.csv"
OUT_DIR = "outputs/tables"
LEVELS_CSV = "outputs/barec/formula_levels.csv"
LABEL_COL = "difficulty_level"
K = 5


def metrics(yt, yp):
    m = ~pd.isna(yp)
    yt2 = np.asarray(yt)[m].astype(int)
    yp2 = np.asarray(yp)[m].astype(int)
    diff = np.abs(yp2 - yt2)
    return {
        "n": int(m.sum()),
        "exact": round(float(np.mean(diff == 0)), 3),
        "within_1": round(float(np.mean(diff <= 1)), 3),
        "within_2": round(float(np.mean(diff <= 2)), 3),
        "mae": round(float(np.mean(diff)), 3),
        "spearman": round(float(spearmanr(yp2, yt2).statistic), 3),
        "qwk": round(float(cohen_kappa_score(yt2, np.clip(yp2, 0, K - 1),
                                              weights="quadratic",
                                              labels=list(range(K)))), 3),
    }


def reverse_levels(levels, k=K):
    """Map level x to (k - 1) - x, keeping missing values."""
    v = np.asarray(levels, dtype=float)
    out = np.full(len(v), np.nan)
    m = ~np.isnan(v)
    out[m] = (k - 1) - v[m]
    return out


def rank_bin(scores, k=K):
    """Equal-frequency rank binning, higher score to higher level."""
    s = pd.Series(np.asarray(scores, dtype=float))
    bins = pd.qcut(s.rank(method="first"), q=k, labels=False, duplicates="drop")
    return bins.to_numpy(dtype=float)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in (PRED_CSV, FEATURES_CSV, DATASET):
        if not os.path.exists(f):
            sys.exit(f"missing input: {f}")

    pred = pd.read_csv(PRED_CSV)
    feat = pd.read_csv(FEATURES_CSV)
    gold = pd.read_parquet(DATASET)[LABEL_COL].astype(int).to_numpy()

    # Flesch and OSMAN come from the harness with ease-oriented bins.
    osman = reverse_levels(pred["osman_only"].to_numpy(dtype=float))
    flesch = reverse_levels(pred["flesch_only"].to_numpy(dtype=float))
    # AARI is binned here from its raw score, already difficulty-oriented.
    aari = rank_bin(feat["aari"].to_numpy(dtype=float))

    rows = []
    for label, lv in [("OSMAN", osman), ("Flesch", flesch), ("AARI", aari)]:
        m = metrics(gold, lv)
        rows.append({"method": label, **m})
        print(f"  {label:8s} exact {m['exact']:.3f}  within-1 {m['within_1']:.3f}  "
              f"within-2 {m['within_2']:.3f}  MAE {m['mae']:.3f}  "
              f"Spearman {m['spearman']:.3f}  QWK {m['qwk']:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "table1_formula_rows.csv"), index=False)
    pd.DataFrame({"id": pred["id"], "gold": gold,
                  "osman": osman, "flesch": flesch, "aari": aari}).to_csv(LEVELS_CSV, index=False)
    print(f"\nwritten: {OUT_DIR}/table1_formula_rows.csv")
    print(f"written: {LEVELS_CSV}")


if __name__ == "__main__":
    main()
