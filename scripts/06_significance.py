"""
Significance testing with a paired bootstrap over seeds and documents.

Two sources of variability affect a cross-validated score: which documents
form the evaluation set, and which random seed drives the fold assignment.
Both are taken into account. For each of the ten seeds, CoGAR out-of-fold
predictions are regenerated, then documents are resampled with replacement
B times. The same resampled indices are used for every system, so every
comparison is paired. Deterministic systems (LLM raters, formulas) do not
depend on the seed and are simply evaluated on the same draws.

For each pair of systems A and B the script reports the mean difference
QWK(A) - QWK(B), its 95% interval and a two-sided p-value computed as
2 * min(P(diff <= 0), P(diff >= 0)).

Inputs
    outputs/graph_features.csv          from 01_extract_features.py
    outputs/barec/predictions.csv       from 02_evaluate.py (LLM raters)
    outputs/barec/formula_levels.csv    from 04_formulas.py (oriented formulas)
    data/barec_5levels.parquet          gold labels

Outputs
    outputs/tables/significance.csv

Usage
    python scripts/06_significance.py

With N_SEEDS = 10 and B = 3000 the script takes a few minutes.
"""
import os
import sys
import time

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
FORMULA_CSV = "outputs/barec/formula_levels.csv"
OUT_DIR = "outputs/tables"
LABEL_COL = "difficulty_level"
N_SEEDS = 10
B = 3000
K = 5
RNG_SEED = 0

_W = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2 / (K - 1) ** 2


def qwk(yt, yp):
    O = np.bincount(yt * K + yp, minlength=K * K).reshape(K, K).astype(np.float64)
    tot = O.sum()
    if tot == 0:
        return np.nan
    E = np.outer(O.sum(1), O.sum(0)) / tot
    d = (_W * E).sum()
    return 1.0 - (_W * O).sum() / d if d > 0 else np.nan


def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", mord.LogisticAT()),
    ])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in (DATASET, FEATURES_CSV, PRED_CSV, FORMULA_CSV):
        if not os.path.exists(f):
            sys.exit(f"missing input: {f}")

    gold = pd.read_parquet(DATASET)[LABEL_COL].astype(int).to_numpy()
    feat = pd.read_csv(FEATURES_CSV)
    pred = pd.read_csv(PRED_CSV)
    formulas = pd.read_csv(FORMULA_CSV)
    n = len(gold)

    COOC = [c for c in feat.columns if c.startswith("cooc_")]
    COOC_TOPO = [c for c in COOC if c not in ("cooc_n_nodes", "cooc_n_edges")]
    ALL = ([c for c in feat.columns if c.startswith("lex_")] + COOC +
           [c for c in feat.columns if c.startswith("ner_")] +
           [c for c in feat.columns if c.startswith("rel_")])

    # ----------------------------------------------------------------------
    # CoGAR predictions for every seed
    # ----------------------------------------------------------------------
    print(f"regenerating CoGAR out-of-fold predictions for {N_SEEDS} seeds")
    t0 = time.time()
    preds = {"CoGAR topology": np.zeros((N_SEEDS, n), dtype=int),
             "CoGAR all features": np.zeros((N_SEEDS, n), dtype=int)}
    for s in range(N_SEEDS):
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        preds["CoGAR topology"][s] = cross_val_predict(make_pipe(), feat[COOC_TOPO].to_numpy(), gold, cv=cv)
        preds["CoGAR all features"][s] = cross_val_predict(make_pipe(), feat[ALL].to_numpy(), gold, cv=cv)
    print(f"  done in {time.time() - t0:.0f}s")
    for name, P in preds.items():
        q = np.array([qwk(gold, P[s]) for s in range(N_SEEDS)])
        print(f"  {name:20s} QWK = {q.mean():.3f} +/- {q.std():.3f}")

    # ----------------------------------------------------------------------
    # Deterministic systems
    # ----------------------------------------------------------------------
    det = {}
    for label, col in [("GPT-4.1", "llm_gpt-4.1"), ("Qwen-35B", "llm_qwen3.5-35b"),
                       ("Fanar", "llm_fanar-c-2-27b")]:
        if col in pred.columns:
            det[label] = pred[col].to_numpy(dtype=float)
    for label, col in [("OSMAN", "osman"), ("Flesch", "flesch"), ("AARI", "aari")]:
        det[label] = formulas[col].to_numpy(dtype=float)
    print(f"deterministic systems: {list(det)}")

    # ----------------------------------------------------------------------
    # Shared bootstrap draws
    # ----------------------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    idx_boot = rng.integers(0, n, size=(B, n))

    def diff_distribution(pa, pb, seeds_a, seeds_b):
        out = []
        for s in range(N_SEEDS if (seeds_a or seeds_b) else 1):
            a = pa[s].astype(float) if seeds_a else pa
            b = pb[s].astype(float) if seeds_b else pb
            ok = ~np.isnan(a) & ~np.isnan(b)
            ai = np.where(ok, a, 0).astype(int)
            bi = np.where(ok, b, 0).astype(int)
            for k in range(B):
                idx = idx_boot[k]
                m = ok[idx]
                if m.sum() < 20:
                    continue
                yt = gold[idx][m]
                va, vb = qwk(yt, ai[idx][m]), qwk(yt, bi[idx][m])
                if va == va and vb == vb:
                    out.append(va - vb)
        return np.asarray(out)

    def summarize(d, label):
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        sig = lo > 0 or hi < 0
        print(f"  {label:38s} diff {d.mean():+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  "
              f"p = {p:.4f}{'  *' if sig else ''}")
        return {"comparison": label, "mean_diff": round(float(d.mean()), 3),
                "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
                "p_value": round(float(p), 4), "significant_95": bool(sig),
                "n_draws": int(len(d))}

    rows = []
    topo, allf = preds["CoGAR topology"], preds["CoGAR all features"]

    print("\nGPT-4.1 against the two CoGAR configurations")
    rows.append(summarize(diff_distribution(det["GPT-4.1"], topo, False, True), "GPT-4.1 - CoGAR topology"))
    rows.append(summarize(diff_distribution(det["GPT-4.1"], allf, False, True), "GPT-4.1 - CoGAR all features"))

    print("\nThe two CoGAR configurations")
    rows.append(summarize(diff_distribution(topo, allf, True, True), "CoGAR topology - CoGAR all features"))

    print("\nCoGAR topology against the other systems")
    for label, p in det.items():
        if label == "GPT-4.1":
            continue
        rows.append(summarize(diff_distribution(topo, p, True, False), f"CoGAR topology - {label}"))

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "significance.csv"), index=False)
    print(f"\nwritten: {OUT_DIR}/significance.csv")


if __name__ == "__main__":
    main()
