"""
Sensitivity of the topology configuration to the co-occurrence window.

The co-occurrence graph links two content lemmas that appear within a window
of W consecutive content words. This script rebuilds the graph and its ten
topology features for W from 1 to 5, then evaluates each setting with the
same ordinal regressor and cross-validation as the main pipeline, over five
seeds. Only the window changes.

Inputs
    outputs/nlp_cache.json        from 01_extract_features.py (sentence lemmas)
    data/barec_5levels.parquet    gold labels

Outputs
    outputs/tables/window_sensitivity.csv

Usage
    python scripts/08_window_sensitivity.py
"""
import json
import os
import sys

import networkx as nx
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
NLP_CACHE = "outputs/nlp_cache.json"
OUT_DIR = "outputs/tables"
LABEL_COL = "difficulty_level"
WINDOWS = [1, 2, 3, 4, 5]
N_SEEDS = 5
K = 5

TOPO_KEYS = ["cooc_avg_degree", "cooc_max_degree", "cooc_density", "cooc_avg_clustering",
             "cooc_n_components", "cooc_largest_cc_frac", "cooc_avg_path", "cooc_diameter",
             "cooc_degree_entropy", "cooc_assortativity"]

_W = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2 / (K - 1) ** 2


def _entropy(counts):
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def build_cooc_graph(sent_lemmas, window):
    G = nx.Graph()
    for lemmas in sent_lemmas:
        for i, a in enumerate(lemmas):
            G.add_node(a)
            for j in range(i + 1, min(i + window + 1, len(lemmas))):
                b = lemmas[j]
                if a == b:
                    continue
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
    return G


def topo_features(U, prefix="cooc"):
    f = {}
    n = U.number_of_nodes()
    if n > 0:
        deg = [d for _, d in U.degree()]
        f[prefix + "_avg_degree"] = float(np.mean(deg)) if deg else 0.0
        f[prefix + "_max_degree"] = float(np.max(deg)) if deg else 0.0
        f[prefix + "_density"] = nx.density(U) if n > 1 else 0.0
        f[prefix + "_avg_clustering"] = nx.average_clustering(U) if n > 2 else 0.0
        comps = list(nx.connected_components(U))
        f[prefix + "_n_components"] = len(comps)
        largest = max(comps, key=len) if comps else set()
        f[prefix + "_largest_cc_frac"] = len(largest) / n
        H = U.subgraph(largest)
        if H.number_of_nodes() > 1 and nx.is_connected(H):
            try:
                f[prefix + "_avg_path"] = nx.average_shortest_path_length(H)
                f[prefix + "_diameter"] = float(nx.diameter(H))
            except Exception:
                f[prefix + "_avg_path"], f[prefix + "_diameter"] = np.nan, np.nan
        else:
            f[prefix + "_avg_path"], f[prefix + "_diameter"] = np.nan, np.nan
        f[prefix + "_degree_entropy"] = _entropy(np.bincount(np.asarray(deg, dtype=int))) if deg else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            try:
                a = nx.degree_assortativity_coefficient(U)
                f[prefix + "_assortativity"] = float(a) if a == a else np.nan
            except Exception:
                f[prefix + "_assortativity"] = np.nan
    else:
        for k in ["avg_degree", "max_degree", "density", "avg_clustering",
                  "n_components", "largest_cc_frac", "degree_entropy"]:
            f[prefix + "_" + k] = 0.0
        for k in ["avg_path", "diameter", "assortativity"]:
            f[prefix + "_" + k] = np.nan
    return f


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


def evaluate(X, y):
    scores = []
    for s in range(N_SEEDS):
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        scores.append(qwk(y, cross_val_predict(make_pipe(), X, y, cv=cv)))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in (DATASET, NLP_CACHE):
        if not os.path.exists(f):
            sys.exit(f"missing input: {f}")

    gold = pd.read_parquet(DATASET)[LABEL_COL].astype(int).to_numpy()
    with open(NLP_CACHE, encoding="utf-8") as fh:
        nlp = {int(k): v for k, v in json.load(fh).items()}
    if len(nlp) != len(gold):
        sys.exit(f"cache has {len(nlp)} entries, dataset has {len(gold)}")
    sent_lemmas = [nlp[i]["sent_lemmas"] for i in range(len(gold))]

    print(f"windows {WINDOWS}, {N_SEEDS} seeds each\n")
    rows = []
    for W in WINDOWS:
        X = np.asarray([[topo_features(build_cooc_graph(sl, W)).get(k, np.nan) for k in TOPO_KEYS]
                        for sl in sent_lemmas], dtype=float)
        m, sd = evaluate(X, gold)
        rows.append({"window": W, "qwk_mean": round(m, 3), "qwk_sd": round(sd, 3)})
        print(f"  window = {W}   QWK = {m:.3f} +/- {sd:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "window_sensitivity.csv"), index=False)
    span = res["qwk_mean"].max() - res["qwk_mean"].min()
    print(f"\ntotal range across windows: {span:.3f}")
    print(f"written: {OUT_DIR}/window_sensitivity.csv")


if __name__ == "__main__":
    main()
