"""
CoGAR predictions on the unlabelled cross-source corpus.

The topology model is trained once on the 1,000 labelled BAREC paragraphs
and then applied to every chunk of the cross-source corpus. The corpus has
no gold labels; its predictions feed the distributional analysis only.

Feature extraction reuses the functions of 01_extract_features.py so that
the corpus is described exactly like the training set. The SinaTools
analysis of the corpus is slow and cached with periodic checkpoints.

Inputs
    data/barec_5levels.parquet        training set
    outputs/nlp_cache.json            BAREC analyses, from 01_extract_features.py
    data/crosssource_chunks.csv       columns: uid, source, text

Outputs
    outputs/crosssource/nlp_cache_corpus.json
    outputs/crosssource/cogar_predictions.csv   uid, source, graph_cooc_topology

Usage
    python scripts/13_crosssource_cogar.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import mord
except ImportError:
    sys.exit("mord is required (pip install mord).")

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BAREC_FILE = "data/barec_5levels.parquet"
BAREC_CACHE = "outputs/nlp_cache.json"
CORPUS_FILE = "data/crosssource_chunks.csv"
OUT_DIR = "outputs/crosssource"
CORPUS_CACHE = os.path.join(OUT_DIR, "nlp_cache_corpus.json")
OUT_FILE = os.path.join(OUT_DIR, "cogar_predictions.csv")

# Feature functions shared with the training set
_spec = importlib.util.spec_from_file_location(
    "features", os.path.join(os.path.dirname(__file__), "01_extract_features.py"))
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


def extract_with_cache(texts, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            cache = {int(k): v for k, v in json.load(fh).items()}
        print(f"  cache: {len(cache)} of {len(texts)} already analysed")
    if len(cache) < len(texts):
        from sinatools.morphology.morph_analyzer import analyze
        from sinatools.relations.relation_extractor import entities_and_types, relation_extraction
        for i in tqdm(range(len(cache), len(texts)), desc="sinatools"):
            cache[i] = features.extract_nlp_for_text(texts[i], analyze, entities_and_types, relation_extraction)
            if (i + 1) % 1000 == 0:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump({str(k): v for k, v in cache.items()}, fh, ensure_ascii=False)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({str(k): v for k, v in cache.items()}, fh, ensure_ascii=False)
    rows = [features.all_features_for_text(texts[i], cache[i]) for i in range(len(texts))]
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in (BAREC_FILE, BAREC_CACHE, CORPUS_FILE):
        if not os.path.exists(f):
            sys.exit(f"missing input: {f}")

    print("training set features")
    barec = pd.read_parquet(BAREC_FILE)
    Xb = extract_with_cache(barec["text"].astype(str).tolist(), BAREC_CACHE)
    yb = barec["difficulty_level"].astype(int).to_numpy()

    print("cross-source corpus features")
    corpus = pd.read_csv(CORPUS_FILE)
    Xc = extract_with_cache(corpus["text"].astype(str).tolist(), CORPUS_CACHE)
    Xc = Xc.reindex(columns=Xb.columns)

    COOC = [c for c in Xb.columns if c.startswith("cooc_")]
    TOPO = [c for c in COOC if c not in ("cooc_n_nodes", "cooc_n_edges")]
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", mord.LogisticAT()),
    ])
    pipe.fit(Xb[TOPO].to_numpy(), yb)
    pred = pipe.predict(Xc[TOPO].to_numpy()).astype(int)

    out = pd.DataFrame({"uid": corpus["uid"], "source": corpus["source"], "graph_cooc_topology": pred})
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nmean predicted level per source:")
    print(out.groupby("source")["graph_cooc_topology"].mean().round(2).to_string())
    print(f"\nwritten: {OUT_FILE}")


if __name__ == "__main__":
    main()
