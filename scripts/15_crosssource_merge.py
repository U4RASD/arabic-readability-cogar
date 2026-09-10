"""
Merge every method's predictions on the cross-source corpus into one table.

Combines the LLM and formula predictions of the evaluation harness with the
CoGAR and CAMeL predictions, and adds AARI computed from the raw text. The
result is the input of the distributional figure.

Formula levels are taken from the harness as produced. On these 20 to 60
word chunks the rank bins already increase with difficulty. The reversal
applied to the BAREC set in 04_formulas.py concerns the very short level-0
sentences of that set and does not apply here.

Inputs
    data/crosssource_chunks.csv                   uid, source, text
    outputs/crosssource/predictions.csv           from 02_evaluate.py
    outputs/crosssource/cogar_predictions.csv     from 13_crosssource_cogar.py
    outputs/crosssource/camel_predictions.csv     from 14_crosssource_camel.py

Outputs
    outputs/crosssource/all_methods.csv

Usage
    python scripts/15_crosssource_merge.py
"""
import os
import re
import sys

import pandas as pd

CORPUS_FILE = "data/crosssource_chunks.csv"
OUT_DIR = "outputs/crosssource"
HARNESS_PRED = os.path.join(OUT_DIR, "predictions.csv")
COGAR_PRED = os.path.join(OUT_DIR, "cogar_predictions.csv")
CAMEL_PRED = os.path.join(OUT_DIR, "camel_predictions.csv")
OUT_FILE = os.path.join(OUT_DIR, "all_methods.csv")

AR_DIAC = re.compile(r"[\u064B-\u065F\u0670]")


def aari_base(text):
    t = AR_DIAC.sub("", str(text))
    words = t.split()
    n_words = max(len(words), 1)
    n_sents = max(len(re.findall(r"[.!?\u061F\n]+", t)), 1)
    n_chars = len(re.sub(r"[^\w]", "", t, flags=re.UNICODE))
    return 3.28 * n_chars + 1.43 * (n_chars / n_words) + 1.24 * (n_words / n_sents)


def rank_levels(scores, k=5):
    s = pd.Series(list(scores)).reset_index(drop=True)
    r = s.rank(method="first", ascending=True)
    return ((r - 1) / len(s) * k).astype(int).clip(0, k - 1).values


def main():
    for f in (CORPUS_FILE, HARNESS_PRED, COGAR_PRED, CAMEL_PRED):
        if not os.path.exists(f):
            sys.exit(f"missing input: {f}")

    corpus = pd.read_csv(CORPUS_FILE)[["uid", "source", "text"]]
    corpus["uid"] = corpus["uid"].astype(str)

    P = pd.read_csv(HARNESS_PRED).rename(columns={"id": "uid"})
    P["uid"] = P["uid"].astype(str)
    P = P.merge(corpus[["uid", "source"]], on="uid", how="left")

    for path, col in [(COGAR_PRED, "graph_cooc_topology"), (CAMEL_PRED, "camel_arabertv02")]:
        extra = pd.read_csv(path)[["uid", col]]
        extra["uid"] = extra["uid"].astype(str)
        P = P.merge(extra, on="uid", how="left")

    aari = pd.DataFrame({"uid": corpus["uid"], "aari": rank_levels(corpus["text"].map(aari_base))})
    P = P.merge(aari, on="uid", how="left")

    keep = ["uid", "source", "llm_gpt-4.1", "llm_qwen3.5-35b", "llm_fanar-c-2-27b",
            "camel_arabertv02", "graph_cooc_topology", "aari", "flesch_only", "osman_only"]
    keep = [c for c in keep if c in P.columns]
    P = P[keep]
    P.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print("chunks per source:", P["source"].value_counts().to_dict())
    print("\nmean predicted level per source (sanity check, should rise from "
          "children's stories to research books for the learned methods):")
    print(P.groupby("source")[[c for c in keep if c not in ("uid", "source")]].mean().round(2).to_string())
    print(f"\nwritten: {OUT_FILE}")


if __name__ == "__main__":
    main()
