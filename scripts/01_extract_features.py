"""
Extract lexical, co-occurrence, named-entity and entity-relation features for
every paragraph of the evaluation set.

This is the feature extraction step of CoGAR. Each paragraph is analysed with
SinaTools (morphology, named entities, relations), turned into a word
co-occurrence graph over content lemmas and a directed entity-relation graph,
and described by 40 features in four families. The AARI surface score is also
computed for use in the ablation.

Inputs
    data/barec_5levels.parquet   columns: ID, text, difficulty_level

Outputs
    outputs/nlp_cache.json       cached SinaTools analyses, reused on re-runs
    outputs/graph_features.csv   one row per paragraph, 40 features + aari

Usage
    python scripts/01_extract_features.py

The SinaTools analysis is slow on the first run. The cache makes later runs
instantaneous, and the downstream scripts (03 to 09) only need the CSV.
"""
import json
import os
import re
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATASET = "data/barec_5levels.parquet"
TEXT_COL = "text"
LABEL_COL = "difficulty_level"
ID_COL = "ID"

OUT_DIR = "outputs"
NLP_CACHE = os.path.join(OUT_DIR, "nlp_cache.json")
FEATURES_CSV = os.path.join(OUT_DIR, "graph_features.csv")

COOC_WINDOW = 2          # co-occurrence window, in content words
RARE_FREQ_THRESHOLD = 50000

FUNCTION_POS_PREFIX = ("حرف", "أداة", "ضمير")   # particles and pronouns


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def split_sentences(text):
    tmp = str(text)
    for p in ["!", "?", "؟"]:
        tmp = tmp.replace(p, ".")
    tmp = tmp.replace(chr(10), ".")
    return [s.strip() for s in tmp.split(".") if s.strip()]


def aari_score(text):
    """Arabic Automatic Readability Index (Al-Tamimi et al., 2014), base score."""
    chars = sum(1 for c in text if not c.isspace())
    words = max(1, len(text.split()))
    sents = max(1, len(split_sentences(text)))
    return 3.28 * chars + 1.43 * (chars / words) + 1.24 * (words / sents)


def first_lemma(lem):
    """SinaTools may return several lemmas separated by ' | '. Keep the first."""
    return lem.split("|")[0].strip() if isinstance(lem, str) else str(lem)


def is_content(pos):
    return not any(str(pos).startswith(p) for p in FUNCTION_POS_PREFIX)


def _entropy(counts):
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------
# Feature families
# --------------------------------------------------------------------------
def lexical_features(tokens, n_sentences):
    f = {}
    n_tok = len(tokens)
    content = [t for t in tokens if is_content(t["pos"])]
    lemmas = [t["lemma"] for t in tokens]
    clemmas = [t["lemma"] for t in content]
    roots = [t.get("root", "") for t in tokens if t.get("root")]
    freqs = [t["freq"] for t in content if t["freq"] > 0]
    f["lex_n_tokens"] = n_tok
    f["lex_words_per_sent"] = n_tok / max(1, n_sentences)
    f["lex_avg_word_len"] = float(np.mean([len(t["token"]) for t in tokens])) if tokens else 0.0
    f["lex_ttr"] = (len(set(lemmas)) / n_tok) if n_tok else 0.0
    f["lex_content_ttr"] = (len(set(clemmas)) / len(clemmas)) if clemmas else 0.0
    f["lex_root_diversity"] = (len(set(roots)) / n_tok) if n_tok else 0.0
    f["lex_mean_log_freq"] = float(np.mean(np.log1p(freqs))) if freqs else 0.0
    f["lex_rare_ratio"] = (float(np.mean([1.0 if fr < RARE_FREQ_THRESHOLD else 0.0 for fr in freqs]))
                           if freqs else 0.0)
    return f


def build_cooc_graph(sent_lemmas, window=COOC_WINDOW):
    """Undirected co-occurrence graph over content lemmas."""
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


def topo_features(U, prefix):
    """Volume (nodes, edges) and topology measures of an undirected graph."""
    f = {}
    n = U.number_of_nodes()
    f[prefix + "_n_nodes"] = n
    f[prefix + "_n_edges"] = U.number_of_edges()
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


def ner_features(entities, n_mentions, n_sentences):
    cats = list(entities.values())
    f = {}
    f["ner_n_entities"] = len(entities)
    f["ner_n_mentions"] = n_mentions
    f["ner_distinct_cats"] = len(set(cats))
    f["ner_entities_per_sent"] = n_mentions / max(1, n_sentences)
    f["ner_cat_entropy"] = _entropy(list(Counter(cats).values())) if cats else 0.0
    return f


def build_rel_graph(relations, entities):
    """Directed multigraph of entities linked by extracted relations."""
    G = nx.MultiDiGraph()
    for r in relations:
        if len(r) < 3:
            continue
        e1, rel, e2 = r[0], r[1], r[2]
        G.add_node(e1, ntype=entities.get(e1, "?"))
        G.add_node(e2, ntype=entities.get(e2, "?"))
        G.add_edge(e1, e2, relation=rel)
    return G


def rel_features(G):
    f = {}
    rels = [d.get("relation") for _, _, d in G.edges(data=True)]
    f["rel_n_relations"] = G.number_of_edges()
    f["rel_n_rel_types"] = len(set(rels))
    f["rel_rel_entropy"] = _entropy(list(Counter(rels).values())) if rels else 0.0
    U = nx.Graph()
    U.add_nodes_from(G.nodes())
    U.add_edges_from((u, v) for u, v in G.edges() if u != v)
    f.update(topo_features(U, "rel"))
    return f


def all_features_for_text(text, nlp):
    f = {}
    f.update(lexical_features(nlp["tokens"], nlp["n_sentences"]))
    f.update(topo_features(build_cooc_graph(nlp["sent_lemmas"]), "cooc"))
    f.update(ner_features(nlp["entities"], nlp["n_mentions"], nlp["n_sentences"]))
    f.update(rel_features(build_rel_graph(nlp["relations"], nlp["entities"])))
    f["aari"] = aari_score(text)
    return f


# --------------------------------------------------------------------------
# SinaTools extraction with cache
# --------------------------------------------------------------------------
def extract_nlp_for_text(text, analyze, entities_and_types, relation_extraction):
    sents = split_sentences(text)
    sent_lemmas, all_tokens, entities, relations, n_mentions = [], [], {}, [], 0
    for s in sents:
        try:
            morph = analyze(s) or []
        except Exception:
            morph = []
        toks = []
        for m in morph:
            toks.append({"token": m.get("token", ""),
                         "lemma": first_lemma(m.get("lemma", "")),
                         "pos": m.get("pos", ""),
                         "root": m.get("root", ""),
                         "freq": int(m.get("frequency", 0) or 0)})
        all_tokens += toks
        sent_lemmas.append([t["lemma"] for t in toks if is_content(t["pos"]) and t["lemma"]])
        try:
            ents = entities_and_types(s) or {}
        except Exception:
            ents = {}
        for k, v in ents.items():
            entities[k] = v
        n_mentions += len(ents)
        try:
            relations += (relation_extraction(s) or [])
        except Exception:
            pass
    return {"n_sentences": max(1, len(sents)), "tokens": all_tokens,
            "sent_lemmas": sent_lemmas, "entities": entities,
            "n_mentions": n_mentions, "relations": relations}


def load_or_build_cache(texts):
    if os.path.exists(NLP_CACHE):
        with open(NLP_CACHE, encoding="utf-8") as fh:
            cache = {int(k): v for k, v in json.load(fh).items()}
        if len(cache) == len(texts):
            print(f"SinaTools analyses loaded from cache: {len(cache)} texts")
            return cache
        print(f"cache has {len(cache)} entries but dataset has {len(texts)}, rebuilding")

    from sinatools.morphology.morph_analyzer import analyze
    from sinatools.relations.relation_extractor import entities_and_types, relation_extraction

    cache = {}
    for i, txt in enumerate(texts):
        cache[i] = extract_nlp_for_text(txt, analyze, entities_and_types, relation_extraction)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} / {len(texts)}")
    with open(NLP_CACHE, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in cache.items()}, fh, ensure_ascii=False)
    print(f"cache written to {NLP_CACHE}")
    return cache


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(DATASET)
    texts = df[TEXT_COL].astype(str).tolist()
    print(f"dataset: {len(df)} paragraphs, "
          f"levels: {df[LABEL_COL].value_counts().sort_index().to_dict()}")

    nlp = load_or_build_cache(texts)

    rows = []
    for i, txt in enumerate(texts):
        row = {"id": df[ID_COL].iloc[i]}
        row.update(all_features_for_text(txt, nlp[i]))
        rows.append(row)
    feat = pd.DataFrame(rows)

    fam = {p: sum(c.startswith(p + "_") for c in feat.columns) for p in ("lex", "cooc", "ner", "rel")}
    print(f"features: lexical {fam['lex']}, co-occurrence {fam['cooc']}, "
          f"named entities {fam['ner']}, entity-relation {fam['rel']}, "
          f"total {sum(fam.values())}")

    feat.to_csv(FEATURES_CSV, index=False)
    print(f"written: {FEATURES_CSV}")


if __name__ == "__main__":
    main()
