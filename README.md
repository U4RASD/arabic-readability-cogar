# Knowledge Graphs and Distributional Readability of Arabic Documents

Code for the paper *Knowledge Graphs and Distributional Readability of Arabic
Documents* (ArabicNLP 2026).

The paper compares classical readability formulas, fine-tuned Arabic
encoders, zero-shot large language models and CoGAR, an interpretable and
LLM-free method that represents a document as a word co-occurrence network
over content lemmas and predicts its reading level from the topology of that
network. This repository reproduces every table and figure of the paper.

## Pipeline

```
data/barec_5levels.parquet
        |
        v
01_extract_features.py  ---------------->  outputs/graph_features.csv
        |                                  outputs/nlp_cache.json
        v
02_evaluate.py (eval_barec.toml) ------->  outputs/barec/predictions.csv
        |
        +--> 03_cogar_crossval.py -------->  Table 1 (CoGAR rows), Table 2
        +--> 04_formulas.py -------------->  Table 1 (formula rows)
        +--> 05_camel_baselines.py ------->  Table 1 (CAMeL rows)
        +--> 06_significance.py ---------->  significance tests
        +--> 07_stacking.py -------------->  Table 4
        +--> 08_window_sensitivity.py ---->  window sensitivity table
        +--> 09_level0_excluded.py ------->  level-0-excluded evaluation

data/silver_chunks.csv
        |
        v
10_label_with_gemini.py --> 11_split_silver.py --> 12_finetune_neoarabert.py
                                                            |
                                                            v
                                                    Table 3, Figure 1

data/crosssource_chunks.csv
        |
        +--> 02_evaluate.py (eval_crosssource.toml)
        +--> 13_crosssource_cogar.py
        +--> 14_crosssource_camel.py
                    |
                    v
        15_crosssource_merge.py --> 16_figure_crosssource.py --> Figure 2
```

Every script is run from the repository root and documents its inputs,
outputs and usage in its header.

## Setup

```bash
git clone <repository-url>
cd <repository>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your API keys
```

Two dependencies need attention.

**SinaTools 1.0.9** provides the morphological analysis, named-entity
recognition and relation extraction. Follow its installation instructions
for the language models it downloads on first use.

**dalla-data-processing** provides the Arabic-adapted Flesch formula and the
rank binning used by the evaluation harness. Install it from its
repository, or point `[formula].dalla_path` in the config to a local checkout.

Python 3.11 or later is recommended. On earlier versions, `tomli` is used
to read the config files.

## Data

The datasets are not included. `data/README.md` describes each file, its
columns and how to obtain or build it.

## Reproducing the main results

The commands below reproduce Tables 1, 2 and 4 and the robustness analyses.
Each script prints its results and writes a CSV under `outputs/tables/`.

```bash
# 1. Features. Slow on the first run (SinaTools), cached afterwards.
python scripts/01_extract_features.py

# 2. LLM raters and formulas. Needs API keys in .env.
python scripts/02_evaluate.py --config configs/eval_barec.toml

# 3. CoGAR under cross-validation, 10 seeds. Tables 1 and 2.
python scripts/03_cogar_crossval.py

# 4. Formula rows of Table 1.
python scripts/04_formulas.py

# 5. CAMeL baselines. Needs data/barec_release/ and camel-tools.
python scripts/05_camel_baselines.py

# 6. Paired bootstrap over seeds and documents.
python scripts/06_significance.py

# 7. Stacking, Table 4.
python scripts/07_stacking.py

# 8. Co-occurrence window sensitivity.
python scripts/08_window_sensitivity.py

# 9. Evaluation restricted to the four BAREC levels.
python scripts/09_level0_excluded.py
```

Expected values, quadratic weighted kappa on the 1,000-paragraph set. CoGAR
values are means over ten cross-validation seeds.

| system | QWK |
|---|---|
| CAMeL arabertv2-d3tok-reg | 0.860 |
| GPT-4.1 (zero-shot) | 0.784 |
| CoGAR (topology) | 0.756 ± 0.001 |
| Qwen-35B (zero-shot) | 0.751 |
| CoGAR (all features) | 0.750 ± 0.005 |
| NeoAraBERT (distilled, focal) | 0.714 |
| Fanar (zero-shot) | 0.702 |
| CAMeL arabertv02-word-CE | 0.694 |
| AARI | 0.666 |
| OSMAN | 0.215 |
| Flesch | 0.186 |
| Stacking, Qwen-35B + topology | 0.852 ± 0.002 |
| Stacking, GPT-4.1 + topology | 0.839 ± 0.001 |

The LLM raters are deterministic under greedy decoding, but their outputs
depend on the exact model version served by each provider at query time.
Small drifts are possible if a provider updates a model.

## Reproducing the distillation

```bash
python scripts/10_label_with_gemini.py       # needs GEMINI_API_KEY
python scripts/11_split_silver.py
python scripts/12_finetune_neoarabert.py     # GPU recommended
```

The student never sees a human label. It is evaluated on the same
1,000-paragraph BAREC set as every other system.

## Reproducing the cross-source analysis

```bash
python scripts/02_evaluate.py --config configs/eval_crosssource.toml
python scripts/13_crosssource_cogar.py
python scripts/14_crosssource_camel.py
python scripts/15_crosssource_merge.py
python scripts/16_figure_crosssource.py
```

The corpus is unlabelled, so this analysis probes generalisation rather
than accuracy. `16_figure_crosssource.py` writes the figure together with
the mean level per source, the full level distributions and the Kendall
tau of every method against the intuitive difficulty order.

## The CoGAR features

Forty features in four families, listed in `01_extract_features.py` and in
the appendix of the paper.

| family | count | content |
|---|---|---|
| lexical | 8 | token counts, sentence length, type-token ratios, root diversity, lemma frequency |
| co-occurrence, volume | 2 | number of nodes and edges of the co-occurrence graph |
| co-occurrence, topology | 10 | density, clustering, path length, diameter, degree statistics, assortativity |
| named entities | 5 | entity and mention counts, category diversity |
| entity-relation graph | 15 | relation counts and types, and the same topology measures on the entity graph |

The topology configuration uses the ten co-occurrence topology features
only. It is the method adopted in the paper.

## Ordinal evaluation

Readability is treated as an ordinal task throughout. The regressor is
`mord.LogisticAT`, the primary metric is quadratic weighted kappa, and every
supervised result is out-of-fold under stratified five-fold
cross-validation. The stacking meta-learner has no tuned hyper-parameters,
so a single cross-validation level is unbiased.

## Citation

```bibtex
@inproceedings{chhaytle-etal-2026-knowledge,
  title     = {Knowledge Graphs and Distributional Readability of {A}rabic Documents},
  author    = {Chhaytle, Mohamad and Hamoud, Hadi and Choker, Sally and
               Abou Chakra, Chadi and Ballout, Mohamad and Zaraket, Fadi A.},
  booktitle = {Proceedings of the Fourth Arabic Natural Language Processing Conference},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

## License

MIT. See `LICENSE`.
