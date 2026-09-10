# Data

The datasets are not distributed with this repository. This file describes
how to obtain or build each of them and the exact layout the scripts expect.

## Expected layout

```
data/
  barec_5levels.parquet          evaluation set, 1,000 paragraphs
  barec_release/
    train.csv, dev.csv, test.csv official BAREC sentence files
  crosssource_chunks.csv         unlabelled corpus, 4,200 chunks
  silver_chunks.csv              distillation corpus, 30,000 chunks
```

## barec_5levels.parquet

The 1,000-paragraph evaluation set with five ordinal levels, 200 per level.

| column | content |
|---|---|
| `ID` | unique paragraph identifier |
| `text` | paragraph text |
| `difficulty_level` | integer 0 to 4 |

Levels 1 to 4 are built from the BAREC corpus (Elmadani et al., 2025). Each
paragraph accumulates consecutive sentences of a single BAREC document, in
document order, until a window of 20 to 60 words is reached. Its level is
that of its hardest sentence, following the BAREC aggregation rule. The four
levels correspond to BAREC levels 10, 12, 14 and 16.

Level 0 consists of short sentences from a first-grade Arabic-language
textbook of the Saudi national curriculum. It is not part of BAREC.

BAREC is released at https://huggingface.co/datasets/CAMeL-Lab/BAREC-Shared-Task-2025-sent
under its own terms.

## barec_release/

The official BAREC sentence files, used by `05_camel_baselines.py` to read
the 19-to-7 level grouping. Download `train.csv`, `dev.csv` and `test.csv`
from the BAREC release and place them here. The relevant columns are
`Readability_Level_19` and `Readability_Level_7`.

## crosssource_chunks.csv

Unlabelled chunks of 20 to 60 words from seven sources, 600 per source.

| column | content |
|---|---|
| `uid` | unique chunk identifier |
| `source` | one of `children`, `detective`, `college`, `lycee`, `wikipedia`, `assafir`, `acrps` |
| `text` | chunk text |
| `gold_dummy` | all zeros, fills the harness gold slot |

Sources: children's stories, detective stories, middle-school textbooks,
high-school textbooks, Arabic Wikipedia, As-Safir newspaper articles and
research manuscripts from the Arab Center for Research and Policy Studies.

## silver_chunks.csv

Chunks used for knowledge distillation, 6,000 per source from five sources.

| column | content |
|---|---|
| `chunk_id` | unique chunk identifier |
| `source` | one of `hindawi_easy`, `textbooks`, `wikipedia`, `assafir`, `acrps` |
| `text` | chunk text |

Sources: Hindawi Foundation open-access literature, school textbooks, Arabic
Wikipedia, As-Safir newspaper articles and research manuscripts from the
Arab Center for Research and Policy Studies.

## Provenance

The cross-source and distillation corpora were drawn from the collection of
the Arab Center for Research and Policy Studies. None of the collections
contains personal data. See the appendix of the paper for the full
provenance table.
