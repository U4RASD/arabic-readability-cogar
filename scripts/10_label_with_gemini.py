"""
Silver labelling of the distillation corpus with Gemini.

Every chunk is rated on the five-level scale by Gemini, with the same
instruction as the zero-shot raters. The labels serve as silver training
data for the distilled encoder. Responses are cached so that an interrupted
run can resume.

Inputs
    data/silver_chunks.csv     columns: chunk_id, source, text
                               (30,000 chunks, six sources)

Outputs
    outputs/silver/gemini_labels.csv          cache, chunk_id and level
    outputs/silver/silver_chunks_labeled.csv  input rows with a level column

Usage
    GEMINI_API_KEY=... python scripts/10_label_with_gemini.py
    python scripts/10_label_with_gemini.py --limit 200    quick test

The key is read from the GEMINI_API_KEY environment variable or from a
.env file at the repository root.
"""
import argparse
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

IN_FILE = "data/silver_chunks.csv"
OUT_DIR = "outputs/silver"
CACHE = os.path.join(OUT_DIR, "gemini_labels.csv")
OUT_FILE = os.path.join(OUT_DIR, "silver_chunks_labeled.csv")
MODEL = "gemini-3.5-flash"
WORKERS = 8

PROMPT = (
    "You are an expert in Arabic linguistics and reading comprehension. Your task is to assess "
    "the readability level of a given Arabic text. Use the following 5-level scale: "
    "1 = Very Easy (texts for early primary school, very simple vocabulary and short sentences) "
    "2 = Easy (texts for upper primary school, common everyday vocabulary) "
    "3 = Medium (texts for middle school, moderately complex vocabulary and sentence structure) "
    "4 = Difficult (texts for high school or undergraduate level, advanced vocabulary and complex syntax) "
    "5 = Very Difficult (specialized or academic texts, technical vocabulary, complex argumentation) "
    "Rules: Base your judgment on vocabulary complexity, sentence length and structure, and level of "
    "abstraction. Do NOT consider the topic or domain alone, focus on linguistic difficulty. "
    "Respond with ONLY a single integer: 1, 2, 3, 4 or 5. Nothing else."
)
DIGIT = re.compile(r"[1-5]")


def load_env_file(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="label only the first N chunks per source")
    args = ap.parse_args()

    load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(system_instruction=PROMPT)

    def label_text(text):
        for attempt in range(5):
            try:
                r = client.models.generate_content(model=MODEL, contents=str(text), config=config)
                m = DIGIT.findall(r.text or "")
                return int(m[-1]) if m else None
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_FILE)
    if args.limit:
        df = df.groupby("source", group_keys=False).head(args.limit)
    print(f"chunks: {len(df)}  sources: {df['source'].value_counts().to_dict()}")

    done = {}
    if os.path.exists(CACHE):
        prev = pd.read_csv(CACHE)
        done = dict(zip(prev["chunk_id"].astype(str), prev["level"]))
    todo = df[~df["chunk_id"].astype(str).isin(done)]
    results = dict(done)
    lock = threading.Lock()

    def flush():
        with lock:
            pd.DataFrame({"chunk_id": list(results), "level": list(results.values())}).to_csv(CACHE, index=False)

    def work(cid, text):
        lvl = label_text(text)
        with lock:
            results[cid] = lvl
        return cid

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, str(r.chunk_id), r.text) for r in todo.itertuples(index=False)]
        for i, _ in enumerate(tqdm(as_completed(futs), total=len(futs), desc="gemini")):
            if (i + 1) % 200 == 0:
                flush()
    flush()

    labels = pd.DataFrame({"chunk_id": list(results), "level": list(results.values())})
    labels["chunk_id"] = labels["chunk_id"].astype(str)
    df["chunk_id"] = df["chunk_id"].astype(str)
    labeled = df.merge(labels, on="chunk_id", how="left")
    labeled.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"labelled: {labeled['level'].notna().sum()} / {len(labeled)}")
    print(f"written: {OUT_FILE}")


if __name__ == "__main__":
    main()
