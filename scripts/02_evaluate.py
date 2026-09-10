"""
Evaluate readability formulas and LLM raters on a dataset.

Given a dataset and a text column (configured in a TOML file), this script
runs every configured method against the gold labels and writes a comparison.

  - Formula methods: OSMAN and the Arabic-adapted Flesch, rank-binned into
    the five levels (from the dalla-data-processing library).
  - LLM methods: any number of OpenAI-compatible models, one [[llm]] block
    each in the config.

Outputs (under [eval].output_dir)
  - predictions.csv : one row per document, one column per method
  - summary.csv     : metrics per method (exact, within-1, within-2, Spearman, QWK)
  - raw_<llm>.csv   : cached raw LLM responses, reused on re-runs

Usage (from the repository root)
  python scripts/02_evaluate.py --config configs/eval_barec.toml
  python scripts/02_evaluate.py --config configs/eval_barec.toml --limit 50
  python scripts/02_evaluate.py --config configs/eval_barec.toml --only gpt-4.1

API keys are read from a .env file at the repository root or from the
environment. They are never stored in the config files.
"""

import argparse
import csv
import os
import re
import sys
import threading
import time
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

# ----------------------------- metrics -----------------------------


def qwk(gold, pred, k):
    pairs = [(int(g), int(p)) for g, p in zip(gold, pred) if p is not None]
    if not pairs:
        return float("nan")
    n = len(pairs)
    O = [[0] * k for _ in range(k)]
    for a, b in pairs:
        O[a][b] += 1
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * O[i][j]
            den += w * row[i] * col[j] / n
    return 1 - num / den if den else float("nan")


def score_method(gold, pred, k):
    """Return metrics dict over docs where pred is present."""
    pairs = [(int(g), int(p)) for g, p in zip(gold, pred) if p is not None]
    n = len(pairs)
    total = len(pred)
    if n == 0:
        return dict(n=0, coverage=0.0, exact=0.0, within_1=0.0, within_2=0.0,
                    spearman=float("nan"), qwk=float("nan"))
    diffs = [abs(g - p) for g, p in pairs]
    g_only = [g for g, _ in pairs]
    rho = spearmanr(g_only, [p for _, p in pairs])[0] if len(set(g_only)) > 1 else float("nan")
    return dict(
        n=n,
        coverage=n / total,
        exact=sum(d == 0 for d in diffs) / n,
        within_1=sum(d <= 1 for d in diffs) / n,
        within_2=sum(d <= 2 for d in diffs) / n,
        spearman=rho,
        qwk=qwk(gold, pred, k),
    )


# --------------------------- formula methods ---------------------------


def ranks_desc(scores):
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    r = [0] * n
    for k, i in enumerate(order):
        r[i] = k + 1
    return r


def run_formula(texts, cfg):
    """Return {method_name: [level or None per doc]} for the formula methods."""
    # Prefer the installed dalla-data-processing package; fall back to a local
    # checkout only if dalla_path is set and points at one.
    dalla_path = cfg.get("dalla_path")
    if dalla_path and os.path.isdir(dalla_path):
        sys.path.insert(0, dalla_path)
    import textstat
    from dalla_data_processing.readability.arabic_flesch import arabic_flesch_reading_ease
    from dalla_data_processing.readability.ranking import (
        CONSERVATIVE, WEIGHTED, bin_ranks, decide_final_level,
    )
    textstat.set_lang("ar")

    osman, flesch = [], []
    for t in tqdm(texts, desc="formula:score", unit="doc"):
        osman.append(textstat.osman(t) if t and t.strip() else None)
        flesch.append(arabic_flesch_reading_ease(t) if t and t.strip() else None)

    # bin only docs with both scores; others stay None
    valid = [i for i in range(len(texts)) if osman[i] is not None and flesch[i] is not None]
    o_bins_v = bin_ranks(ranks_desc([osman[i] for i in valid]))
    f_bins_v = bin_ranks(ranks_desc([flesch[i] for i in valid]))
    o_bin = [None] * len(texts)
    f_bin = [None] * len(texts)
    for j, i in enumerate(valid):
        o_bin[i] = o_bins_v[j]
        f_bin[i] = f_bins_v[j]

    methods = {}
    for w in cfg["osman_weights"]:
        name = "osman_only" if w == 1.0 else f"weighted_{int(w*100)}_{int(round((1-w)*100))}"
        methods[name] = [
            None if o_bin[i] is None
            else decide_final_level(o_bin[i], f_bin[i], method=WEIGHTED, osman_weight=w)
            for i in range(len(texts))
        ]
    if cfg.get("include_old"):
        methods["old"] = [
            None if o_bin[i] is None
            else decide_final_level(o_bin[i], f_bin[i], method=CONSERVATIVE)
            for i in range(len(texts))
        ]
    if cfg.get("include_flesch_only"):
        methods["flesch_only"] = f_bin
    return methods, osman, flesch, o_bin, f_bin


# ----------------------------- LLM methods -----------------------------

SPLIT_MARKER = "Text to assess:"
PLACEHOLDER = "{PASTE THE ARABIC TEXT HERE}"


def load_env_file(path):
    if not path or not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_messages(template, text):
    if SPLIT_MARKER in template:
        return [{"role": "system", "content": template.split(SPLIT_MARKER)[0].strip()},
                {"role": "user", "content": f'{SPLIT_MARKER}\n"""\n{text}\n"""'}]
    return [{"role": "user", "content": template.replace(PLACEHOLDER, text)}]


def parse_level(content, num_levels):
    if not content:
        return None
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    matches = re.findall(rf"[1-{num_levels}]", content)
    return int(matches[-1]) if matches else None


def call_model(client, model, messages, temperature, max_tokens, extra_body, retries=5):
    delay = 2.0
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, extra_body=extra_body or {})
            return r.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                return f"__ERROR__: {type(e).__name__}: {e}"
            time.sleep(delay)
            delay *= 2


def run_llm(llm, texts, ids, prompt_template, num_levels, output_dir, use_cache, retry_failed):
    """Run one LLM over all docs; returns [pred0 or None]. Caches raw responses.

    By default only docs missing from the cache are called; docs already recorded
    as failed (empty/unparseable) are left as-is. Pass retry_failed=True (e.g. after
    raising max_tokens) to re-call those too.
    """
    from openai import OpenAI
    load_env_file(llm.get("env_file", ".env"))
    api_key = llm.get("api_key") or os.environ.get(llm.get("api_key_env", "OPENAI_API_KEY"))
    if not api_key:
        api_key = "EMPTY" if llm.get("base_url") else None
    if not api_key:
        print(f"  [skip {llm['name']}] no API key")
        return [None] * len(texts)

    client = OpenAI(api_key=api_key, base_url=llm.get("base_url") or None)
    extra_body = ({"chat_template_kwargs": {"enable_thinking": False}}
                  if llm.get("disable_thinking") else None)

    cache_path = os.path.join(output_dir, f"raw_{llm['name']}.csv")
    cache = {}
    if use_cache and os.path.exists(cache_path):
        c = pd.read_csv(cache_path, dtype={"id": str, "raw": str}, keep_default_na=False)
        for _, r in c.iterrows():
            cache[r["id"]] = r["raw"] if r["raw"] != "" else None

    if retry_failed:  # re-call anything not currently parseable
        todo = [i for i in range(len(texts))
                if parse_level(cache.get(str(ids[i])), num_levels) is None]
    else:  # only call docs we have never tried (missing from cache)
        todo = [i for i in range(len(texts)) if str(ids[i]) not in cache]
    if todo:
        lock = threading.Lock()

        def work(i):
            raw = call_model(client, llm["model"], build_messages(prompt_template, texts[i]),
                             llm.get("temperature", 0.0), llm.get("max_tokens", 32), extra_body)
            with lock:
                cache[str(ids[i])] = raw
            return i

        with ThreadPoolExecutor(max_workers=llm.get("workers", 4)) as ex:
            futs = [ex.submit(work, i) for i in todo]
            for _ in tqdm(as_completed(futs), total=len(futs),
                          desc=f"llm:{llm['name']}", unit="doc"):
                pass

        # persist cache
        with open(cache_path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["id", "raw"])
            for i in range(len(texts)):
                raw = cache.get(str(ids[i]))
                wr.writerow([ids[i], (raw or "").replace("\n", " ")[:300]])

    # LLM answers on a 1..num_levels scale; map to the 0-based gold scale.
    out = []
    for i in range(len(texts)):
        lv = parse_level(cache.get(str(ids[i])), num_levels)
        out.append(lv - 1 if lv is not None else None)
    return out


# ------------------------------- main -------------------------------


def load_dataset(d, limit):
    path = d["path"]
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return df


def main():
    ap = argparse.ArgumentParser(description="Unified readability evaluation harness")
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N docs (testing)")
    ap.add_argument("--only", default=None, help="Run a single LLM by name (skips others)")
    ap.add_argument("--no-formula", action="store_true", help="Skip formula methods")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Re-call LLM docs recorded as failed (use after raising max_tokens)")
    args = ap.parse_args()

    with open(args.config, "rb") as fh:
        cfg = tomllib.load(fh)

    ds_cfg, ev = cfg["dataset"], cfg["eval"]
    k = ev["num_levels"]
    out = ev["output_dir"]
    os.makedirs(out, exist_ok=True)
    # Prompt: inline in the config ([eval].prompt) takes precedence; otherwise
    # read it from [eval].prompt_file.
    prompt_template = ev.get("prompt") or open(ev["prompt_file"]).read()

    df = load_dataset(ds_cfg, args.limit)
    texts = df[ds_cfg["text_column"]].astype(str).tolist()
    ids = df[ds_cfg["id_column"]].astype(str).tolist()
    gold = [int(g) - ds_cfg.get("gold_base", 0) for g in df[ds_cfg["gold_column"]].tolist()]
    print(f"dataset: {ds_cfg['path']}  docs={len(df)}  levels={k}")

    columns = {"id": ids, "gold": gold}

    # formula methods
    if cfg.get("formula", {}).get("enabled") and not args.no_formula:
        fmethods, *_ = run_formula(texts, cfg["formula"])
        columns.update(fmethods)

    # llm methods
    for llm in cfg.get("llm", []):
        if args.only and llm["name"] != args.only:
            continue
        columns[f"llm_{llm['name']}"] = run_llm(
            llm, texts, ids, prompt_template, k, out, ev.get("use_cache", True),
            args.retry_failed)

    # write predictions
    method_cols = [c for c in columns if c not in ("id", "gold")]
    pred_path = os.path.join(out, "predictions.csv")
    pd.DataFrame(columns).to_csv(pred_path, index=False)

    # write + print summary
    summ_path = os.path.join(out, "summary.csv")
    rows = [(m, score_method(gold, columns[m], k)) for m in method_cols]
    rows.sort(key=lambda r: (-(r[1]["exact"]), r[0]))
    with open(summ_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "n", "coverage", "exact", "within_1", "within_2", "spearman", "qwk"])
        for m, s in rows:
            wr.writerow([m, s["n"], round(s["coverage"], 3), round(s["exact"], 3),
                         round(s["within_1"], 3), round(s["within_2"], 3),
                         round(s["spearman"], 3), round(s["qwk"], 3)])

    print(f"\nwrote {pred_path}\nwrote {summ_path}\n")
    print(f"{'method':22s} {'n':>5} {'cov':>5} {'exact':>7} {'w1':>6} {'w2':>6} {'rho':>6} {'qwk':>6}")
    for m, s in rows:
        print(f"{m:22s} {s['n']:>5} {s['coverage']:>5.2f} {s['exact']:>7.3f} "
              f"{s['within_1']:>6.3f} {s['within_2']:>6.3f} {s['spearman']:>6.3f} {s['qwk']:>6.3f}")


if __name__ == "__main__":
    main()
