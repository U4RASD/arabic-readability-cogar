"""
Cross-source distributional figure and its supporting tables.

For each of the seven sources and each of the eight methods, plots the
distribution of predicted levels and the mean predicted level. Also exports
the underlying numbers and the Kendall tau agreement of every method with
the intuitive difficulty order over six sources. Arabic Wikipedia is
reported separately, since it has no defensible a priori rank.

Inputs
    outputs/crosssource/all_methods.csv   from 15_crosssource_merge.py

Outputs
    outputs/crosssource/figure_cross_source.png
    outputs/crosssource/mean_levels.csv
    outputs/crosssource/distributions.csv
    outputs/crosssource/kendall_tau.csv

Usage
    python scripts/16_figure_crosssource.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

DATA = "outputs/crosssource/all_methods.csv"
OUT_DIR = "outputs/crosssource"
LEVELS = [0, 1, 2, 3, 4]
LABEL_MIN = 1.0
MEAN_MAX = 4.0

METHOD_MAP = {
    "llm_gpt-4.1": "GPT-4.1",
    "llm_qwen3.5-35b": "Qwen-35B",
    "llm_fanar-c-2-27b": "Fanar",
    "camel_arabertv02": "CAMeL",
    "graph_cooc_topology": "CoGAR",
    "aari": "AARI",
    "flesch_only": "Flesch",
    "osman_only": "OSMAN",
}
SOURCE_MAP = {
    "children": "Children's stories",
    "detective": "Detective stories",
    "college": "Middle-school textbooks",
    "lycee": "High-school textbooks",
    "wikipedia": "Arabic Wikipedia",
    "assafir": "Newspaper",
    "acrps": "Research books",
}
SOURCE_ORDER = ["children", "detective", "college", "lycee", "wikipedia", "assafir", "acrps"]
INTUITIVE_ORDER = ["children", "detective", "college", "lycee", "assafir", "acrps"]


def main():
    if not os.path.exists(DATA):
        sys.exit(f"missing input: {DATA}")
    df = pd.read_csv(DATA)
    methods_raw = [m for m in METHOD_MAP if m in df.columns]
    methods = [METHOD_MAP[m] for m in methods_raw]

    def dist_vec(m, s):
        v = df.loc[df["source"] == s, m].dropna().astype(int)
        if len(v) == 0:
            return np.zeros(len(LEVELS))
        p = v.value_counts(normalize=True) * 100
        return np.array([p.get(l, 0.0) for l in LEVELS])

    def mean_lvl(m, s):
        v = df.loc[df["source"] == s, m].dropna()
        return float(v.mean()) if len(v) else np.nan

    dist = {METHOD_MAP[m]: {s: dist_vec(m, s) for s in SOURCE_ORDER} for m in methods_raw}
    mean = {METHOD_MAP[m]: {s: mean_lvl(m, s) for s in SOURCE_ORDER} for m in methods_raw}

    # ----------------------------------------------------------------------
    # Tables
    # ----------------------------------------------------------------------
    means_df = pd.DataFrame({SOURCE_MAP[s]: [round(mean[m][s], 3) for m in methods] for s in SOURCE_ORDER},
                            index=methods)
    means_df.index.name = "Method"
    means_df.to_csv(os.path.join(OUT_DIR, "mean_levels.csv"), encoding="utf-8-sig")

    rows = []
    for m in methods:
        for s in SOURCE_ORDER:
            r = {"Method": m, "Source": SOURCE_MAP[s]}
            for l in LEVELS:
                r[f"Level {l} (%)"] = round(dist[m][s][l], 1)
            r["Mean level"] = round(mean[m][s], 3)
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "distributions.csv"), index=False, encoding="utf-8-sig")

    ranks = list(range(len(INTUITIVE_ORDER)))
    tau_rows = []
    for m in methods:
        t, p = kendalltau(ranks, [mean[m][s] for s in INTUITIVE_ORDER])
        tau_rows.append({"Method": m, "Kendall tau (6 sources)": round(float(t), 3),
                         "p-value": round(float(p), 4),
                         "Wikipedia mean (separate)": round(mean[m]["wikipedia"], 3)})
    tau_df = pd.DataFrame(tau_rows).sort_values("Kendall tau (6 sources)", ascending=False)
    tau_df.to_csv(os.path.join(OUT_DIR, "kendall_tau.csv"), index=False, encoding="utf-8-sig")

    print("mean predicted level per source")
    print(means_df.to_string())
    print("\nKendall tau against the intuitive order, Wikipedia reported separately")
    print(tau_df.to_string(index=False))

    # ----------------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------------
    colors = [plt.cm.tab10(i) for i in range(len(methods))]
    x = np.arange(len(methods))
    ncols = len(LEVELS) + 1
    fig, axes = plt.subplots(len(SOURCE_ORDER), ncols, figsize=(16, 11))

    for i, s in enumerate(SOURCE_ORDER):
        for lvl in LEVELS:
            ax = axes[i][lvl]
            vals = [dist[m][s][lvl] for m in methods]
            bars = ax.bar(x, vals, color=colors, width=0.85)
            ax.set_ylim(0, 100)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            if lvl == 0:
                ax.text(0.03, 0.83 if i == 0 else 0.95, SOURCE_MAP[s], transform=ax.transAxes,
                        fontsize=7, fontweight="bold", ha="left", va="top")
            for b, v in zip(bars, vals):
                if v >= LABEL_MIN:
                    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                            ha="center", va="bottom", fontsize=4.5, rotation=90)
            if i == 0:
                ax.text(0.5, 1.0, f"Level {lvl}", transform=ax.transAxes,
                        fontsize=9, fontweight="bold", ha="center", va="top")
        ax = axes[i][ncols - 1]
        vals = [mean[m][s] for m in methods]
        bars = ax.bar(x, vals, color=colors, width=0.85)
        ax.set_ylim(0, MEAN_MAX)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=4.5, rotation=90)
        if i == 0:
            ax.text(0.5, 1.0, "Mean level", transform=ax.transAxes,
                    fontsize=9, fontweight="bold", ha="center", va="top")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in range(len(methods))]
    fig.legend(handles, methods, loc="lower center", ncol=len(methods),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_png = os.path.join(OUT_DIR, "figure_cross_source.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nwritten: {out_png}")
    print(f"written: {OUT_DIR}/mean_levels.csv, distributions.csv, kendall_tau.csv")


if __name__ == "__main__":
    main()
