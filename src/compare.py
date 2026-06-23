"""Compare experiment result files: aggregate table, per-question diffs, slice analysis, charts.

  python -m src.compare results/baseline.json results/chunk512.json ...
  python -m src.compare --all                  # every results/*.json
  python -m src.compare --all --baseline baseline --slice synthesizer

Outputs a console table, a per-slice breakdown, and a grouped bar chart to
reports/compare_<...>.png. Per-question diffs vs the baseline surface which questions a
change fixed or broke.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C

METRICS = ["faithfulness", "answer_relevancy", "context_recall"]


def load(paths: list[Path]) -> dict[str, dict]:
    runs = {}
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        runs[data["config"]["name"]] = data
    return runs


def aggregate_table(runs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, d in runs.items():
        row = {"config": name, "n": d["n_questions"]}
        row.update({m: d["aggregate"].get(m) for m in METRICS})
        c = d["config"]
        row.update(chunk=c.get("chunk_size"), overlap=c.get("chunk_overlap"),
                   k=c.get("top_k"), prompt=c.get("prompt"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("config")


def per_question_frame(d: dict) -> pd.DataFrame:
    return pd.DataFrame(d["per_question"]).set_index("question")


def slice_table(runs: dict[str, dict], slice_key: str) -> pd.DataFrame:
    frames = []
    for name, d in runs.items():
        df = pd.DataFrame(d["per_question"])
        if slice_key not in df.columns:
            continue
        g = df.groupby(slice_key)[METRICS].mean().round(4)
        g.columns = pd.MultiIndex.from_product([[name], g.columns])
        frames.append(g)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def diff_vs_baseline(runs: dict[str, dict], baseline: str) -> dict[str, pd.DataFrame]:
    base = per_question_frame(runs[baseline])
    out = {}
    for name, d in runs.items():
        if name == baseline:
            continue
        cur = per_question_frame(d)
        joined = base[METRICS].join(cur[METRICS], lsuffix="_base", rsuffix="_new", how="inner")
        for m in METRICS:
            joined[f"{m}_delta"] = (joined[f"{m}_new"] - joined[f"{m}_base"]).round(4)
        out[name] = joined
    return out


def chart(table: pd.DataFrame, out_path: Path) -> None:
    ax = table[METRICS].plot(kind="bar", figsize=(max(7, 1.6 * len(table)), 5), rot=20, ylim=(0, 1))
    ax.set_title("RAGAS metrics by experiment")
    ax.set_ylabel("score (0–1)")
    ax.legend(loc="lower right")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=7, padding=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nChart -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare RAGAS experiment results")
    ap.add_argument("results", nargs="*", help="result JSON files")
    ap.add_argument("--all", action="store_true", help="Use every results/*.json")
    ap.add_argument("--baseline", default="baseline", help="Config name to diff others against")
    ap.add_argument("--slice", dest="slice_key", default="synthesizer",
                    help="Per-question field to slice by (e.g. synthesizer)")
    ap.add_argument("--top", type=int, default=8, help="Show top-N most-changed questions per run")
    args = ap.parse_args()

    paths = sorted(C.RESULTS_DIR.glob("*.json")) if args.all else [Path(p) for p in args.results]
    if not paths:
        raise SystemExit("No result files. Pass paths or --all.")
    runs = load(paths)

    pd.set_option("display.width", 160, "display.max_columns", 30)
    table = aggregate_table(runs)
    print("\n=== AGGREGATE METRICS ===")
    print(table.to_string())

    st = slice_table(runs, args.slice_key)
    if not st.empty:
        print(f"\n=== SLICED BY '{args.slice_key}' (mean per group) ===")
        print(st.to_string())

    if args.baseline in runs and len(runs) > 1:
        print(f"\n=== PER-QUESTION DIFF vs '{args.baseline}' (top {args.top} by |context_recall delta|) ===")
        for name, df in diff_vs_baseline(runs, args.baseline).items():
            top = df.reindex(df["context_recall_delta"].abs().sort_values(ascending=False).index).head(args.top)
            cols = [f"{m}_delta" for m in METRICS]
            print(f"\n-- {name} --")
            print(top[cols].to_string())

    out_png = C.REPORTS_DIR / ("compare_all.png" if args.all else "compare.png")
    chart(table, out_png)


if __name__ == "__main__":
    main()
