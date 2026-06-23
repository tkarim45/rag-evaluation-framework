"""Run the pipeline over the frozen test set and score it with RAGAS.

  python -m src.run_eval --config configs/baseline.yaml
  python -m src.run_eval --config configs/baseline.yaml --limit 10   # cheap partial run

Writes results/<config>.json with BOTH aggregate and per-question scores (per-question is
where the interesting analysis lives — which questions fail and why). Refuses to overwrite
an existing result file unless --force, per the immutable-results rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

from . import config as C
from .bedrock import ragas_embeddings, ragas_judge, ragas_run_config
from .config import ExperimentConfig, load_config
from .rag import RAGPipeline

METRICS = [faithfulness, answer_relevancy, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall"]


def load_testset(limit: int | None, path: Path | None = None) -> list[dict]:
    path = path or (C.TESTSET_DIR / "testset.json")
    if not path.exists():
        raise SystemExit(f"No frozen test set at {path}. Run: python -m src.testset --n 150")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[:limit] if limit else data


def run_pipeline(cfg: ExperimentConfig, testset: list[dict]) -> list[dict]:
    pipe = RAGPipeline(cfg)
    rows = []
    print(f"Running pipeline (gen={cfg.gen_model}, k={cfg.top_k}, prompt={cfg.prompt}) "
          f"over {len(testset)} questions...")
    for i, item in enumerate(testset, 1):
        out = pipe.answer(item["question"])
        rows.append({
            "user_input": item["question"],
            "response": out["answer"],
            "retrieved_contexts": out["contexts"],
            "reference": item["ground_truth"],
            "synthesizer": item.get("synthesizer", ""),
            "sources": out["sources"],
        })
        if i % 10 == 0 or i == len(testset):
            print(f"  {i}/{len(testset)} answered")
    return rows


def score(rows: list[dict]):
    eval_rows = [{k: r[k] for k in ("user_input", "response", "retrieved_contexts", "reference")} for r in rows]
    dataset = EvaluationDataset.from_list(eval_rows)
    print(f"Scoring with RAGAS (judge={C.JUDGE_MODEL}): {', '.join(METRIC_NAMES)}...")
    result = evaluate(
        dataset=dataset, metrics=METRICS,
        llm=ragas_judge(), embeddings=ragas_embeddings(),
        run_config=ragas_run_config(), show_progress=True,
    )
    return result


def assemble(cfg: ExperimentConfig, rows: list[dict], result) -> dict:
    df = result.to_pandas()
    per_question = []
    for r, (_, scored) in zip(rows, df.iterrows()):
        entry = {
            "question": r["user_input"],
            "synthesizer": r["synthesizer"],
            "sources": r["sources"],
            "n_contexts": len(r["retrieved_contexts"]),
            "answer_chars": len(r["response"]),
        }
        for m in METRIC_NAMES:
            v = scored.get(m)
            entry[m] = float(v) if v is not None and v == v else None  # NaN -> None
        per_question.append(entry)

    def agg(metric):
        vals = [e[metric] for e in per_question if e[metric] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "config": cfg.to_manifest(),
        "n_questions": len(per_question),
        "aggregate": {m: agg(m) for m in METRIC_NAMES},
        "per_question": per_question,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run + RAGAS-score one experiment config")
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, help="Score only the first N questions (cheap partial run)")
    ap.add_argument("--testset", help="Path to a test set JSON (default testset/testset.json)")
    ap.add_argument("--out", help="Output result path (default results/<config>.json)")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing result file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_path = Path(args.out) if args.out else C.RESULTS_DIR / f"{cfg.name}.json"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists. Results are immutable — use --force or a new config name.")

    testset = load_testset(args.limit, Path(args.testset) if args.testset else None)
    rows = run_pipeline(cfg, testset)
    result = score(rows)
    payload = assemble(cfg, rows, result)

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== AGGREGATE ===")
    for m, v in payload["aggregate"].items():
        print(f"  {m:18s} {v}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
