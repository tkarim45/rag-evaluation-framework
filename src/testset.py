"""Generate the RAGAS synthetic test set from the corpus — ONCE, then freeze + commit.

  python -m src.testset --n 150
  python -m src.testset --n 5 --out testset/smoke.json    # tiny validation run

Output schema (testset/testset.json): a list of
  {question, ground_truth, reference_contexts, synthesizer}
The same frozen file scores every experiment, so DO NOT regenerate mid-project.
"""
from __future__ import annotations

import argparse
import json

from ragas.testset import TestsetGenerator

from . import config as C
from .bedrock import ragas_embeddings, ragas_judge, ragas_run_config
from .ingest import load_corpus


def generate(n: int, out_path):
    docs = load_corpus()
    print(f"Loaded {len(docs)} corpus docs. Generating {n} QA pairs with judge={C.JUDGE_MODEL}...")
    print("NOTE: this makes many Bedrock LLM calls (knowledge-graph build + question synthesis). "
          "Expect several minutes.")

    generator = TestsetGenerator(llm=ragas_judge(), embedding_model=ragas_embeddings())
    dataset = generator.generate_with_langchain_docs(
        docs, testset_size=n, run_config=ragas_run_config(), with_debugging_logs=False,
    )

    df = dataset.to_pandas()
    records = []
    for _, row in df.iterrows():
        records.append({
            "question": row["user_input"],
            "ground_truth": row["reference"],
            "reference_contexts": list(row.get("reference_contexts", []) or []),
            "synthesizer": row.get("synthesizer_name", ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(records)} QA pairs to {out_path}.")
    print("ACTION: hand-review ~20 questions, drop malformed ones, then commit this file.")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the frozen RAGAS test set")
    ap.add_argument("--n", type=int, default=150, help="Number of QA pairs (target 100-200)")
    ap.add_argument("--out", default=str(C.TESTSET_DIR / "testset.json"))
    args = ap.parse_args()
    from pathlib import Path
    generate(args.n, Path(args.out))


if __name__ == "__main__":
    main()
