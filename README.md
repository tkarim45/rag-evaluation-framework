# RAG Evaluation Framework

> A config-driven RAG pipeline **and** an automated evaluation suite that scores it on
> **Faithfulness**, **Answer Relevancy**, and **Context Recall** — then measures the impact of
> every change (chunk size, retrieval depth, prompt) with reproducible before/after numbers.

Built on **AWS Bedrock** (Claude + Titan embeddings), **LangChain**, **ChromaDB**, and
**RAGAS**. The point isn't "build a RAG app" — most do, then test five questions by hand and
call it done. The point is to *measure* the pipeline, change one variable, and *measure again*.

---

## Why this exists

"The RAG is bad" is useless. "Context Recall is 0.67 at k=2 because retrieval misses evidence on
multi-fact questions" is actionable. This project decomposes RAG quality into three orthogonal
metrics that localize *where* a pipeline fails:

| Metric | Question it answers | Failure it catches |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? | generation hallucinating beyond the docs |
| **Answer Relevancy** | Does the answer address the question? | evasive / off-topic answers |
| **Context Recall** | Did retrieval surface the right documents? | retrieval missing the needed evidence |

The test set is **synthetic** — RAGAS generates question/ground-truth pairs from the corpus, so
there's no human labeling. It's generated once, frozen, and committed, so every experiment scores
against an identical benchmark.

---

## Architecture

```
corpus/ (28 Wikipedia articles)
   │ load + split (chunk_size, overlap)         configs/*.yaml  ← ONE experiment = ONE config
   ▼
┌──────────┐  Titan v2 embeddings   ┌──────────┐
│ ingest.py│───────────────────────▶│ ChromaDB │   (collection cached by chunking params)
└──────────┘                        └────┬─────┘
                                         │ top-k retrieval
   question ────────────────────────────▶│
                                         ▼
                                  ┌────────────┐   answer + retrieved contexts
                                  │   rag.py   │ (Claude Haiku 4.5, temp=0)
                                  └────────────┘
testset/testset.json (frozen QA) ───────────────▶ ┌──────────────┐
                                                  │ run_eval.py  │  RAGAS (judge: Claude Sonnet 4.6)
                                                  └──────┬───────┘  faithfulness · answer_relevancy · context_recall
                                                         ▼
                                              results/<config>.json   (aggregate + per-question)
                                                         │
                                                  compare.py → table · slices · per-question diffs · chart
```

**Design principles**

- **Config-driven experiments** — chunk size, overlap, top-k, prompt, and generation model live in
  a YAML config. A run is fully described by its config + the frozen test set.
- **Fixed judge & embeddings** — the RAGAS judge model and the embedding model are pinned
  *project-wide*, never per-experiment. Changing them would make scores incomparable.
- **Immutable results** — every run writes a new `results/<config>.json`; comparisons read old files.
- **Per-question scores kept** — the interesting analysis is *which* questions fail (multi-hop?
  long docs? a topic?), not just the aggregate.
- **No wasted API spend** — the Chroma collection is keyed by chunking params, so configs that
  differ only in `top_k` or `prompt` reuse an existing vector store instead of re-embedding.

---

## Stack

| Component | Choice |
|---|---|
| Generation LLM | AWS Bedrock — **Claude Haiku 4.5** (`global.anthropic.claude-haiku-4-5-...`) |
| RAGAS judge LLM | AWS Bedrock — **Claude Sonnet 4.6** (fixed across all experiments) |
| Embeddings | AWS Bedrock — **Titan Embed Text v2** (`amazon.titan-embed-text-v2:0`, 1024-dim) |
| Vector store | **ChromaDB** (local, persisted to `chroma_db/`) |
| Orchestration | **LangChain** (`langchain-aws`, loaders, splitters) |
| Evaluation | **RAGAS 0.2.15** (test-set generation + 3 core metrics) |
| Corpus | 28 Wikipedia articles on space exploration |

> **Bedrock note:** Claude on Bedrock requires an **inference-profile** id (the `global.*` / `us.*`
> prefix). Bare model ids are rejected for on-demand invocation. Titan embeddings use the bare id.

---

## Quickstart

Requires an AWS account with Bedrock access in `us-east-1` and **model access enabled** for Claude
Haiku 4.5, Claude Sonnet 4.6, and Titan Embed Text v2.

```bash
# 1. install (Python 3.12)
pip install -r requirements.txt

# 2. credentials — copy and fill in
cp .env.example .env        # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION

# 3. fetch the corpus (committed; skip if corpus/ is populated)
python -m src.fetch_corpus

# 4. build the baseline vector store
python -m src.ingest --config baseline

# 5. smoke-test retrieval + generation
python -m src.rag --config baseline --q "Who first walked on the Moon, and on which mission?"

# 6. ONE-TIME: generate + freeze the synthetic test set, hand-review ~20 Qs, then commit it
python -m src.testset --n 150

# 7. score the baseline
python -m src.run_eval --config baseline        # -> results/baseline.json

# 8. run the experiment sweep (one variable each), then compare
for c in chunk256 chunk512 chunk2000 k2 k8 prompt_permissive; do
  python -m src.ingest --config $c              # k2/k8/prompt_* reuse the baseline store
  python -m src.run_eval --config $c
done
python -m src.compare --all --baseline baseline --slice synthesizer
```

See **[RUNBOOK.md](RUNBOOK.md)** for full operational detail, cost knobs, and the OpenAI→Bedrock mapping.

---

## Experiments

Each experiment changes exactly one variable from the baseline (chunk 1000 / overlap 200, k=4,
strict prompt). Configs live in `configs/`:

| Config | Variable | Hypothesis |
|---|---|---|
| `chunk256` / `chunk512` / `chunk2000` | chunk size | smaller chunks → recall up, faithfulness risk on fragmented context |
| `k2` / `k8` | retrieval depth | higher k → recall up, relevancy/faithfulness may dip from noise |
| `prompt_permissive` | prompt | permissive vs strict grounding → faithfulness/relevancy trade-off |

`compare.py` produces an aggregate table, a per-slice breakdown (by question type), per-question
diffs vs the baseline (which questions a change fixed or broke), and a grouped bar chart in
`reports/`.

---

## Project layout

```
src/
  config.py        # paths, model defaults, ExperimentConfig (judge + embeddings fixed project-wide)
  bedrock.py       # Bedrock model factories + RAGAS wrappers; adaptive-retry client, parallel embeds
  prompts.py       # strict vs permissive grounding prompts
  fetch_corpus.py  # Wikipedia -> corpus/*.txt
  ingest.py        # load -> split -> embed -> persist Chroma collection
  rag.py           # retrieve top-k -> prompt -> generate; returns {answer, contexts}
  testset.py       # RAGAS synthetic test-set generation (one-time, frozen)
  run_eval.py      # run pipeline over test set + RAGAS scoring -> results/<config>.json
  compare.py       # comparison table, slices, per-question diffs, charts
configs/           # one YAML per experiment (baseline + 6 sweeps)
corpus/            # source documents (committed)
testset/           # frozen QA benchmark (committed once generated + reviewed)
results/           # immutable per-run outputs + scores
reports/           # experiment writeups + charts
```

---

## Notes & limitations

- **Judge noise** — RAGAS metrics are themselves LLM-computed; expect ±0.02–0.05 run-to-run
  variance. Don't over-read tiny deltas. The judge model and `temperature=0` are pinned.
- **RAGAS + Claude test-set generation** — RAGAS's `SummaryExtractor` expects strict JSON; Claude
  occasionally returns prose, so some summary nodes are dropped during knowledge-graph construction
  (logged as "Invalid json output"). The run still completes; the effect is fewer multi-hop
  questions. Over-generate (`--n`) for more coverage.
- **Synthetic test-set bias** — questions are generated from the corpus by an LLM, so they inherit
  that model's framing. Hand-review before freezing.
- **Throttling** — Bedrock on-demand throughput throttles under load. The client uses adaptive
  retry; lower `RAGAS_MAX_WORKERS` in `.env` if you still hit `ThrottlingException`.

---

## License

MIT — see [LICENSE](LICENSE).
