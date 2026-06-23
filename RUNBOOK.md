# RUNBOOK — Bedrock build

The implementation uses **AWS Bedrock** (not OpenAI as the README spec suggests). Mapping:

| README (spec) | This build |
|---|---|
| OpenAI embeddings | Bedrock **Titan Embed v2** (`amazon.titan-embed-text-v2:0`, 1024-dim) |
| OpenAI generation | Bedrock **Claude Haiku 4.5** (`global.anthropic.claude-haiku-4-5-...`) |
| OpenAI RAGAS judge | Bedrock **Claude Sonnet 4.6** (`global.anthropic.claude-sonnet-4-6`) — fixed across all experiments |
| ChromaDB local | unchanged (ChromaDB local, `chroma_db/`) |

> Claude on Bedrock requires an **inference-profile** id (`global.*` / `us.*` prefix). Bare
> model ids are rejected for on-demand use. Titan embeddings use the bare model id.

## Setup

```bash
PY=~/miniconda3/envs/personal/bin/python
~/miniconda3/envs/personal/bin/pip install -r requirements.txt
cp .env.example .env        # then fill AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
```

`.env` needs an IAM user with Bedrock invoke access in `us-east-1`, plus **model access enabled**
in the Bedrock console for: Claude Haiku 4.5, Claude Sonnet 4.6, Titan Embed Text v2.

## Pipeline

```bash
PY=~/miniconda3/envs/personal/bin/python

# 0. one-time corpus fetch (28 Wikipedia space-exploration articles) — committed, skip if present
$PY -m src.fetch_corpus

# 1. build the vector store for a config (collection cached by chunking params)
$PY -m src.ingest --config baseline            # add --rebuild to force re-embed

# 2. smoke-test retrieval + generation on one question
$PY -m src.rag --config baseline --q "Who first walked on the Moon, and on which mission?"

# 3. ONE-TIME: generate + freeze the synthetic test set, then commit testset/testset.json
$PY -m src.testset --n 150                      # ~15-25 min, many judge calls; hand-review ~20 Qs after
#    (validate cheaply first:  $PY -m src.testset --n 4 --out testset/smoke.json)

# 4. run + RAGAS-score a config -> results/<name>.json (immutable; --force to overwrite)
$PY -m src.run_eval --config baseline           # add --limit 10 for a cheap partial run

# 5. experiments — one variable each (configs/ already has the sweep)
for c in chunk256 chunk512 chunk2000 k2 k8 prompt_permissive; do
  $PY -m src.ingest   --config $c               # k2/k8/prompt_* reuse the baseline store (no re-embed)
  $PY -m src.run_eval --config $c
done

# 6. compare all runs -> table + slices + per-question diffs + reports/compare_all.png
$PY -m src.compare --all --baseline baseline --slice synthesizer
```

## Design notes

- **Judge + embed models are fixed project-wide** (`src/config.py`), NOT per-experiment — changing
  them would make scores incomparable. Only chunk_size / overlap / top_k / prompt / gen_model vary per config.
- **Chroma collection name** is keyed by chunking params + embed model, so configs that differ only
  in `top_k` or `prompt` transparently reuse an existing store (no wasted embedding API spend).
- **Throttling**: Bedrock on-demand throttles under load. The shared boto3 client uses adaptive
  retry (`src/bedrock.py`); embedding is fanned across `RAGAS_MAX_WORKERS` threads. Lower that env
  var if you still hit `ThrottlingException`.
- **Known caveat — RAGAS testset gen + Claude**: RAGAS's `SummaryExtractor` asks for strict JSON;
  Claude sometimes returns prose, so some summary nodes are dropped (logged as
  "Invalid json output"). The run still completes; effect is fewer multi-hop questions. If you want
  more multi-hop coverage, raise `--n` to over-generate, or pin a stricter judge.

## Cost knobs

- `GEN_MODEL`, `JUDGE_MODEL`, `EMBED_MODEL`, `RAGAS_MAX_WORKERS`, `GEN_MAX_TOKENS` — all overridable in `.env`.
- The expensive step is the one-time `testset.py` (knowledge-graph build) and each `run_eval` (judge calls ≈ n_questions × 3 metrics × several sub-calls).
