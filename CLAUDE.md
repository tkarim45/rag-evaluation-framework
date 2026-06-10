# CLAUDE.md — RAG Evaluation Framework

## What this repo is

An LLM evaluation project: a working RAG pipeline (LangChain + ChromaDB + OpenAI) plus a
RAGAS evaluation suite that scores it on Faithfulness, Answer Relevancy, and Context
Recall over a synthetic test set generated from the corpus — then measures the impact of
pipeline changes (chunk size, retrieval depth, prompts) before/after. See `README.md`
for the full project specification and build plan.

## Environment

- Python env: conda `personal` env (`~/miniconda3/envs/personal/bin/python`).
  NEVER use the conda `base` env or system Python.
- Install deps: `~/miniconda3/envs/personal/bin/pip install -r requirements.txt`
- Secrets in `.env` (gitignored). Required: `OPENAI_API_KEY`. Load via `python-dotenv`.

## Project structure (target layout)

```
.
├── CLAUDE.md
├── README.md                  # full project spec — read this first
├── requirements.txt
├── .env.example
├── corpus/                    # source documents (wiki articles / PDFs)
├── src/
│   ├── ingest.py              # load, chunk, embed, store in ChromaDB
│   ├── rag.py                 # retrieval + answer generation pipeline
│   ├── testset.py             # RAGAS synthetic test-set generation
│   ├── run_eval.py            # run pipeline over test set + RAGAS scoring
│   └── compare.py             # before/after experiment comparison
├── configs/                   # experiment configs (chunk size, k, prompt) as YAML/JSON
├── testset/                   # generated QA pairs (committed — they're the benchmark)
├── results/                   # per-run outputs + scores, named by config
└── reports/                   # experiment writeups + charts (committed)
```

## Conventions

- The synthetic test set is generated ONCE, then frozen and committed — every experiment
  scores against the same questions or comparisons are meaningless.
- One experiment = one config file. Results saved as `results/<config-name>.json`.
  Never overwrite a previous run.
- Change ONE variable per experiment (chunk size OR k OR prompt), never several.
- Cache embeddings and model outputs; re-analysis must not re-spend API calls.
- Pin versions: `ragas`, `langchain`, embedding model, generation model, temperature=0.
- Tests/scratch scripts Claude writes for itself → `claude` env, not `personal`.

## Key commands (once built)

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY src/ingest.py --config configs/baseline.yaml      # build vector store
$PY src/testset.py --n 150                            # one-time test-set generation
$PY src/run_eval.py --config configs/baseline.yaml    # answer + RAGAS score
$PY src/compare.py results/baseline.json results/chunk512.json
```
