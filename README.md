# RAG Evaluation Framework

> **Project 2 of 5 — LLM Evaluation Portfolio.**
> Build a RAG pipeline AND an evaluation suite that scores it across three dimensions —
> Faithfulness, Answer Relevancy, and Context Recall — then prove you can measure the
> impact of every change you make (chunking, retrieval depth, prompts) with before/after
> numbers.

---

## 1. What this project is

Most people who build RAG apps have no idea how well they actually work — they test a
handful of questions manually and call it done. This project closes that gap. You will:

1. Build a simple but real RAG pipeline (documents → chunks → embeddings → vector store →
   retrieval → answer generation).
2. Auto-generate a **synthetic evaluation dataset** (100–200 QA pairs) from your own
   corpus using RAGAS's test-set generator — no human labeling required.
3. Score the pipeline on three orthogonal metrics that localize failures:

   | Metric | Question it answers | Failure it catches |
   |---|---|---|
   | **Faithfulness** | Does the answer stay grounded in retrieved context? | generation hallucinating beyond the docs |
   | **Answer Relevancy** | Does the answer actually address the question? | evasive / off-topic answers |
   | **Context Recall** | Did retrieval surface the right documents? | retrieval missing the needed evidence |

4. Run controlled **experiments**: change one variable (chunk size, top-k, prompt
   template), re-run eval, and analyze the trade-off (one score up, another down?).

The deliverable is an experiment log with before/after scores — *measured, changed,
measured again*. That pattern is what interviewers want to see.

### Why it matters (industry context)

Pinecone, LangChain, and LlamaIndex all use RAGAS as the standard RAG evaluation
framework in their docs and tooling. It became the de facto open-source standard largely
because it needs **no human-labeled data** — it generates its own test set from your
documents. The three-metric decomposition matters because "the RAG is bad" is useless;
"Context Recall is 0.71 on long documents → retrieval is the bottleneck" is actionable.

---

## 2. Dataset — RAGAS synthetic test set (generated from your corpus)

- **Corpus:** any document set works. Good options:
  - 20–50 Wikipedia articles on one subject you know well (lets you sanity-check answers)
  - a set of PDFs (docs, manuals, papers)
  - alternatively, seed from HuggingFace QA corpora (SQuAD, Natural Questions) as documents
- **Test set:** RAGAS `TestsetGenerator` produces realistic question/ground-truth pairs
  from the corpus, including different question types (simple, reasoning, multi-context).
  Target **100–200 pairs**.
- **Critical rule:** generate once, eyeball ~20 questions for quality (drop broken ones),
  then **freeze and commit** the test set. All experiments score against the same frozen set.
- Docs: https://docs.ragas.io/en/stable/getstarted/rag_eval

## 3. Tools / frameworks

| Tool | Role | Install |
|---|---|---|
| **RAGAS** | test-set generation + 3 core metrics | `pip install ragas` |
| **LangChain** | RAG orchestration (loaders, splitters, chains) | `pip install langchain langchain-openai langchain-community` |
| **ChromaDB** | local vector store, zero infra | `pip install chromadb` |
| `openai` | embeddings + generation + RAGAS judge LLM | `pip install openai` |
| `pandas`, `matplotlib` | experiment comparison + charts | `pip install pandas matplotlib` |

- RAGAS: https://github.com/explodinggradients/ragas — works with any LLM/retriever,
  integrates directly with LangChain and LlamaIndex.
- Note RAGAS metrics are themselves LLM-computed (judge model) — pin the judge model and
  temperature, and expect ±0.02–0.05 run-to-run noise; don't over-read tiny deltas.

---

## 4. Architecture

```
corpus/ (docs)
   │ load + split (chunk_size, overlap)        configs/*.yaml  ← ONE experiment = ONE config
   ▼
┌──────────┐  embeddings   ┌──────────┐
│ ingest.py│──────────────▶│ ChromaDB │
└──────────┘               └────┬─────┘
                                │ top-k retrieval
   question ───────────────────▶│
                                ▼
                         ┌────────────┐   answer + retrieved contexts
                         │   rag.py   │──────────────┐
                         └────────────┘              ▼
testset/ (frozen QA pairs) ────────────────▶ ┌──────────────┐
                                             │ run_eval.py  │  RAGAS:
                                             │              │  faithfulness
                                             │              │  answer_relevancy
                                             │              │  context_recall
                                             └──────┬───────┘
                                                    ▼
                                          results/<config>.json
                                                    │
                                             compare.py → before/after table, charts
```

Design principles:
- **Config-driven experiments**: chunk size, overlap, top-k, prompt template, models all
  live in a YAML config; a run is fully described by its config + frozen test set.
- **Immutable results**: every run writes a new file; comparisons read old files.
- **Per-question scores kept**, not just aggregates — the interesting analysis is *which*
  questions fail (long docs? multi-hop? specific topics?).

---

## 5. Build plan (step by step)

### Phase 0 — Setup (30 min)
1. `~/miniconda3/envs/personal/bin/pip install ragas langchain langchain-openai langchain-community chromadb openai pandas matplotlib python-dotenv pyyaml`
2. `.env` with `OPENAI_API_KEY`; collect corpus into `corpus/` (start with ~20 Wikipedia
   articles on one topic — fetch with `wikipedia` package or save manually).

### Phase 1 — RAG pipeline (half day)
1. `src/ingest.py`: load docs → `RecursiveCharacterTextSplitter(chunk_size, overlap)` →
   OpenAI embeddings → persist ChromaDB collection named after the config.
2. `src/rag.py`: `retrieve(question, k)` → stuff contexts into a prompt template →
   generate answer (temperature=0). Return `{answer, contexts}` — RAGAS needs both.
3. Manual smoke test on 5 questions you can verify yourself.

### Phase 2 — Synthetic test set (2–3 hrs)
1. `src/testset.py`: RAGAS `TestsetGenerator` over the corpus → 150 QA pairs with ground
   truths. Review ~20 by hand; drop malformed ones.
2. Freeze: write `testset/testset.json`, commit it. Never regenerate mid-project.

### Phase 3 — Evaluation harness (half day)
1. `src/run_eval.py`: for each test question → run pipeline → collect
   `{question, answer, contexts, ground_truth}` → build RAGAS `EvaluationDataset` →
   `evaluate()` with `faithfulness`, `answer_relevancy`, `context_recall`.
2. Save aggregate + per-question scores to `results/<config>.json`.
3. Run the **baseline config** (e.g. chunk 1000/overlap 200, k=4). These are your
   "before" numbers.

### Phase 4 — Experiments (1 day, the core of the project)
Run one-variable experiments, each as a new config:
1. **Chunk size sweep:** 256 / 512 / 1000 / 2000. Hypothesis: small chunks → recall up,
   faithfulness risk on fragmented context.
2. **Retrieval depth sweep:** k = 2 / 4 / 8. Hypothesis: higher k → recall up,
   relevancy/faithfulness may dip from noise.
3. **Prompt variants:** strict grounding prompt ("answer ONLY from context, else say you
   don't know") vs permissive. Watch faithfulness vs relevancy trade-off.
4. `src/compare.py`: table of all runs × 3 metrics; per-question diffs to find which
   questions a change fixed/broke. Slice by question type / document length —
   "ContextRecall 0.71 on long documents" is the kind of localized finding that makes
   the project.

### Phase 5 — Report (2–3 hrs)
`reports/report.md`: baseline scores, experiment matrix, the trade-off you found,
per-slice analysis, the config you'd ship and why, limitations (judge noise, synthetic
test-set bias).

### Stretch goals
- Add a reranker (e.g. Cohere rerank or a cross-encoder) as an experiment.
- Add hybrid retrieval (BM25 + dense) via LangChain's `EnsembleRetriever`.
- Track ContextPrecision as a 4th metric; compare against ContextRecall.

---

## 6. Definition of done

- [ ] Working RAG pipeline, config-driven
- [ ] Frozen, committed synthetic test set (100–200 QA pairs, hand-reviewed sample)
- [ ] Baseline run with all 3 RAGAS metrics, per-question scores saved
- [ ] ≥4 one-variable experiments with immutable result files
- [ ] Comparison table + at least one localized finding (metric × slice)
- [ ] `reports/report.md` with before/after numbers and a shipped-config recommendation

## 7. Resume bullets (template — replace with YOUR real numbers)

- *Built RAGAS evaluation suite scoring RAG pipeline on [N] synthetic test cases:
  Faithfulness [x.xx], AnswerRelevancy [x.xx], ContextRecall [x.xx].*
- *Identified retrieval as bottleneck (ContextRecall [x.xx] on long documents) —
  chunking strategy change improved score to [x.xx] without affecting Faithfulness.*

Run eval before and after a change; the before/after comparison IS the bullet.

## 8. References

- RAGAS docs: https://docs.ragas.io/en/stable/getstarted/rag_eval
- RAGAS repo: https://github.com/explodinggradients/ragas
- RAGAS paper: *"RAGAS: Automated Evaluation of Retrieval Augmented Generation"* (Es et al., 2023)
- LangChain RAG tutorial: https://python.langchain.com/docs/tutorials/rag/
