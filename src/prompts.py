"""Prompt templates for the RAG answer step. Selected per-experiment by `prompt:` key.

The strict/permissive pair is itself an experiment variable (Phase 4): strict grounding
trades Answer Relevancy for Faithfulness; permissive does the reverse.
"""

STRICT = """You are a precise question-answering assistant. Answer the question using ONLY the \
context below. If the context does not contain the answer, reply exactly: "I don't know." \
Do not use outside knowledge. Do not speculate.

Context:
{context}

Question: {question}

Answer:"""

PERMISSIVE = """You are a helpful question-answering assistant. Use the context below to answer \
the question. The context is your primary source, but you may add brief relevant background to \
make the answer complete and useful.

Context:
{context}

Question: {question}

Answer:"""

PROMPTS = {"strict": STRICT, "permissive": PERMISSIVE}


def get_prompt(key: str) -> str:
    if key not in PROMPTS:
        raise KeyError(f"Unknown prompt '{key}'. Available: {sorted(PROMPTS)}")
    return PROMPTS[key]
