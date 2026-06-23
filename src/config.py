"""Config loading + project paths + model defaults.

An experiment is fully described by a YAML file in configs/. The JUDGE model and
the EMBED model are deliberately NOT part of the experiment config: they are fixed
project-wide so that every run's scores are comparable. Only the variables you are
sweeping (chunk_size, chunk_overlap, top_k, prompt, gen_model) live per-config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
CONFIG_DIR = ROOT / "configs"
TESTSET_DIR = ROOT / "testset"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
CHROMA_DIR = ROOT / "chroma_db"
CACHE_DIR = ROOT / ".cache"

for _d in (CORPUS_DIR, CONFIG_DIR, TESTSET_DIR, RESULTS_DIR, REPORTS_DIR, CHROMA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

# --- fixed, project-wide model + infra defaults ------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Generation model for the RAG pipeline under test (can be overridden per-config).
GEN_MODEL = os.getenv("GEN_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
# RAGAS judge — FIXED across experiments. Override only via .env, never per-config.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "global.anthropic.claude-sonnet-4-6")
# Embeddings — FIXED across experiments (changing it invalidates the vector store).
EMBED_MODEL = os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")

# Bedrock concurrency. Titan + Claude on-demand throttle easily; keep modest.
MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "4"))
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "1024"))


@dataclass
class ExperimentConfig:
    """One experiment = one of these, loaded from configs/<name>.yaml."""

    name: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    prompt: str = "strict"          # key into src.prompts.PROMPTS
    gen_model: str = GEN_MODEL
    embed_model: str = EMBED_MODEL  # fixed; surfaced here for the run manifest
    description: str = ""

    @property
    def collection_name(self) -> str:
        """Chroma collection is keyed by the ingest-affecting params only, so configs
        that differ solely in top_k / prompt reuse the same (cached) vector store."""
        safe_embed = self.embed_model.replace(":", "_").replace(".", "_").replace("/", "_")
        return f"cs{self.chunk_size}_co{self.chunk_overlap}_{safe_embed}"

    def to_manifest(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(judge_model=JUDGE_MODEL, region=AWS_REGION, collection=self.collection_name)
        return d


def load_config(path_or_name: str) -> ExperimentConfig:
    """Accept either a path to a YAML file or a bare config name (configs/<name>.yaml)."""
    p = Path(path_or_name)
    if not p.exists():
        p = CONFIG_DIR / (path_or_name if path_or_name.endswith((".yaml", ".yml")) else f"{path_or_name}.yaml")
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path_or_name} (looked at {p})")
    data = yaml.safe_load(p.read_text()) or {}
    data.setdefault("name", p.stem)
    known = {f for f in ExperimentConfig.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {p.name}: {sorted(unknown)}")
    return ExperimentConfig(**data)
