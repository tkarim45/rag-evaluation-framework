"""Bedrock model factories + RAGAS wrappers.

Everything that talks to AWS Bedrock funnels through here so model ids, region, and
temperature are set in exactly one place. temperature=0 everywhere for reproducibility.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import boto3
from botocore.config import Config as BotoConfig
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

from . import config as C

# Bedrock on-demand throughput throttles under bursty load. Adaptive retry + a deep retry
# budget lets the client ride out ThrottlingExceptions instead of failing the whole run.
_BOTO_CFG = BotoConfig(
    region_name=C.AWS_REGION,
    retries={"max_attempts": 12, "mode": "adaptive"},
    read_timeout=120,
    connect_timeout=15,
)


@lru_cache(maxsize=1)
def runtime_client():
    """Shared bedrock-runtime client with throttle-resilient retry config."""
    return boto3.client("bedrock-runtime", config=_BOTO_CFG)


def chat(model_id: str, *, temperature: float = 0.0, max_tokens: int = C.GEN_MAX_TOKENS) -> ChatBedrockConverse:
    """A Bedrock Converse chat model. `model_id` must be an inference-profile id for Claude."""
    return ChatBedrockConverse(
        model=model_id,
        client=runtime_client(),
        temperature=temperature,
        max_tokens=max_tokens,
    )


class ParallelBedrockEmbeddings(BedrockEmbeddings):
    """Titan v2 only embeds one text per request, so embed_documents is otherwise serial.
    Fan the per-chunk calls across a thread pool — ingest of a chunk256 store drops from
    ~thousands of serial calls to a few hundred seconds. Single-query path is unchanged.
    """

    embed_workers: int = C.MAX_WORKERS

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with ThreadPoolExecutor(max_workers=self.embed_workers) as pool:
            return list(pool.map(self.embed_query, texts))


@lru_cache(maxsize=4)
def embeddings(model_id: str = C.EMBED_MODEL) -> ParallelBedrockEmbeddings:
    """Bedrock embeddings (Titan v2 by default). Cached so ingest + eval share one client."""
    return ParallelBedrockEmbeddings(model_id=model_id, client=runtime_client())


def gen_llm(model_id: str) -> ChatBedrockConverse:
    """The pipeline-under-test answer model."""
    return chat(model_id)


def ragas_run_config() -> RunConfig:
    """Throttle-safe concurrency for RAGAS (testset gen + evaluate)."""
    return RunConfig(max_workers=C.MAX_WORKERS, timeout=180, max_retries=10, max_wait=60)


def ragas_judge() -> LangchainLLMWrapper:
    """RAGAS judge LLM — FIXED project-wide (src.config.JUDGE_MODEL)."""
    return LangchainLLMWrapper(chat(C.JUDGE_MODEL, max_tokens=2048), run_config=ragas_run_config())


def ragas_embeddings(model_id: str = C.EMBED_MODEL) -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(embeddings(model_id), run_config=ragas_run_config())
