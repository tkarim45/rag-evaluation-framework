"""Fetch a Wikipedia corpus into corpus/ as .txt files (one article per file).

The corpus is the ground source for both the RAG store and the RAGAS synthetic test set,
so it is fetched ONCE and committed. Re-running skips files that already exist.

  python -m src.fetch_corpus                     # default: space-exploration seed list
  python -m src.fetch_corpus --topic "Formula 1" --n 30
"""
from __future__ import annotations

import argparse
import re

import wikipedia

from .config import CORPUS_DIR

# Be polite + dodge the empty-response/JSON errors the API throws under bursty access.
wikipedia.set_rate_limiting(True)

# Curated seed list for the chosen topic (space exploration). Curated > search-scraped:
# guarantees substantial, on-topic articles and a stable, committable corpus.
SPACE_EXPLORATION = [
    "Space exploration", "Apollo program", "Apollo 11", "International Space Station",
    "SpaceX", "Falcon 9", "Saturn V", "NASA", "Hubble Space Telescope",
    "James Webb Space Telescope", "Voyager 1", "Voyager 2", "Mars rover",
    "Curiosity (rover)", "Perseverance (rover)", "Space Shuttle", "Project Gemini",
    "Project Mercury", "Yuri Gagarin", "Vostok 1", "Sputnik 1",
    "Hayabusa2", "Cassini–Huygens", "New Horizons", "Artemis program",
    "European Space Agency", "Roscosmos", "Soyuz (rocket family)", "Kepler space telescope",
    "Tiangong space station",
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def _resolve(title: str):
    """Resolve a title to a page object, trying a couple of fallbacks. None on failure."""
    try:
        return wikipedia.page(title, auto_suggest=False, redirect=True)
    except wikipedia.DisambiguationError as e:
        try:
            return wikipedia.page(e.options[0], auto_suggest=False)
        except Exception:
            return None
    except Exception:
        try:
            return wikipedia.page(title, auto_suggest=True)
        except Exception:
            return None


def fetch_one(title: str, retries: int = 2) -> tuple[str, str] | None:
    """Resolve a title and return (resolved_title, plaintext). None on failure.

    `.content` is lazy and triggers a second API call that can return non-JSON under
    load, so the whole resolve+content path is retried.
    """
    for _ in range(retries + 1):
        page = _resolve(title)
        if page is None:
            continue
        try:
            text = page.content.strip()
        except Exception:
            continue
        if len(text) > 500:
            return page.title, text
    return None


def fetch_corpus(titles: list[str]) -> int:
    written = 0
    for title in titles:
        path = CORPUS_DIR / f"{_slug(title)}.txt"
        if path.exists():
            print(f"  skip (exists): {path.name}")
            continue
        res = fetch_one(title)
        if res is None:
            print(f"  FAIL: {title!r}")
            continue
        resolved, text = res
        path.write_text(f"# {resolved}\n\n{text}\n", encoding="utf-8")
        print(f"  wrote {path.name}  ({len(text):,} chars)  <- {resolved}")
        written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a Wikipedia corpus into corpus/")
    ap.add_argument("--topic", help="Search topic; if omitted, uses the curated space-exploration list")
    ap.add_argument("--n", type=int, default=30, help="Number of articles when --topic is given")
    args = ap.parse_args()

    if args.topic:
        print(f"Searching Wikipedia for {args.topic!r} (top {args.n})...")
        titles = wikipedia.search(args.topic, results=args.n)
    else:
        print("Using curated space-exploration seed list.")
        titles = SPACE_EXPLORATION

    n = fetch_corpus(titles)
    total = len(list(CORPUS_DIR.glob("*.txt")))
    print(f"\nDone. {n} new file(s). Corpus now holds {total} article(s) in {CORPUS_DIR}.")


if __name__ == "__main__":
    main()
