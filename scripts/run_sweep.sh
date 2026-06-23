#!/usr/bin/env bash
# Drive the full experiment sweep end-to-end. Resumable: skips ingests whose collection
# already exists and evals whose result file already exists. Safe to re-run after a crash.
#
#   bash scripts/run_sweep.sh
#
# Env: PY (python interpreter), LIMIT (optional --limit N for a faster partial pass).
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-$HOME/miniconda3/envs/personal/bin/python}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"

CONFIGS=(baseline chunk256 chunk512 chunk2000 k2 k8 prompt_permissive)

echo "### Sweep start. PY=$PY  LIMIT=${LIMIT:-none}"

# 0. require the frozen test set
if [ ! -f testset/testset.json ]; then
  echo "!!! testset/testset.json missing. Generate it first: \$PY -m src.testset --n 150"
  exit 1
fi

# 1. ingest each config (ingest itself reuses an existing collection unless --rebuild)
for c in "${CONFIGS[@]}"; do
  echo "### ingest: $c"
  "$PY" -m src.ingest --config "$c" 2>&1 | grep -vE "it/s\]$|^\s*$" | tail -3
done

# 2. eval each config (skip if result already exists — immutable)
for c in "${CONFIGS[@]}"; do
  if [ -f "results/$c.json" ]; then
    echo "### eval: $c — SKIP (results/$c.json exists)"
    continue
  fi
  echo "### eval: $c"
  "$PY" -m src.run_eval --config "$c" $LIMIT_ARG 2>&1 | grep -E "Running|Scoring|faithful|answer_rel|context_rec|Saved|Error|Traceback"
done

# 3. compare everything
echo "### compare"
"$PY" -m src.compare --all --baseline baseline --slice synthesizer 2>&1 | tail -60

echo "### Sweep done."
