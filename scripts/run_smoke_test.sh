#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 OSWORLD_ROOT MANIFEST [RESULT_DIR]" >&2
  exit 2
fi

: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL to the model server /v1 URL}"
: "${OPENAI_API_KEY:=dummy}"
export OPENAI_API_KEY

OSWORLD_ROOT=$1
MANIFEST=$2
RESULT_DIR=${3:-./results/cogagent-smoke}

exec osworld-cogagent \
  --osworld-root "$OSWORLD_ROOT" \
  --provider_name docker \
  --headless \
  --model cogagent-9b-20241220 \
  --test_all_meta_path "$MANIFEST" \
  --temperature 0 \
  --max_tokens 1024 \
  --history_n 15 \
  --max_steps 15 \
  --num_envs 1 \
  --result_dir "$RESULT_DIR"
