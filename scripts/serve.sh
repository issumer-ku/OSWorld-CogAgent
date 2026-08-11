#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [profile.env]" >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$1"
  set +a
fi

exec cogagent-openai-server
