#!/usr/bin/env bash
set -euo pipefail

DESTINATION=${1:-external/OSWorld}
REF=${OSWORLD_REF:-main}
URL=${OSWORLD_REPOSITORY:-https://github.com/xlang-ai/OSWorld.git}

if [[ -e "$DESTINATION" ]]; then
  echo "destination already exists: $DESTINATION" >&2
  exit 1
fi

git clone --branch "$REF" --depth 1 "$URL" "$DESTINATION"
echo "OSWorld cloned to $DESTINATION"
echo "Follow the upstream OSWorld setup guide before running evaluations."
