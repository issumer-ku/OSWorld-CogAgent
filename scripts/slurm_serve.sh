#!/usr/bin/env bash
#SBATCH --job-name=cogagent-server
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/cogagent-%j.log

set -euo pipefail
mkdir -p logs

PROFILE=${1:-profiles/lab.env}
set -a
# shellcheck disable=SC1090
source "$PROFILE"
set +a

exec cogagent-openai-server
