# OSWorld-CogAgent

An unofficial, provider-neutral CogAgent-9B adapter, OpenAI-compatible model
server, and reproducible multi-environment runner for
[OSWorld](https://github.com/xlang-ai/OSWorld).

This project integrates
[`zai-org/cogagent-9b-20241220`](https://huggingface.co/zai-org/cogagent-9b-20241220)
with OSWorld. It does **not** redistribute model weights and is not affiliated
with Z.ai or the OSWorld maintainers.

## Features

- Native CogAgent `Task → History → Platform → Status-Action-Operation` prompt
- Strict AST-based parser for official `Grounded Operation` outputs
- Normalized box-to-screen coordinate conversion
- Bounded malformed-output recovery
- Static-screen repeated-action guard
- Independent completion audit before accepting `END`
- OpenAI-compatible BF16 server for RunPod, lab servers, or local NVIDIA hosts
- Serialized GPU generation with concurrent OSWorld environment preparation
- Standalone OSWorld runner with no Qwen agent dependency

## Architecture

```text
OSWorld checkout + Docker environments
                 │
                 │ OpenAI-compatible HTTP API
                 ▼
        OSWorld-CogAgent server
    (RunPod / lab GPU / local NVIDIA)
```

Changing inference providers requires only `OPENAI_BASE_URL`,
`OPENAI_API_KEY`, and optionally `MODEL_PATH`; agent code does not change.

## Requirements

- Python 3.10+
- An external OSWorld checkout for evaluation
- CogAgent BF16 inference: a CUDA GPU with at least 29 GB VRAM according to
  the upstream documentation; 40 GB or more is recommended for headroom

## Install the client and runner

```bash
git clone https://github.com/xlang-ai/OSWorld.git
git clone https://github.com/issumer-ku/OSWorld-CogAgent.git

cd OSWorld-CogAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install OSWorld itself according to its upstream documentation, including the
Docker provider and benchmark image.

## Start the model server

On a RunPod, lab server, or local CUDA workstation:

```bash
python3 -m venv .venv-server
source .venv-server/bin/activate
pip install -e ".[server]"

cp profiles/lab.env.example profiles/lab.env
# Edit MODEL_PATH, HF_HOME, COGAGENT_API_KEY, and CUDA_VISIBLE_DEVICES.
./scripts/serve.sh profiles/lab.env
```

`MODEL_PATH` accepts either form:

```bash
MODEL_PATH=zai-org/cogagent-9b-20241220
MODEL_PATH=/data/models/cogagent-9b-20241220
```

Verify the server:

```bash
curl http://127.0.0.1:8000/health
curl -H "Authorization: Bearer $COGAGENT_API_KEY" \
  http://127.0.0.1:8000/v1/models
```

The server serializes `model.generate` calls. `--num_envs 2` can overlap
OSWorld reset/setup time while avoiding simultaneous generation and VRAM
spikes on one model replica.

## Connect OSWorld

```bash
export OPENAI_BASE_URL=http://LAB_SERVER_IP:8000/v1
export OPENAI_API_KEY=YOUR_API_KEY

osworld-cogagent \
  --osworld-root ../OSWorld \
  --provider_name docker \
  --headless \
  --model cogagent-9b-20241220 \
  --test_all_meta_path ../OSWorld/evaluation_examples/test_nogdrive.json \
  --temperature 0 \
  --max_tokens 1024 \
  --history_n 15 \
  --max_steps 15 \
  --num_envs 2 \
  --result_dir ./results/cogagent9b
```

Use `--num_envs 1` for the first smoke test. With RunPod's HTTP proxy, keep
queued request time below its proxy timeout; a lab server or SSH tunnel does
not have that proxy-specific limitation.

## Deployment guides

- [RunPod](docs/runpod.md)
- [Lab GPU server](docs/lab-server.md)
- [SSH tunnel](docs/ssh-tunnel.md)
- [OSWorld evaluation](docs/osworld-evaluation.md)

## Supported operations

`CLICK`, `DOUBLE_CLICK`, `RIGHT_CLICK`, `HOVER`, `TYPE`, `SCROLL_UP`,
`SCROLL_DOWN`, `SCROLL_LEFT`, `SCROLL_RIGHT`, `KEY_PRESS`, `KEY_DOWN`,
`KEY_UP`, `GESTURE`, `LAUNCH`, and `END`.

`QUOTE_TEXT`, `QUOTE_CLIPBOARD`, and `LLM` are rejected because they require
the upstream CogAgent application's external OCR/variable runtime.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q src server tests
python -m build
```

## Attribution and license

Project code is released under Apache-2.0. See [NOTICE](NOTICE) for upstream
attribution. CogAgent model weights remain subject to the model's own license.
