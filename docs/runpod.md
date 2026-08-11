# RunPod

Deploy one GPU with at least 40 GB VRAM for practical BF16 headroom. Allocate
80 GB or more at `/workspace` for model weights, dependencies, and cache. A
network volume preserves the Hugging Face cache across Pod replacement.

Expose internal HTTP port `8000`. RunPod's external URL has this form:

```text
https://POD_ID-8000.proxy.runpod.net
```

## Install and run

```bash
cd /workspace
git clone https://github.com/issumer-ku/OSWorld-CogAgent.git
cd OSWorld-CogAgent

python3 -m venv /workspace/.venv-cogagent
source /workspace/.venv-cogagent/bin/activate
pip install -e ".[server]"

cp profiles/runpod.env.example profiles/runpod.env
# Set a strong COGAGENT_API_KEY in profiles/runpod.env.
./scripts/serve.sh profiles/runpod.env
```

Wait for model loading, then test inside the Pod:

```bash
curl http://127.0.0.1:8000/health
curl -H "Authorization: Bearer $COGAGENT_API_KEY" \
  http://127.0.0.1:8000/v1/models
```

From the OSWorld host:

```bash
export OPENAI_BASE_URL=https://POD_ID-8000.proxy.runpod.net/v1
export OPENAI_API_KEY=YOUR_API_KEY
```

RunPod's HTTP proxy has a request-duration limit. Warm the model inside the Pod
before benchmarking and begin with one environment. Use two only when the
queued inference latency stays comfortably below the proxy limit.
