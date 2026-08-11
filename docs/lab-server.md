# Lab GPU server

The same server process used on RunPod runs on a lab server. No provider SDK is
required.

## Install

```bash
git clone https://github.com/issumer-ku/OSWorld-CogAgent.git
cd OSWorld-CogAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[server]"
```

## Configure

```bash
cp profiles/lab.env.example profiles/lab.env
```

Edit the profile:

```bash
MODEL_PATH=/data/models/cogagent-9b-20241220
HF_HOME=/data/huggingface
COGAGENT_API_KEY=REPLACE_WITH_A_RANDOM_SECRET
CUDA_VISIBLE_DEVICES=0
HOST=0.0.0.0
PORT=8000
```

`MODEL_PATH=zai-org/cogagent-9b-20241220` downloads from Hugging Face instead.

## Run

```bash
./scripts/serve.sh profiles/lab.env
```

For a persistent shell session, use the lab's process manager, `tmux`, or the
provided Slurm template:

```bash
cp profiles/lab.env.example profiles/lab.env
sbatch scripts/slurm_serve.sh profiles/lab.env
```

Test from a machine that can reach the lab network:

```bash
curl http://LAB_SERVER_IP:8000/health
curl -H "Authorization: Bearer $COGAGENT_API_KEY" \
  http://LAB_SERVER_IP:8000/v1/models
```

Keep the API key enabled on any shared network and restrict port 8000 with the
lab firewall. If inbound ports are unavailable, use the SSH tunnel guide.
