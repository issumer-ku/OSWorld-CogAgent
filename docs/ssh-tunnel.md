# SSH tunnel

Use an SSH tunnel when the lab server cannot expose port 8000 or when transport
encryption is required.

Start the model server bound to loopback:

```bash
HOST=127.0.0.1 PORT=8000 ./scripts/serve.sh profiles/lab.env
```

On the OSWorld machine, keep this command running:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@LAB_SERVER
```

Point the runner at the tunnel:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=YOUR_API_KEY
```

No agent or model-server code changes are needed when switching between a
direct lab URL, an SSH tunnel, and RunPod.
