# OSWorld evaluation

Install OSWorld separately and verify its Docker provider before connecting a
model server. The standalone runner accepts any compatible checkout through
`--osworld-root`.

## Smoke test

Create a one-task manifest or select an existing small manifest, then run:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=YOUR_API_KEY

./scripts/run_smoke_test.sh \
  ../OSWorld \
  ../OSWorld/evaluation_examples/test_small.json \
  ./results/smoke
```

## Comparable benchmark

For comparison with another model, keep all non-model variables fixed:

- identical task manifest and exclusions
- identical OSWorld commit and Docker image
- 1920×1080 screen
- identical `max_steps`, reset wait, evaluator wait, and action delay
- temperature 0
- the same number of task retries
- separate Google Drive runs if account state is shared

Example:

```bash
osworld-cogagent \
  --osworld-root ../OSWorld \
  --provider_name docker \
  --headless \
  --test_all_meta_path ../OSWorld/evaluation_examples/test_nogdrive.json \
  --model cogagent-9b-20241220 \
  --temperature 0 \
  --top_p 1 \
  --max_tokens 1024 \
  --history_n 15 \
  --loop_action_limit 3 \
  --format_failure_limit 3 \
  --completion_verification \
  --sleep_after_execution 3 \
  --screen_width 1920 \
  --screen_height 1080 \
  --max_steps 15 \
  --num_envs 2 \
  --result_dir ./results/cogagent9b
```

The runner resumes automatically by skipping task directories containing a
`result.txt`. It writes `args.json`, trajectories, screenshots, recordings,
termination provenance, and scores below the result directory.

Do not publish raw trajectories or screenshots until they have been checked
for account names, Drive content, tokens, and other private data.
