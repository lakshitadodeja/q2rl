# FMP checkpoints: Lift, 20% demonstrations

One training run using 40 of 200 Lift / PH low-dimensional v15 demonstrations (the existing `20_percent` mask), training seed 1, Oscar job 8957390.

| Checkpoint | Epoch | BC gradient updates | Training-time success rate |
|---|---:|---:|---:|
| Early | 8 | 800 | 58% (29/50) |
| Final | 2000 | 200,000 | 100% (50/50) |

Each training-time evaluation used 50 rollouts, horizon 400. Neither attached checkpoint has a reported independent 100-episode evaluation. The early checkpoint is the only recorded point in the inclusive 50–60% target interval for this run.

The `.pth` assets are original, unmodified training checkpoints. `SHA256SUMS` identifies their exact bytes and the attached metadata. `config.json`, `manifest.json`, `subset.json`, and `evaluation_metrics.csv` document the configuration, environment, exact demonstration IDs, and all 89 training-time evaluations. The dataset itself is not included.

Loading requires the custom `flow_policy` implementation. Use the corresponding Flow-Matching code and the environment recorded in `manifest.json`: robosuite 1.5.1, MuJoCo 3.2.3, torch 2.9.0, numpy 1.26.4. Keep the recorded EMA, normalization and sampling settings when loading the checkpoint. From the compatible repository root, with that environment activated:

```bash
export PYTHONPATH="$PWD/robomimic${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python robomimic/robomimic/scripts/run_trained_agent.py \
  --agent /path/to/downloaded_checkpoint.pth --n_rollouts 20 --horizon 400 --seed 1000
```

The release tag points to the repository's Flow-Matching branch when the release is created; it is not an independent verification of the original Oscar source tree. The training source provenance is recorded in the attached manifest.
