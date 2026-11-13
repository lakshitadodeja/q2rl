# FMP checkpoints: Lift v15, 20% data

This directory is on the `Flow-Matching` branch. It contains model metadata and a downloader. Original checkpoint binaries are hosted as assets in the [dedicated GitHub Release](https://github.com/lakshitadodeja/q2rl/releases/tag/fmp-v15-20pct-seed1-8957390), not as Git LFS files in this branch. GitHub rejected new LFS object uploads to this public fork.

| Checkpoint | Epoch | BC gradient updates | Training-time success rate |
|---|---:|---:|---:|
| [Early: epoch 8](https://github.com/lakshitadodeja/q2rl/releases/download/fmp-v15-20pct-seed1-8957390/model_epoch_8_low_dim_v15_success_0.58.pth) | 8 | 800 | 58% (29/50) |
| [Final: epoch 2000](https://github.com/lakshitadodeja/q2rl/releases/download/fmp-v15-20pct-seed1-8957390/model_epoch_2000.pth) | 2000 | 200,000 | 100% (50/50) |

One training run, seed 1, 40 demonstrations selected by `20_percent`, Oscar job 8957390. Every training-time evaluation used 50 rollouts, horizon 400. Neither checkpoint has a reported independent 100-episode evaluation.

## Download

From this directory, using Python 3 and an internet connection:

```bash
python3 download_models.py --checkpoint early
python3 download_models.py --checkpoint final
```

Or download both with `python3 download_models.py`. Each checkpoint is approximately 1 GB. The script prints progress, verifies size and SHA256 before saving, and skips existing files only if their hashes match. No Git LFS installation is needed. After downloading both, verify the original model and metadata checksums with `sha256sum -c SHA256SUMS`.

## Configuration and loading

`config.json`, `manifest.json`, `subset.json`, and `evaluation_metrics.csv` preserve the actual configuration, software versions, exact demonstration IDs, and all 89 observed training-time evaluation points. `RELEASE_NOTES.md` is the original release description. `SHA256SUMS` is the original release checksum file.

Loading requires the custom `flow_policy` implementation and matching environment: robosuite 1.5.1, MuJoCo 3.2.3, torch 2.9.0, numpy 1.26.4. From a compatible repository root with that environment activated:

```bash
export PYTHONPATH="$PWD/robomimic${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python robomimic/robomimic/scripts/run_trained_agent.py \
  --agent pretrained_models/lift_flow_v15_20pct_seed1/model_epoch_2000.pth \
  --n_rollouts 20 --horizon 400 --seed 1000
```

The downloaded weights are the original, unmodified checkpoints. This metadata commit does not independently verify that every branch source file equals the original Oscar training source; see the manifest for recorded provenance.
