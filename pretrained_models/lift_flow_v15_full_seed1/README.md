# FMP checkpoints: Lift v15, full data

Training job 8946875 used all 200 Lift PH low-dimensional v15 demonstrations, seed 1, 2000 epochs x 100 gradient updates. robosuite 1.5.1, MuJoCo 3.2.3; see manifest.json.

| Checkpoint | Epoch | Updates | Training-time evaluation | Independent evaluation |
|---|---:|---:|---:|---|
| Early | 8 | 800 | 54% (27/50) | 54% (54/100), seeds 1000–1004, 20 rollouts each |
| Final | 2000 | 200000 | 100% (50/50) | Not performed |

Training-time evaluations use 50 rollouts and horizon 400. This is one training seed. The original v1.4.1 final-model independent result of 100/100 does not apply to these v15 models. The dataset is not attached; hdf5_filter_key=null selects all demonstrations. SHA256SUMS identifies original unmodified weights and metadata. Keep the matching flow_policy code, EMA and sampling configuration when loading.

The release tag references the current Flow-Matching branch, not an independently verified copy of the historical Oscar training source. Provenance is recorded in manifest.json. The existing 20%-data release is a separate run and remains available.

## Download

From this directory run `python3 download_models.py` for both models, or add `--checkpoint early` / `--checkpoint final`. The script downloads original Release assets and verifies SHA256 and size. After both downloads, use `sha256sum -c SHA256SUMS`. Keep the Release assets available; this directory stores metadata and download code.

[20%-data models](../lift_flow_v15_20pct_seed1/README.md) | [Current v15 results](../../docs/results/lift_flow_v15/README.md)
