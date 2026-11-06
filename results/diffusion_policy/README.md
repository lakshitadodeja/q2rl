# robomimic Diffusion Policy results

Diffusion Policy (UNet) trained with robomimic 0.5.0 (commit `d309eae`) on low-dimensional
observations from the proficient-human (`ph`) v1.5 datasets. Both runs used 2000 epochs and seed 1.

| Task | Environment | Demos | Success rate at epoch 2000 (50 episodes) | Videos |
|---|---|---|---|---|
| Lift | `Lift` | 200 | 0.98 | [`lift/videos/`](lift/videos/) (epoch 1950) |
| Can | `PickPlaceCan` | 200 | 0.98 | [`can/videos/`](can/videos/) (epoch 2000) |

Each video folder has 10 rollouts plus `all_episodes.mp4`, which joins them. All 10 succeeded in
both folders. The Lift videos come from the epoch 1950 checkpoint, not epoch 2000.

The training configuration for each run is in `lift/config.json` and `can/config.json`. Paths in
those files refer to the original training machine.

## Model weights

Each checkpoint is about 1 GB, so they are stored outside this repository.

Hugging Face link: to be added.
