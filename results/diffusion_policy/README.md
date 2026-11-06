# robomimic Diffusion Policy results

Diffusion Policy (UNet) trained with robomimic 0.5.0 (commit `d309eae`) on low-dimensional
observations from the proficient-human (`ph`) v1.5 datasets. Both runs used 2000 epochs and seed 1.

| Task | Environment | Demos | Success rate at epoch 2000 (50 episodes) | Videos |
|---|---|---|---|---|
| Lift | `Lift` | 200 | 0.98 | [`lift/videos/`](lift/videos/) (epoch 2000) |
| Can | `PickPlaceCan` | 200 | 0.98 | [`can/videos/`](can/videos/) (epoch 2000) |

Each video folder has 10 rollouts plus `all_episodes.mp4`, which joins them. All 10 succeeded in
both folders. The videos come from the same epoch 2000 checkpoints linked below.

The training configuration for each run is in `lift/config.json` and `can/config.json`. Paths in
those files refer to the original training machine.

## Model weights

Each checkpoint is about 1 GB, so they are stored on Hugging Face:

[lukelu33/robomimic-diffusion-policy](https://huggingface.co/lukelu33/robomimic-diffusion-policy)

- [`lift/model_epoch_2000.pth`](https://huggingface.co/lukelu33/robomimic-diffusion-policy/resolve/main/lift/model_epoch_2000.pth)
- [`can/model_epoch_2000.pth`](https://huggingface.co/lukelu33/robomimic-diffusion-policy/resolve/main/can/model_epoch_2000.pth)
