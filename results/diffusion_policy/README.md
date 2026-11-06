# robomimic Diffusion Policy results

This folder holds Diffusion Policy baselines for the robomimic Lift and Can tasks. Each policy was
trained on the proficient-human demonstrations using low-dimensional observations, then evaluated
in simulation. Both reached a 0.98 success rate at epoch 2000 over 50 episodes.

## Quick links

| Task | Model | Videos | Training config |
|---|---|---|---|
| Lift | [`lift/model_epoch_2000.pth`](https://huggingface.co/lukelu33/robomimic-diffusion-policy/resolve/main/lift/model_epoch_2000.pth) | [`lift/videos/`](lift/videos/) | [`lift/config.json`](lift/config.json) |
| Can | [`can/model_epoch_2000.pth`](https://huggingface.co/lukelu33/robomimic-diffusion-policy/resolve/main/can/model_epoch_2000.pth) | [`can/videos/`](can/videos/) | [`can/config.json`](can/config.json) |

Each checkpoint is about 1 GB, so the models are stored on Hugging Face at
[lukelu33/robomimic-diffusion-policy](https://huggingface.co/lukelu33/robomimic-diffusion-policy).

## Folder contents

```text
results/diffusion_policy/
├── README.md
├── lift/
│   ├── config.json
│   └── videos/
│       ├── all_episodes.mp4               # all 10 rollouts joined
│       └── episode_00_success.mp4 ... 09  # one file per rollout
└── can/
    ├── config.json
    └── videos/                            # same layout as lift/videos/
```

## Datasets

Both policies use the `ph` (proficient-human) `low_dim` v1.5 datasets from
[robomimic](https://github.com/ARISE-Initiative/robomimic). Each demonstration is a teleoperated
Panda arm trajectory recorded in robosuite at 20 Hz.

| | Lift | Can |
|---|---|---|
| Environment | `Lift` | `PickPlaceCan` |
| Goal | Lift the cube | Pick up the can and place it in its bin |
| Demonstrations | 200 | 200 |
| Total frames | 9666 | 23207 |
| Dataset file | `lift/ph/low_dim_v15.hdf5` | `can/ph/low_dim_v15.hdf5` |

Download the datasets with robomimic's script:

```bash
python robomimic/scripts/download_datasets.py --tasks lift can --dataset_types ph --hdf5_types low_dim
```

### Observations and actions

| Name | Dimensions | Meaning |
|---|---|---|
| `robot0_eef_pos` | 3 | End-effector position |
| `robot0_eef_quat` | 4 | End-effector orientation |
| `robot0_gripper_qpos` | 2 | Gripper finger positions |
| `object` | 10 (Lift), 14 (Can) | Object state for the task |
| `actions` | 7 | `OSC_POSE` delta command: 3 position, 3 rotation, 1 gripper |

Training used all 200 demonstrations, with no train/validation split. It did not use the dataset's
rewards.

## Training

Each iteration samples 16-step action sequences with 2 frames of observations, adds noise at a
random diffusion timestep, and trains a 1D UNet to predict that noise given the observations. At
inference, the policy denoises Gaussian noise into 16 actions over 100 steps, runs 8 of them, and
then replans.

| Setting | Value |
|---|---|
| Algorithm | Diffusion Policy with a UNet (`down_dims` 256, 512, 1024) |
| Observation / prediction / action horizon | 2 / 16 / 8 steps |
| Noise scheduler | DDPM, 100 train and 100 inference steps, `squaredcos_cap_v2` |
| Prediction target | Noise (`epsilon`) |
| Action normalization | None |
| EMA | Enabled, power 0.75 |
| Optimizer | AdamW, learning rate 1e-4, cosine schedule, 500 warmup steps |
| Batch size | 100 |
| Length | 2000 epochs of 100 steps each (200,000 gradient steps) |
| Seed | 1 |

The two configs differ only in experiment name and dataset path. Training took 10 h 27 min for Lift
and 16 h 15 min for Can on one GPU, including evaluation.

### Software versions

| Package | Version |
|---|---|
| Python | 3.10.22 |
| robomimic | 0.5.0 (commit `d309eae`) |
| robosuite | 1.5.1 |
| MuJoCo | 3.2.3 |
| PyTorch | 2.4.0+cu121 |
| diffusers | 0.11.1 |

## Results

During training, each policy ran 50 simulated episodes every 50 epochs, with a 400-step limit per
episode.

| Task | Success rate at epoch 2000 | Mean, epochs 1000-2000 | Range, epochs 1000-2000 |
|---|---|---|---|
| Lift | 0.98 | 0.994 | 0.96-1.00 |
| Can | 0.98 | 0.968 | 0.92-1.00 |

With 50 episodes, one failure changes the rate by 0.02, so small differences between evaluations
are mostly noise.

### Videos

The videos come from the epoch 2000 checkpoints above. They show the agentview camera at 480x480,
with rollout seed 1234.

| Task | Successful rollouts | Steps per rollout |
|---|---|---|
| Lift | 10 of 10 | 39-50 |
| Can | 10 of 10 | 96-131 |

These 10 rollouts are for viewing. Use the 50-episode results above to judge performance.

## Using the models

Install robomimic and robosuite at the versions above, then load a checkpoint:

```python
from huggingface_hub import hf_hub_download
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

path = hf_hub_download("lukelu33/robomimic-diffusion-policy", "can/model_epoch_2000.pth")
device = TorchUtils.get_torch_device(try_to_use_cuda=True)
policy, ckpt = FileUtils.policy_from_checkpoint(ckpt_path=path, device=device, verbose=False)
env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt, render=False, render_offscreen=True, verbose=False)

policy.start_episode()
obs = env.reset()
action = policy(ob=obs)
```

Checkpoints are pickled PyTorch files, so only load them from a source you trust.

## Reproducing training

The configs contain paths from the original training machine. Before training, change these fields
to match your setup:

- `train.data[0].path`: the dataset file
- `train.output_dir`: where to save results

Then run:

```bash
python robomimic/scripts/train.py --config results/diffusion_policy/can/config.json
```

Each saved checkpoint is about 1 GB, and these configs save one every 50 epochs, so a run needs
about 42 GB of disk space. Set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` for rendering on a
headless machine.
