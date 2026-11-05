# Flow Matching BC for Low-Dimensional Lift

This branch adds a conditional flow matching policy to the official robomimic training framework. It learns from demonstrations using behavior cloning (BC), then evaluates the frozen policy in the Lift simulator.

## Results

| Experiment | Training | Independent evaluation |
|---|---|---|
| Initial pilot (legacy framework) | 100 epochs, batch size 64, constant learning rate, seed 0 | 99/100 successes |
| Aligned experiment (official framework) | 2000 epochs, batch size 100, cosine schedule, seed 1 | **100/100 successes** |

The aligned model's mean independent rollout length was **43.88 steps**. Its final in-training evaluation achieved 50/50 successes, with a mean length of 44.02 steps.

[Open the aligned results, plots, tables, and video](docs/results/lift_flow_seed1_2000/README.md).

These results come from one training seed per experiment. The five independent evaluation seeds are five groups of tests of the same model, not five independently trained policies. The pilot and aligned runs use different training settings; their success rates do not isolate the effect of training longer. No Diffusion comparison result is reported here yet.

## Framework and source files

The bundled `robomimic/` sources are pinned to [ARISE-Initiative/robomimic commit d309eaecc18acf4152a830a895a6984b8ac71b05](https://github.com/ARISE-Initiative/robomimic/tree/d309eaecc18acf4152a830a895a6984b8ac71b05), version 0.5.0. The official `train.py`, data loading, scheduler utilities, checkpoint handling, and evaluation script are retained. See [framework provenance](docs/robomimic_upstream.json).

| Upstream file | Flow adaptation |
|---|---|
| `robomimic/algo/diffusion_policy.py` | `robomimic/robomimic/algo/flow_policy.py` |
| `robomimic/models/diffusion_policy_nets.py` | `robomimic/robomimic/models/flow_policy_nets.py` |
| `robomimic/config/diffusion_policy_config.py` | `robomimic/robomimic/config/flow_policy_config.py` |

The [algorithm diff](docs/flow_policy_vs_diffusion.patch) records the changes relative to the pinned Diffusion implementation. Original license files are retained.

Only the BC training and evaluation path has been validated. The original Q2RL online training programs have not been revalidated against the upgraded framework. Flow has not been integrated into Q2RL's Q estimation or Q gating. Use the separate legacy q2rl checkout and environment for the earlier online experiments.

## Algorithm

Given a demonstration action sequence `actions`, sample Gaussian `noise` and uniform time `t` in `[0, 1]`:

```python
x_t = (1 - t) * noise + t * actions
target_velocity = actions - noise
loss = mse(velocity_network(x_t, t, observation), target_velocity)
```

At inference, start from Gaussian noise and use 10 Euler integration steps from `t=0` to `t=1`. Time embeddings use `t * 1000`; the integration interval remains `[0, 1]`. Generated actions are clipped to `[-1, 1]`.

The policy retains the observation encoder, ConditionalUnet1D, action chunks, EMA, and checkpoint interfaces. Its U-Net supports configurable dimensions, and scalar time inputs retain floating-point precision. Both Flow and Diffusion use the EMA implementation from **diffusers 0.11.1**. The old vendored `flow_ema.py` remains for historical reference and is not imported by the current policy.

Flow accepts both single-frame observations through a history queue and frame-stacked observations from the official FrameStackWrapper.

## Paired training configurations

- Flow: [`experiments/configs/flow/lift_low_dim.json`](experiments/configs/flow/lift_low_dim.json)
- Diffusion reference: [`experiments/configs/flow/lift_diffusion_reference.json`](experiments/configs/flow/lift_diffusion_reference.json)
- Original supplied Diffusion template: [`docs/reference/diffusion_policy_user_template.json`](docs/reference/diffusion_policy_user_template.json)
- Legacy pilot configuration: `experiments/configs/flow/lift_low_dim_pilot_legacy.json`, intended for the pre-migration implementation at commit `d81d1f2`.

The paired configurations share the following settings:

| Setting | Value |
|---|---|
| Dataset | Lift / proficient-human (`ph`) / `low_dim_v141.hdf5` |
| Demonstrations | 200; no demo filter or held-out validation split in the supplied template |
| Observation keys | `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `object` |
| Observation / prediction / action horizon | 2 / 16 / 8 |
| U-Net widths / time embedding dimension | `[256, 512, 1024]` / 256 |
| Training budget | 2000 epochs x 100 updates = 200,000 BC parameter updates |
| Batch size / training seed | 100 / 1 |
| Optimizer | AdamW, learning rate `1e-4`, L2 regularization `1e-6` |
| Learning-rate schedule | 500-update warmup, then half-cycle cosine decay; stepped each batch |
| In-training evaluation | 50 episodes every 50 epochs, maximum 400 steps, stop on success |
| EMA | Enabled, power 0.75 |
| Checkpoints | Every 50 epochs, best evaluation success, and `last.pth` / `last_bak.pth` |

They differ only in algorithm selection, experiment/output names, and DDPM/DDIM versus Flow settings. The Diffusion reference uses DDPM with 100 inference steps, whereas Flow uses 10 Euler steps. This does not match inference computation; report network evaluations or measured latency separately when comparing speed.

The training-curve horizontal axis is **BC parameter updates**, not environment steps. An epoch here is a fixed number of updates rather than necessarily one complete dataset pass.

For comparisons, fix the same robomimic commit, dataset checksum, simulator versions, evaluation seeds, and checkpoint-selection rule. Use the final epoch-2000 checkpoint for both methods rather than selecting a best checkpoint for only one.

## Oscar setup and training

Run these commands from the project root on Oscar. The setup script creates this checkout's own `.venv`, using Python 3.10 and the pinned dependencies in `scripts/flow/requirements-oscar.txt`:

```bash
bash scripts/flow/setup_oscar.sh
mkdir -p logs
sbatch --time=00:30:00 scripts/flow/run_aligned.sbatch smoke
```

The dataset should already be at:

```text
data/lift/ph/low_dim_v141.hdf5
```

The smoke test uses the full U-Net and batch size 100, but runs only two epochs of three updates and two Lift rollouts per epoch. It checks GPU, simulator, video, and checkpoint integration; it does not estimate trained performance.

After a successful smoke test:

```bash
sbatch scripts/flow/run_aligned.sbatch train
```

Each job requests one GPU. The 12-hour limit is a wall-time ceiling. Outputs are stored under `results/flow_aligned_<mode>_<jobid>/`, including a resolved configuration and a manifest containing the dataset SHA256, dependency versions, Python version, and GPU model. Previous experiments are retained.

The simulator dependencies are robosuite 1.4.1 and MuJoCo 3.1.6. The Flow branch's setup script is separate from the root Q2RL installation instructions and requirements.

## Recovery from interrupted training

The aligned run's initial job, 8357773, stopped during the epoch-350 evaluation with a model-resource read error. The resource existed afterward. This does not establish whether the cause was file-handle exhaustion or another resource-access issue.

The last complete checkpoint was epoch 349. Job 8367567 restored the model, optimizer, scheduler, and EMA and completed epochs 350–2000. The resume script raised the soft file limit to the existing hard limit of 131072. This recovery succeeded, but does not prove the original failure's cause.

To resume this run:

```bash
sbatch scripts/flow/resume_aligned.sbatch results/flow_aligned_train_8357773
```

The script checks the original dataset hash, software versions, saved epoch, optimizer, scheduler, and EMA state. It retains the original configuration and output directory, backs up logs/configuration that the official resume path would overwrite, and writes recovery diagnostics to `results/flow_resume_<jobid>/`.

Resume does not guarantee the same RNG sequence or data order as an uninterrupted run. New experiments start from scratch; do not initialize them with the old pilot weights.

## Independent final evaluation and report

The completed final evaluation is job **8401742**. It reloaded `model_epoch_2000.pth` and tested seeds **1000–1004**, 20 episodes per seed, with a 400-step horizon. Every group achieved 20/20 successes. No policy updates occurred.

To reproduce the evaluation for this run:

```bash
sbatch scripts/flow/evaluate_final.sbatch
```

Each evaluation seed runs in a fresh process. The script requests one GPU with a one-hour wall-time limit and exports to `results/flow_final_eval_<jobid>/report/`:

- `independent_evaluation.csv` and `independent_summary.json`: independent evaluation results.
- `training_metrics.csv` and `evaluation_metrics.csv`: merged training logs. Later complete records replace duplicate epochs after resume; partial failed evaluations are excluded.
- `lift_success.png/pdf` and `training_loss.png/pdf`: curves for one training seed, without a cross-training-seed confidence band.
- `Lift_epoch_2000.mp4`: the final **in-training** evaluation video, not a recording of the independent 100 episodes.

The adjacent `evaluation_manifest.json` and per-seed logs remain on Oscar. The report summary includes the evaluated checkpoint SHA256; model weights and datasets are not included in Git.

## Validation

With the project environment active:

```bash
export PYTHONPATH="$PWD/robomimic${PYTHONPATH:+:$PYTHONPATH}"
python robomimic/tests/test_flow_policy.py
python robomimic/tests/test_run_trained_agent_wrappers.py
python robomimic/tests/test_flow_upstream.py
python scripts/flow/test_export_results.py
```

Coverage includes Flow targets, action/history alignment, model/optimizer/EMA restoration, wrapped-environment rollouts, paired configuration equality, warmup/cosine scheduling, both policies through the official `train.py` with synthetic HDF5, Flow CLI resume, and resumed-log/result export. Oscar smoke tests, the completed full training run, and independent evaluation provide simulator validation for the aligned setup.
