# Q2RL with a Diffusion Policy base (robomimic Lift, state)

Q2RL on robomimic Lift (low-dim state) with a robomimic **Diffusion Policy** as the BC policy,
instead of the original GMM. This folder has everything needed to run another seed on Oscar.
The run in progress uses `SEED=0`; use different seeds (e.g. 1 and 2) for the extra runs.

## Setup

| Item | Setting |
|---|---|
| Task | robomimic Lift, low-dim state |
| Base policy | Diffusion Policy trained on the full proficient-human data (200 demos), **epoch 8** |
| Base success rate | **53% (16/30)**, base policy alone, same execution as in Q2RL (below), 300-step episodes |
| Diffusion inference | robomimic default: DDPM, 100 denoising steps; 2 observation frames in, 16 actions predicted, **only the first action executed, re-predicted every step** (`EXEC_HORIZON=1`) |
| BC log-prob / entropy | Phase 1 uses V_BC only, so both are returned as 0 (marked `TODO(phase 1, V_BC only)` in the code) |
| Demo buffer | not used (`--get_demo_buffer=False`); 100 initial BC rollouts are still collected |
| Q2RL hyperparameters | as in the original lift-state `launch_q2rl_without_data.sh`: 20k Q-estimation steps, 500k online steps, utd 4, batch 256, reward scale 5, reward bias -1, bc_weight 0.3 |
| Evaluation | every 20k online steps, 20 episodes |
| Logging | W&B off (`--debug=True`); every metric goes to `metrics.jsonl` in the run directory |

The original checkpoint file name says `success_0.52`: that was robomimic's in-training evaluation, which
executes 8 actions per prediction. It is not the setting used here; the 53% above is.
All eight full-data checkpoints screened this way are in
[`base_policy_eval/full_data_exec1_screen.json`](base_policy_eval/full_data_exec1_screen.json).

## Code

| File | What it does |
|---|---|
| [`main/q2rl/q2rl_robosuite_diffusion.py`](../main/q2rl/q2rl_robosuite_diffusion.py) | Diffusion version of `main/q2rl/q2rl_robosuite.py`; changes are marked `DIFFUSION:` |
| [`backend/common/dp_bc_query.py`](../backend/common/dp_bc_query.py) | Wraps the DP checkpoint as Q2RL's BC policy: frame stacking, execution horizon, `(action, 0, 0)` |
| [`backend/common/resume_state.py`](../backend/common/resume_state.py) | Resume an online run in a later job: agent and optimizer, online replay buffer, step, RNG states |
| [`backend/agents/q2rl_robosuite.py`](../backend/agents/q2rl_robosuite.py) | Only TODO comments at the two log-prob/entropy uses |
| [`run_q2rl_dp.sbatch`](run_q2rl_dp.sbatch) | Slurm script: `MODE=smoke` (pipeline check) and `MODE=full` |
| [`eval_dp_exec_horizon.py`](eval_dp_exec_horizon.py) | Success rate of the base policy alone under a given execution horizon |
| [`pypath/`](pypath/) | Import path for the scripts (see step 3) |
| [`python_envs/`](python_envs/) | `pip freeze` of the two Python environments |

## Steps

### 1. Code

```bash
# robomimic: official repo at d309eae ("Lazy CLIP download (#303)"), unmodified
git clone https://github.com/ARISE-Initiative/robomimic.git $ROOT
git -C $ROOT checkout d309eae
# this repo, branch diffusion
git clone -b diffusion https://github.com/lakshitadodeja/q2rl.git $CODE
```

### 2. Python environments (build them in a `batch` partition job, not on a login node)

1. **robomimic env**: Python 3.10; `pip install --no-deps -r python_envs/robomimic_env_pip_freeze.txt`
   (drop the `-e` line), then `pip install --no-deps -e $ROOT`. Key versions: robosuite 1.5.1,
   mujoco 3.2.3, numpy 2.2.6, torch 2.4.0, diffusers 0.11.1. `--no-deps` is needed because the
   list pins mink 1.3.0 next to mujoco 3.2.3, which pip's resolver rejects.
2. **q2rl-dp env** (`$ENV`): `conda create -p $ENV --clone <robomimic env>`, then install the
   Q2RL packages with the existing versions as constraints so nothing in the training stack changes:
   ```bash
   $ENV/bin/python -m pip list --format=freeze | grep -v -i '^robomimic==' > constraints.txt
   $ENV/bin/python -m pip install -c constraints.txt \
     "jax[cuda12]==0.4.33" jaxlib==0.4.33 flax==0.9.0 chex==0.1.86 optax==0.1.5 distrax==0.1.2 \
     tensorflow-probability==0.25.0 ml_collections==1.1.0 gym==0.23.1 \
     absl-py opencv-python tqdm orbax-checkpoint einops overrides "wandb<0.31"
   ```
   The resulting versions are in `python_envs/q2rl_dp_env_pip_freeze.txt`.

### 3. Import path (already handled by the scripts)

The repo root has a `robomimic/` folder that would shadow the installed robomimic, so the repo root
must not be on `PYTHONPATH` or be the working directory. The scripts use
`PYTHONPATH=$CODE/diffusion_q2rl_lift/pypath`, which holds symlinks to `backend/` and
`experiments/` plus placeholder `d4rl` / `mj_envs` modules (`env_common.py` imports them; the
robosuite path never uses them). They also set `USE_FLAX=0`: diffusers 0.11.1 would otherwise
import Flax schedulers that break on jax 0.4.33. The script checks that the robomimic it imports
is `$ROOT/robomimic/`.

### 4. Checkpoint

On Oscar, copy it from the shared folder (about 500 MB) and check it:

```bash
cp /oscar/scratch/hlu84/q2rl_share/q2rl_base_full_data_epoch_8.pth .
echo "6481c97eca9166122191e6bd3a59ea037060a1b321d6e4722935aeba3c0cb2a0  q2rl_base_full_data_epoch_8.pth" | sha256sum -c
```

`ls` on that folder's parent is not allowed, so use the full path. This file is the original
checkpoint with the optimizer state removed (it is only needed to resume DP training); the network
and EMA weights are unchanged and give identical actions.

### 5. Smoke test (CPU, a few minutes)

```bash
mkdir -p slurm_logs   # sbatch writes logs to slurm_logs/ under the directory you submit from
sbatch -p batch --gres=none -c 4 --mem=32G -t 00:45:00 \
  --export=ALL,MODE=smoke,EXEC_HORIZON=1,ROOT=$ROOT,ENV=$ENV,CODE=$CODE,CKPT=<checkpoint path> \
  $CODE/diffusion_q2rl_lift/run_q2rl_dp.sbatch
```

It passes when the log ends with `SMOKE TEST PASSED (incl. resume)`. That covers rollouts, Q
estimation, online updates, evaluation, `metrics.jsonl`, and stopping and resuming a run.

### 6. Full run, in two jobs per seed

Measured on an RTX A5000: about 0.66 s per online step (one 100-step DDPM plan plus 4 updates) and
about 50 minutes per evaluation early on (shorter once episodes succeed early), about 120 hours for
500k steps in total. Oscar limits each user's **running** gpu jobs to 4 GPU-days of requested time
(time limit × GPUs, summed), so a single 5-day job never starts. Each seed is therefore split:

```bash
X="ROOT=$ROOT,ENV=$ENV,CODE=$CODE,CKPT=<checkpoint path>,MODE=full,SEED=<seed>,EXEC_HORIZON=1"
S=$CODE/diffusion_q2rl_lift/run_q2rl_dp.sbatch
# Part 1: rollouts, Q estimation, online steps 0-300k (~70 h; may hit the 3-day limit, which is fine)
J1=$(sbatch --parsable -t 3-00:00:00 --export=ALL,$X,ONLINE_STEPS=300000 $S)
# Part 2: resume from part 1's latest state and continue to 500k; starts when part 1 ends
sbatch -t 3-00:00:00 --dependency=afterany:$J1 \
  --export=ALL,$X,ONLINE_STEPS=500000,RESUME_STATE=$ROOT/q2rl_runs/full-$J1/resume_state.pkl $S
```

- Each run writes `$ROOT/q2rl_runs/full-<job id>/resume_state.pkl` every 50k steps and at its end, so
  if part 1 hits its time limit, part 2 continues from the last save.
- The episode still in progress when the state is saved (at most 300 steps) is dropped; part 2
  starts a new episode. Those transitions could not be sampled yet anyway, since Q2RL samples only
  completed trajectories.
- **Two seeds:** with the 4 GPU-day limit, one account can run only one 3-day part at a time (plus
  about one more GPU-day). Run the two seeds from two accounts, or one after the other.

### 7. Results

Each part writes `q2rl_runs/full-<job id>/<group>/<run name>/metrics.jsonl`. The logged `step` is the
online step + 30,000 (the 100 × 300 rollout steps). Main keys: `evaluation/success_rate`,
`evaluation/bc_actions` (fraction of steps where Q2RL chose the BC action), `evaluation/rl_q`,
`evaluation/bc_q`.

For reference, the seed-0 run so far (20 episodes per evaluation):

| Online step | Success rate | BC action fraction | Avg. episode length |
|---|---|---|---|
| 0 (before RL training) | 0.45 | 0.79 | 230 |
| 20k | 0.50 | 0.19 | 187 |
| 40k | 0.75 | 0.21 | 115 |
| 60k | 0.90 | 0.10 | 76 |

### 8. Optional: check the base success rate

```bash
PYTHONPATH=$CODE/diffusion_q2rl_lift/pypath USE_FLAX=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  $ENV/bin/python $CODE/diffusion_q2rl_lift/eval_dp_exec_horizon.py \
  --ckpts <checkpoint path> --exec_horizon 1 --episodes 30 --horizon 300 --out_dir <output dir>
```

Run it in a GPU job; 30 episodes take about 45 minutes.
