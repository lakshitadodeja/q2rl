#!/bin/bash
#SBATCH -J rm-q2rl-dp
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu3006
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x-%j.out   # relative to the directory you run sbatch from; create slurm_logs/ first

# Q2RL with the Lift Diffusion Policy as the BC policy (phase 1: V_BC only, log-prob and
# entropy set to 0). Code: this repo, branch diffusion (main/q2rl/q2rl_robosuite_diffusion.py).
# See diffusion_q2rl_lift/README.md.
#
# REQUIRED paths (set them below, or pass them with --export=ALL,ROOT=...,ENV=...,CODE=...):
#   ROOT  robomimic checkout at commit d309eae (installed editable in ENV); runs go to $ROOT/q2rl_runs
#   ENV   the q2rl-dp Python env
#   CODE  this repo (branch diffusion)
#   CKPT  the Diffusion Policy checkpoint (no default)
#
#   MODE=smoke  minimal pipeline check, no W&B (--debug), 4-step episodes: 2 BC rollouts ->
#               19 Q-estimation updates -> checkpoint -> 10 online RL steps (Q-greedy BC/RL choice;
#               agent updates start once the first online episode ends, at step 4) -> 1 eval episode.
#               sbatch -t 01:00:00 --export=ALL,MODE=smoke run_q2rl_dp.sbatch
#   MODE=wbtest short check of what MODE=smoke leaves out, with W&B online (needs `wandb login`):
#               the compute node reaching wandb.ai, eval results logged, and eval video rendering.
#               20-step episodes, 2 BC rollouts, 20 Q-estimation updates, 30 online steps, one
#               eval (2 episodes + video) at online step 20. Logged to W&B group lift_dp_wbtest.
#               sbatch -t 01:00:00 --export=ALL,MODE=wbtest run_q2rl_dp.sbatch
#   MODE=full   one seed with the hyperparameters of Q2RL's lift-state launch_q2rl_without_data.sh.
#               W&B is off (--debug); every metric is written to <run dir>/metrics.jsonl instead,
#               and eval videos are not recorded (they are only uploaded through W&B).
#               sbatch --export=ALL,MODE=full,SEED=0 run_q2rl_dp.sbatch
#   Options (any MODE): REPLAN=1 adds --dp_replan_on_rl_action=True (action-chunking variant B);
#   REUSE_ROLLOUTS=<rollouts.pkl> loads that run's initial BC rollouts instead of collecting
#   new ones, so Q estimation sees identical data. Example (variant B vs. the default run A):
#     sbatch --export=ALL,MODE=full,SEED=0,REPLAN=1,REUSE_ROLLOUTS=<A run dir>/rollouts.pkl run_q2rl_dp.sbatch
#   ONLINE_STEPS=<n> (MODE=full, default 500000) sets --num_online_steps, the total online steps
#   counted from the start of the run. Every run writes $OUT/resume_state.pkl every 50k steps and
#   at its end; RESUME_STATE=<that file> continues it in a new job up to ONLINE_STEPS (no new
#   rollouts, no Q estimation). Example, 300k now and the remaining 200k later:
#     sbatch -t 3-00:00:00 --export=ALL,MODE=full,SEED=0,EXEC_HORIZON=1,CKPT=<ckpt>,ONLINE_STEPS=300000 run_q2rl_dp.sbatch
#     sbatch -t 3-00:00:00 --dependency=afterany:<part-1 job> --export=ALL,MODE=full,SEED=0,EXEC_HORIZON=1,CKPT=<ckpt>,RESUME_STATE=<part-1 run dir>/resume_state.pkl run_q2rl_dp.sbatch
#   (the GPU quota allows at most 4 GPU-days of requested time across running jobs).
set -euo pipefail
: "${MODE:?set MODE=smoke, wbtest or full}"
SEED=${SEED:-0}
: "${ROOT:?set ROOT=<robomimic checkout>}" "${ENV:?set ENV=<q2rl-dp env>}" "${CODE:?set CODE=<q2rl clone>}" "${CKPT:?set CKPT=<DP checkpoint .pth>}"
OUT=$ROOT/q2rl_runs/$MODE-$SLURM_JOB_ID
mkdir -p "$OUT"

# The repo root has a robomimic/ project folder that would shadow the installed robomimic,
# so only backend/ and experiments/ are put on the import path (via the diffusion_q2rl_lift/pypath symlinks),
# and Python is never started with the repo root as its working directory.
# diffusion_q2rl_lift/pypath holds symlinks to backend/ and experiments/ plus placeholder d4rl/mj_envs
# modules (env_common.py imports them; the robosuite path never uses them).
export PYTHONPATH=$CODE/diffusion_q2rl_lift/pypath
# diffusers 0.11.1 (used by the Diffusion Policy) would also import its Flax schedulers once
# flax is installed; those use jax.random.KeyArray, removed in jax 0.4.33. The DP only uses
# the PyTorch schedulers, so turn off diffusers' Flax detection. Q2RL's own jax/flax are unaffected.
export USE_FLAX=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # jax and torch share the GPU
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_DIR=$OUT
cd "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU on this node"
"$ENV/bin/python" -c "import jax; print('jax devices:', jax.devices())"
git -C "$CODE" log -1 --oneline; git -C "$CODE" status --short
"$ENV/bin/python" -c "import robomimic, sys; f = robomimic.__file__ or ''; print('robomimic', robomimic.__version__ if f else None, f); sys.exit(0 if f.startswith('$ROOT/robomimic/') else 'wrong robomimic imported')"
echo "checkpoint: $CKPT"; sha256sum "$CKPT"

COMMON=(
    --agent=q2rl_rs
    --config=$CODE/experiments/configs/train_config.py:robosuite_q2rl
    --env=lift
    --resume_path_bc="$CKPT"
    --reward_scale=5.0 --reward_bias=-1.0 --bc_weight=0.3
    --get_demo_buffer=False
    --get_new_rollouts=True --replay_path="$OUT/rollouts.pkl"
    --save_dir="$OUT"
    --seed="$SEED"
    --render=False
)
# absl keeps the last value of a repeated flag, so these override COMMON.
# EXEC_HORIZON=1: predict the chunk, execute only its first action (replan every step).
if [ -n "${EXEC_HORIZON:-}" ]; then
    COMMON+=(--dp_exec_horizon="$EXEC_HORIZON")
    echo "Diffusion Policy executes $EXEC_HORIZON action(s) per predicted chunk"
fi
[ "${REPLAN:-0}" = 1 ] || unset REPLAN   # so the group-name suffix below only appears for B
if [ "${REPLAN:-0}" = 1 ]; then
    COMMON+=(--dp_replan_on_rl_action=True)
    echo "action chunking: variant B (replan after an executed RL action)"
else
    echo "action chunking: variant A (keep the queued plan)"
fi
if [ -n "${RESUME_STATE:-}" ]; then
    [ -s "$RESUME_STATE" ] || { echo "RESUME_STATE not found: $RESUME_STATE" >&2; exit 1; }
    COMMON+=(--resume_state="$RESUME_STATE")
    echo "resuming: $RESUME_STATE"; ls -l "$RESUME_STATE"
fi
if [ -n "${REUSE_ROLLOUTS:-}" ]; then
    [ -s "$REUSE_ROLLOUTS" ] || { echo "REUSE_ROLLOUTS not found: $REUSE_ROLLOUTS" >&2; exit 1; }
    COMMON+=(--get_new_rollouts=False --replay_path="$REUSE_ROLLOUTS")
    echo "reusing initial BC rollouts: $REUSE_ROLLOUTS"; sha256sum "$REUSE_ROLLOUTS"
fi

if [ "$MODE" = smoke ]; then
    "$ENV/bin/python" "$CODE/main/q2rl/q2rl_robosuite_diffusion.py" "${COMMON[@]}" \
        --debug=True --group=lift_dp_smoke --max_traj_length=4 \
        --num_eval_rollouts=2 --num_q_est_steps=20 --num_online_steps=10 \
        --batch_size=64 --utd=1 \
        --log_interval=5 --eval_interval=26 --n_eval_trajs=1 \
        --video_interval=1000000 --save_interval=1000000 \
        2>&1 | tee "$OUT/run.log"
    status=${PIPESTATUS[0]}
    "$ENV/bin/python" - "$OUT" "${REUSE_ROLLOUTS:-$OUT/rollouts.pkl}" <<'EOF'
import os, pickle, sys
out = sys.argv[1]
tr = pickle.load(open(sys.argv[2], "rb"))   # this run's rollouts, or the reused file
eps = sum(int(t["dones"]) for t in tr)
print(f"SMOKE rollouts: {len(tr)} transitions, {eps} finished episodes, "
      f"bc_probs all 0: {all(t['bc_probs'] == 0.0 for t in tr)}, "
      f"bc_entropy all 0: {all(t['bc_entropy'] == 0.0 for t in tr)}")
EOF
    ls -R "$OUT" | grep -i checkpoint | head -3
    # 19 Q-estimation updates; anything above that comes from online updates
    n=$(grep -oP "agent update count \(agent.state.step\): \K[0-9]+" "$OUT/run.log" || echo 0)
    echo "agent updates: $n (Q-estimation 19 + online $((n - 19)))"
    [ "$n" -gt 19 ] || { echo "SMOKE TEST FAILED: no online agent updates"; exit 1; }
    m=$(find "$OUT" -name metrics.jsonl | head -1)
    echo "metrics file: $m ($(wc -l < "$m") rows)"; grep -h '"evaluation/' "$m" | head -2
    grep -q '"evaluation/success_rate"' "$m" || { echo "SMOKE TEST FAILED: no evaluation/success_rate in metrics.jsonl"; exit 1; }
    grep -q "training complete!" "$OUT/run.log" && [ "$status" = 0 ] || { echo "SMOKE TEST FAILED (exit $status)"; exit 1; }
    # Part 2: continue this run from its resume state to 20 online steps (10 more).
    [ -s "$OUT/resume_state.pkl" ] || { echo "SMOKE TEST FAILED: no resume_state.pkl"; exit 1; }
    mv "$OUT/resume_state.pkl" "$OUT/resume_state_part1.pkl"
    "$ENV/bin/python" "$CODE/main/q2rl/q2rl_robosuite_diffusion.py" "${COMMON[@]}" \
        --resume_state="$OUT/resume_state_part1.pkl" \
        --debug=True --group=lift_dp_smoke_resumed --max_traj_length=4 \
        --num_eval_rollouts=2 --num_q_est_steps=20 --num_online_steps=20 \
        --batch_size=64 --utd=1 \
        --log_interval=5 --eval_interval=36 --n_eval_trajs=1 \
        --video_interval=1000000 --save_interval=1000000 \
        2>&1 | tee "$OUT/run_part2.log"
    status=${PIPESTATUS[0]}
    grep "RESUMED from" "$OUT/run_part2.log"
    grep -q "Getting rollouts" "$OUT/run_part2.log" && { echo "SMOKE TEST FAILED: resumed run collected rollouts"; exit 1; }
    n2=$(grep -oP "agent update count \(agent.state.step\): \K[0-9]+" "$OUT/run_part2.log" || echo 0)
    echo "agent updates after resume: $n2 (part 1 ended at $n)"
    [ "$n2" -gt "$n" ] || { echo "SMOKE TEST FAILED: resumed run did not continue updating"; exit 1; }
    grep -q "training complete!" "$OUT/run_part2.log" && [ "$status" = 0 ] && echo "SMOKE TEST PASSED (incl. resume)" || { echo "SMOKE TEST FAILED in resumed part (exit $status)"; exit 1; }
elif [ "$MODE" = wbtest ]; then
    "$ENV/bin/python" - <<'EOF'
import wandb, urllib.request
assert wandb.api.api_key, "no W&B API key: run `wandb login` on a login node first"
print("W&B key found; api.wandb.ai reachable, HTTP", urllib.request.urlopen("https://api.wandb.ai", timeout=20).status)
EOF
    echo "W&B entity: ${WANDB_ENTITY:-<logged-in account default>}"
fi

if [ "$MODE" = wbtest ]; then
    "$ENV/bin/python" "$CODE/main/q2rl/q2rl_robosuite_diffusion.py" "${COMMON[@]}" \
        --group=lift_dp_wbtest --project=q2rl --max_traj_length=20 \
        --num_eval_rollouts=2 --num_q_est_steps=20 --num_online_steps=30 \
        --batch_size=64 --utd=1 \
        --log_interval=10 --eval_interval=40 --video_interval=40 --n_eval_trajs=2 \
        --save_interval=1000000 \
        2>&1 | tee "$OUT/run.log"
    status=${PIPESTATUS[0]}
    grep -q "training complete!" "$OUT/run.log" && [ "$status" = 0 ] || { echo "WBTEST FAILED (exit $status)"; exit 1; }
    grep -oE "https://wandb.ai/[^ ]+" "$OUT/run.log" | sort -u | head -2
    echo "WBTEST PASSED: check the run page for evaluation/success_rate and eval_video"
elif [ "$MODE" = full ]; then
    "$ENV/bin/python" "$CODE/main/q2rl/q2rl_robosuite_diffusion.py" "${COMMON[@]}" \
        --debug=True --group="lift_state_q2rl_diffusion_without_data${EXEC_HORIZON:+_exec$EXEC_HORIZON}${REPLAN:+_replan}" --max_traj_length=300 \
        --num_eval_rollouts=100 --num_q_est_steps=20_000 --num_online_steps="${ONLINE_STEPS:-500000}" \
        --batch_size=256 --utd=4 \
        --log_interval=5000 --eval_interval=20_000 --n_eval_trajs=20 \
        --video_interval=1000000000 --save_interval=50_000 \
        2>&1 | tee "$OUT/run.log"
    status=${PIPESTATUS[0]}
    grep -q "training complete!" "$OUT/run.log" && [ "$status" = 0 ] || { echo "FULL RUN FAILED (exit $status)"; exit 1; }
    echo "FULL RUN DONE"
elif [ "$MODE" != smoke ]; then
    echo "unknown MODE=$MODE" >&2; exit 1
fi
