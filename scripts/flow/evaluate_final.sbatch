#!/bin/bash
#SBATCH --job-name=flow-final-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/flow-final-eval-%j.out
#SBATCH --error=logs/flow-final-eval-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the q2rl-flow project root}"
module load glew/2.2.0-frnm
source .venv/bin/activate
export PYTHONPATH="$PWD/robomimic${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib/nvidia"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 MPLCONFIGDIR="${TMPDIR:-/tmp}/flow-mpl-${SLURM_JOB_ID}"
ulimit -Sn "$(ulimit -Hn)"
export FLOW_FINAL_DIR="$PWD/results/flow_aligned_train_8357773/training/lift_flow_aligned_seed1/20261104150822"
export FLOW_FINAL_MODEL="$FLOW_FINAL_DIR/models/model_epoch_2000.pth"
export FLOW_EVAL_DIR="$PWD/results/flow_final_eval_${SLURM_JOB_ID}"
mkdir -p "$FLOW_EVAL_DIR/report"
python - <<'PY'
import hashlib, importlib.metadata, json, os, sys
from pathlib import Path
import torch, robomimic
root = Path.cwd()
assert Path(robomimic.__file__).resolve() == root / 'robomimic/robomimic/__init__.py'
assert torch.cuda.is_available(), 'GPU unavailable'
model = Path(os.environ['FLOW_FINAL_MODEL'])
assert model.is_file(), f'Missing final checkpoint: {model}'
source = root / 'results/flow_aligned_train_8357773'
original = json.loads((source / 'manifest.json').read_text())
versions = {p: importlib.metadata.version(p) for p in original['versions']}
assert versions == original['versions'], 'Use the same software versions as training'
cfg = json.loads((source / 'config.json').read_text())
assert cfg['algo_name'] == 'flow_policy' and cfg['train']['num_epochs'] == 2000
sha = hashlib.sha256()
with model.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        sha.update(block)
manifest = {'checkpoint': str(model), 'checkpoint_sha256': sha.hexdigest(),
            'training_seed': cfg['train']['seed'], 'evaluation_seeds': [1000,1001,1002,1003,1004],
            'episodes_per_seed': 20, 'horizon': 400, 'versions': versions,
            'python': sys.version, 'gpu': torch.cuda.get_device_name(0),
            'training_dataset_sha256': original['dataset_sha256'],
            'evaluation': 'Fixed epoch-2000 model; no training updates; one fresh process per evaluation seed.'}
(Path(os.environ['FLOW_EVAL_DIR']) / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY
# Export training charts and the existing final-epoch video before evaluating.
python scripts/flow/export_results.py training \
  --logs logs/lift-flow-aligned-8357773.out logs/lift-flow-resume-8367567.out \
  --config results/flow_aligned_train_8357773/config.json \
  --output "$FLOW_EVAL_DIR/report" \
  --video "$FLOW_FINAL_DIR/videos/low_dim_v141_epoch_2000.mp4"
for seed in 1000 1001 1002 1003 1004; do
    python -u robomimic/robomimic/scripts/run_trained_agent.py \
      --agent "$FLOW_FINAL_MODEL" --n_rollouts 20 --horizon 400 --seed "$seed" \
      > "$FLOW_EVAL_DIR/seed_${seed}.log" 2>&1
    grep -q 'Average Rollout Stats' "$FLOW_EVAL_DIR/seed_${seed}.log"
    printf 'Evaluation seed %s completed\n' "$seed"
    tail -n 8 "$FLOW_EVAL_DIR/seed_${seed}.log"
done
python scripts/flow/export_results.py evaluation --directory "$FLOW_EVAL_DIR"
printf 'Completed final evaluation and report: %s/report\n' "$FLOW_EVAL_DIR"
