#!/bin/bash
#SBATCH --job-name=lift-flow-align
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lift-flow-aligned-%j.out
#SBATCH --error=logs/lift-flow-aligned-%j.err
# From project root: mkdir -p logs; sbatch --time=00:30:00 scripts/flow/run_aligned.sbatch smoke
# Full experiment (after smoke passes): sbatch scripts/flow/run_aligned.sbatch train
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the q2rl-flow project root}"
mode="${1:-smoke}"
[[ "$mode" == smoke || "$mode" == train ]] || { echo 'Expected smoke or train'; exit 1; }
module load glew/2.2.0-frnm
source .venv/bin/activate
export PYTHONPATH="$PWD/robomimic${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib/nvidia"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
ulimit -Sn 16384
export FLOW_RUN_MODE="$mode"
export FLOW_RUN_DIR="$PWD/results/flow_aligned_${mode}_${SLURM_JOB_ID}"
mkdir -p "$FLOW_RUN_DIR"
python - <<'PY'
import hashlib, importlib.metadata, json, os, sys
from pathlib import Path
import robomimic, torch
root = Path.cwd()
assert Path(robomimic.__file__).resolve() == root / 'robomimic/robomimic/__init__.py'
assert torch.cuda.is_available(), 'GPU unavailable; stop before training'
output = Path(os.environ['FLOW_RUN_DIR'])
cfg = json.loads((root / 'experiments/configs/flow/lift_low_dim.json').read_text())
data = (root / 'data/lift/ph/low_dim_v141.hdf5').resolve()
assert data.is_file(), f'Missing dataset: {data}'
cfg['train']['data'] = str(data)
cfg['train']['output_dir'] = str(output / 'training')
if os.environ['FLOW_RUN_MODE'] == 'smoke':
    cfg['train']['num_epochs'] = 2
    cfg['experiment']['epoch_every_n_steps'] = 3
    cfg['experiment']['rollout'].update(n=2, rate=1)
    cfg['experiment']['save']['every_n_epochs'] = 1
(output / 'config.json').write_text(json.dumps(cfg, indent=2))
versions = {p: importlib.metadata.version(p) for p in
            ['torch','torchvision','numpy','robosuite','mujoco','diffusers','huggingface_hub','transformers']}
manifest = {'upstream': json.loads((root / 'docs/robomimic_upstream.json').read_text()),
            'dataset_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
            'dataset_path': str(data), 'versions': versions, 'python': sys.version,
            'gpu': torch.cuda.get_device_name(0), 'mode': os.environ['FLOW_RUN_MODE']}
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY
python robomimic/robomimic/scripts/train.py --config "$FLOW_RUN_DIR/config.json" 2>&1 | tee "$FLOW_RUN_DIR/train.log"
# Upstream train.py catches exceptions, so also check its completion message.
grep -q '^finished run successfully!' "$FLOW_RUN_DIR/train.log"
if grep -q '^run failed with error:' "$FLOW_RUN_DIR/train.log"; then exit 1; fi
printf 'Completed %s run. Results: %s\n' "$mode" "$FLOW_RUN_DIR"
