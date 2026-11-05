#!/bin/bash
#SBATCH --job-name=lift-flow-resume
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lift-flow-resume-%j.out
#SBATCH --error=logs/lift-flow-resume-%j.err
# Usage from project root:
# sbatch --exclude=gpu3001 scripts/flow/resume_aligned.sbatch results/flow_aligned_train_8357773
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
[[ $# == 1 ]] || { echo 'Pass the original results/flow_aligned_train_<jobid> directory.'; exit 1; }
module load glew/2.2.0-frnm
source .venv/bin/activate
export PYTHONPATH="$PWD/robomimic${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib/nvidia"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
# Use the already permitted OS limit; do not change any training configuration.
printf 'Original file limits: soft=%s hard=%s\n' "$(ulimit -Sn)" "$(ulimit -Hn)"
ulimit -Sn "$(ulimit -Hn)"
export FLOW_SOURCE_RUN="$1"
export FLOW_RESUME_JOB="$SLURM_JOB_ID"
export FLOW_RESUME_LOG_DIR="$PWD/results/flow_resume_${SLURM_JOB_ID}"
mkdir -p "$FLOW_RESUME_LOG_DIR"
python - <<'PY'
import hashlib, importlib.metadata, json, os, shutil, socket, sys
from pathlib import Path
import torch, robomimic, robosuite
root = Path.cwd()
run = Path(os.environ['FLOW_SOURCE_RUN']).resolve()
assert run.is_relative_to(root / 'results'), 'Expected an experiment under project results/'
config_path = run / 'config.json'
cfg = json.loads(config_path.read_text())
assert cfg['algo_name'] == 'flow_policy'
assert cfg['train']['num_epochs'] == 2000, 'Do not silently change the original training budget'
assert torch.cuda.is_available(), 'GPU unavailable'
assert Path(robomimic.__file__).resolve() == root / 'robomimic/robomimic/__init__.py'
data = Path(cfg['train']['data']).expanduser()
assert data.is_file()
original_manifest = json.loads((run / 'manifest.json').read_text())
assert hashlib.sha256(data.read_bytes()).hexdigest() == original_manifest['dataset_sha256'], 'Dataset changed'
versions = {name: importlib.metadata.version(name) for name in original_manifest['versions']}
assert versions == original_manifest['versions'], f'Environment changed: {versions}'
base = Path(cfg['train']['output_dir']).expanduser() / cfg['experiment']['name']
assert base.is_absolute()
# Match the official resume directory selection, and reject ambiguous non-directories.
experiment = sorted(base.iterdir())[-1]
assert experiment.is_dir()
loaded = None
for name in ('last.pth', 'last_bak.pth'):
    p = experiment / name
    try:
        loaded = torch.load(p, map_location='cpu', weights_only=False)
        saved_epoch = int(loaded['variable_state']['epoch'])
        assert 0 < saved_epoch < cfg['train']['num_epochs']
        model = loaded['model']
        assert model['optimizers']['policy']['state'], 'Missing optimizer state'
        assert model['lr_schedulers']['policy'] is not None, 'Missing LR state'
        assert model['ema'] is not None, 'Missing EMA'
        checkpoint = p
        break
    except Exception as exc:
        print(f'Checkpoint check failed for {p}: {exc}', flush=True)
        loaded = None
assert loaded is not None, 'No valid resume checkpoint'
resume = Path(os.environ['FLOW_RESUME_LOG_DIR'])
report = {'source_run': str(run), 'checkpoint': str(checkpoint),
          'saved_epoch': saved_epoch, 'restart_epoch': saved_epoch + 1,
          'target_epoch': cfg['train']['num_epochs'], 'host': socket.gethostname(),
          'versions': versions, 'python': sys.version,
          'gpu': torch.cuda.get_device_name(0),
          'file_descriptor_count': len(list(Path('/proc/self/fd').iterdir())),
          'note': 'Fresh process; raised soft file limit to existing hard limit. Original config unchanged. This is a recovery attempt, not a confirmed fix for the resource read error.'}
mesh = Path(robosuite.__file__).parent / 'models/assets/robots/panda/obj_meshes/link3_vis/link3_vis_2.obj'
with mesh.open('rb') as f:
    assert f.read(1), f'Empty/unreadable asset: {mesh}'
report['asset_readable'] = str(mesh)
(resume / 'recovery.json').write_text(json.dumps(report, indent=2))
# train.py writes log.txt/config.json in-place on resume; preserve existing copies first.
for rel in ('logs/log.txt', 'config.json'):
    source = experiment / rel
    if source.is_file():
        target = resume / ('before_resume_' + source.name)
        shutil.copy2(source, target)
shutil.copy2(config_path, resume / 'original_config.json')
print(json.dumps(report, indent=2), flush=True)
PY
python -u robomimic/robomimic/scripts/train.py \
  --config "$FLOW_SOURCE_RUN/config.json" --resume 2>&1 | tee "$FLOW_RESUME_LOG_DIR/train.log"
grep -q '^finished run successfully!' "$FLOW_RESUME_LOG_DIR/train.log"
if grep -q '^run failed with error:' "$FLOW_RESUME_LOG_DIR/train.log"; then exit 1; fi
printf 'Completed resumed training. Recovery logs: %s\n' "$FLOW_RESUME_LOG_DIR"
