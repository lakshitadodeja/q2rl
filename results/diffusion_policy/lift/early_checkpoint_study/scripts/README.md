# Reproducibility scripts

These are portable adapters and analysis utilities for the archived experiment. No script
submits a scheduler job automatically. Model execution requires the original compatible
research environment, which has not yet been version-locked; activate it before running.

The commands below assume the shell is in this `early_checkpoint_study/` directory.
Replace quoted example locations with your own paths. The published config preserves
the original source config except for `train.output_dir`, now `experiments`.

## Verify or rebuild the tables

`export_results.py` uses only the Python standard library. Its input is the locally
retained archive layout, with `Work/` containing the training log, two evaluation logs,
and `eval_candidates.sh` and `eval_65.sh`. These logs are intentionally not published.

```bash
python scripts/export_results.py --source-root "/path/to/extracted/archive" --check-against data
python scripts/export_results.py --source-root "/path/to/extracted/archive" --output-dir "/path/to/rebuilt/data"
```

The extractor checks all 120 training rows, 24 preliminary evaluation rows, and four
independent evaluation rows. It checks independent rollout count, horizon, and seed
against the original shell scripts and success counts against logged success rates.
The exported columns match the supplied CSVs; success rates are fractions, not percentages.

## Rebuild the figures

With Matplotlib installed in a separate analysis environment:

```bash
python scripts/plot_curves.py
```

This reads the three published CSVs and writes the three figures. Use `--output-dir`
to write elsewhere. No model, dataset, logs, or GPU is needed. Pixel rendering may differ
between Matplotlib versions, but the numeric data and axes are unchanged.

## Inspect commands without running a model

`reproduce.py` accepts the path to a RoboMimic source checkout containing
`robomimic/scripts/train.py` and `robomimic/scripts/run_trained_agent.py`. `--dry-run`
prints commands without writing a config or starting model execution; example input
paths need not exist in this mode.

```bash
python scripts/reproduce.py --robomimic-root "/path/to/robomimic" --dry-run train --dataset "/path/to/lift/ph/low_dim_v15.hdf5" --output-dir "/path/to/experiments"
python scripts/reproduce.py --robomimic-root "/path/to/robomimic" --dry-run evaluate --checkpoint "/path/to/models/model_epoch_60.pth"
python scripts/reproduce.py --robomimic-root "/path/to/robomimic" --dry-run video --checkpoint "/path/to/models/model_epoch_60.pth" --output "/path/to/new-video.mp4"
```

## Reproduce evaluation

When the trusted checkpoints and intended research environment are available:

```bash
python scripts/reproduce.py --robomimic-root "/path/to/robomimic" evaluate --checkpoint "/path/to/models/model_epoch_60.pth"
```

Repeat with the exact epoch-55, epoch-65, and epoch-75 checkpoint basenames in the study
README. The runner uses the original `--n_rollouts 50 --horizon 400 --seed 123` settings.
Both `MUJOCO_GL` and `PYOPENGL_PLATFORM` are set to `egl`, as in the archived scripts.
The default seed can be overridden explicitly for future multi-seed experiments; those
new evaluations must be reported separately from the archived results.

## Record a separate video

```bash
python scripts/reproduce.py --robomimic-root "/path/to/robomimic" video --checkpoint "/path/to/models/model_epoch_60.pth" --output "/path/to/new-video.mp4"
```

The archived recording used `--n_rollouts 5 --horizon 400 --seed 123 --video_skip 5
--camera_names agentview`, with the original evaluation script writing video at 20 FPS.
This command uses those rollout and rendering flags and refuses to overwrite an existing
video. It does not attach success labels or remove failures. It has not been executed
as part of preparing this contribution.

## Reproduce training only when separately intended

Training is already complete. The following command is documented for future reproduction
and was **not run** for this submission:

```bash
python scripts/reproduce.py --robomimic-root "/path/to/robomimic" train --dataset "/path/to/lift/ph/low_dim_v15.hdf5" --output-dir "/path/to/new-experiments"
```

The source config had `train.data = null`; the archived launch passed `--dataset`, and
the logged runtime config contained `train.data = [{"path": "..."}]`. The adapter writes
that effective dataset structure directly into a temporary config, alongside the chosen
output directory. All learning and checkpoint settings are preserved. Choose a new output
directory to keep existing runs separate. The adapter omits the archived shell script's
separate `--debug` smoke-training invocation and cluster-specific resource directives.
Inspect the training log for `finished run successfully!`; some RoboMimic versions catch
training exceptions internally instead of returning a failing process exit status.

## Validation performed

- Every published CSV cell was compared with the original logs.
- The portable config was compared with the original config and logged runtime config.
- The three figure files were generated and visually inspected.
- The MP4 was decoded through all 263 frames; its bytes match the source archive.
- Python syntax, analysis execution, and all three runner dry-run modes were checked.
- Runner flags were checked against archived commands and the repository CLI definitions.

No training, simulation, checkpoint loading, or GPU allocation was performed. Exact package
versions and the original source commit remain unverified; end-to-end execution in the
research environment is still needed for a full reproduction claim.
