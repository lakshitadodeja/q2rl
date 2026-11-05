"""Export real BC training logs and independent evaluation results; no invented data."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil


def json_after(text, offset):
    return json.JSONDecoder().raw_decode(text[offset:].lstrip())[0]


def parse_training(paths):
    training, evaluation = {}, {}
    for path in paths:
        text = Path(path).read_text(errors='replace')
        for match in re.finditer(r'Train Epoch (\d+)\s*\n\s*(?=\{)', text):
            values = json_after(text, match.end())
            epoch = int(match.group(1))
            training[epoch] = {'epoch': epoch, 'loss': float(values['Loss']),
                               'learning_rate': float(values['Optimizer/policy0_lr'])}
        pattern = r'Epoch (\d+) Rollouts[^\n]*\nEnv: [^\n]+\n\s*(?=\{)'
        for match in re.finditer(pattern, text):
            values = json_after(text, match.end())
            epoch = int(match.group(1))
            rate = float(values['Success_Rate'])
            assert 0 <= rate <= 1
            evaluation[epoch] = {'epoch': epoch, 'success_rate': rate,
                                 'mean_horizon': float(values['Horizon']),
                                 'mean_return': float(values['Return'])}
    return [training[k] for k in sorted(training)], [evaluation[k] for k in sorted(evaluation)]


def write_csv(path, rows):
    assert rows, f'No records for {path}'
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training_report(logs, config_path, output, video=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training, evaluation = parse_training(logs)
    updates_per_epoch = config['experiment']['epoch_every_n_steps']
    rollouts = config['experiment']['rollout']['n']
    assert training and evaluation, 'No complete training or evaluation records found'
    for rows in (training, evaluation):
        for row in rows:
            row['bc_gradient_updates'] = row['epoch'] * updates_per_epoch
    write_csv(output / 'training_metrics.csv', training)
    write_csv(output / 'evaluation_metrics.csv', evaluation)
    for metric, rows, label, filename in [
        ('success_rate', evaluation, 'Lift success rate', 'lift_success'),
        ('loss', training, 'Flow matching training loss', 'training_loss'),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.1), layout='constrained')
        ax.plot([row['bc_gradient_updates'] for row in rows], [row[metric] for row in rows],
                color='#8650b3', linewidth=1.5, marker='o' if metric == 'success_rate' else None,
                markersize=3, label='Flow BC')
        ax.set(xlabel='BC gradient updates', ylabel=label, title='Lift / low-dimensional observations')
        ax.grid(alpha=.22)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        if metric == 'success_rate':
            ax.set_ylim(0, 1.05)
            ax.legend(title=f"Training seed {config['train']['seed']}; {rollouts} episodes/evaluation", fontsize=9, title_fontsize=8)
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'{filename}.{suffix}', dpi=200)
        plt.close(fig)
    summary = {'training_seed': config['train']['seed'], 'target_epochs': config['train']['num_epochs'],
               'last_training_epoch': training[-1]['epoch'], 'last_evaluation': evaluation[-1],
               'episodes_per_training_evaluation': rollouts, 'training_seed_count': 1,
               'x_axis': 'BC gradient updates, not environment steps',
               'uncertainty': 'No cross-training-seed confidence band: only one training run.',
               'source_logs': [str(path) for path in logs],
               'resume_note': 'Later logs replace duplicate epoch entries after checkpoint resume; RNG sequence need not match an uninterrupted run.'}
    if video:
        source = Path(video)
        assert source.is_file(), f'Missing video: {source}'
        target = output / 'Lift_epoch_2000.mp4'
        shutil.copy2(source, target)
        summary['video'] = {'file': target.name, 'source': str(source),
                            'purpose': 'Training epoch 2000 evaluation; not the independent 100-episode evaluation.'}
    (output / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Training plots/CSVs exported: {output}')


def evaluation_report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'evaluation_manifest.json').read_text())
    rows = []
    episodes = manifest['episodes_per_seed']
    for seed in manifest['evaluation_seeds']:
        text = (directory / f'seed_{seed}.log').read_text(errors='replace')
        assert 'Traceback' not in text, f'Inspect error in seed_{seed}.log'
        marker = 'Average Rollout Stats'
        assert marker in text, f'Evaluation incomplete for seed {seed}'
        stats = json_after(text, text.rindex(marker) + len(marker))
        success = float(stats['Num_Success'])
        rate = float(stats['Success_Rate'])
        assert success.is_integer() and 0 <= success <= episodes
        assert math.isclose(rate, success / episodes, abs_tol=1e-8)
        rows.append({'evaluation_seed': seed, 'episodes': episodes, 'successes': int(success),
                     'success_rate': rate, 'mean_horizon': float(stats['Horizon'])})
    total = sum(row['episodes'] for row in rows)
    successes = sum(row['successes'] for row in rows)
    output = directory / 'report'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'independent_evaluation.csv', rows)
    summary = {'successes': successes, 'episodes': total, 'success_rate': successes / total,
               'mean_horizon': sum(row['mean_horizon'] * row['episodes'] for row in rows) / total,
               'evaluation_seeds': manifest['evaluation_seeds'], 'training_seed_count': 1,
               'training_seed': manifest['training_seed'], 'checkpoint': manifest['checkpoint'],
               'checkpoint_sha256': manifest['checkpoint_sha256'],
               'description': 'One frozen epoch-2000 model, independently evaluated; five evaluation seeds are not five training seeds.'}
    (output / 'independent_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Independent evaluation: {successes}/{total} successes ({successes/total:.1%})')
    print(f'Report: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('training')
    training.add_argument('--logs', nargs='+', required=True)
    training.add_argument('--config', required=True)
    training.add_argument('--output', required=True)
    training.add_argument('--video')
    evaluation = commands.add_parser('evaluation')
    evaluation.add_argument('--directory', required=True)
    args = parser.parse_args()
    if args.command == 'training':
        training_report(args.logs, args.config, args.output, args.video)
    else:
        evaluation_report(args.directory)


if __name__ == '__main__':
    main()
