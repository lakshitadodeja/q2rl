"""Reporting checks for resumed epochs, incomplete evaluations and seed aggregation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('export_results', Path(__file__).with_name('export_results.py'))
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def train(epoch, loss):
    return f'Train Epoch {epoch}\n' + json.dumps({'Loss': loss, 'Optimizer/policy0_lr': 1e-4}) + '\n'


def evaluate(epoch, rate):
    return f'Epoch {epoch} Rollouts took 1.6s (avg) with results:\nEnv: low_dim_v141\n' + json.dumps(
        {'Success_Rate': rate, 'Horizon': 44., 'Return': rate}) + '\n'


class ReportTest(unittest.TestCase):
    def test_resume_replaces_duplicates_and_ignores_incomplete_rollout(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / 'old.out', Path(folder) / 'resume.out'
            a.write_text(train(349, .1) + train(350, .2) +
                         'rollout: num_episodes=50\n52%| 26/50\nrun failed with error:\n')
            b.write_text(train(350, .03) + evaluate(350, .98) + train(2000, .01) + evaluate(2000, 1.))
            training, evaluation = REPORT.parse_training([a, b])
            self.assertEqual([r['epoch'] for r in training], [349, 350, 2000])
            self.assertEqual(training[1]['loss'], .03)
            self.assertEqual([r['epoch'] for r in evaluation], [350, 2000])

    def test_report_and_independent_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = root / 'run.out'
            log.write_text(train(50, .2) + evaluate(50, .9) + train(2000, .02) + evaluate(2000, 1.))
            config = root / 'config.json'
            config.write_text(json.dumps({'experiment': {'epoch_every_n_steps': 100, 'rollout': {'n': 50}},
                                           'train': {'seed': 1, 'num_epochs': 2000}}))
            REPORT.training_report([log], config, root / 'report')
            summary = json.loads((root / 'report/training_summary.json').read_text())
            self.assertEqual(summary['last_evaluation']['bc_gradient_updates'], 200000)
            self.assertTrue((root / 'report/lift_success.pdf').is_file())
            manifest = {'episodes_per_seed': 20, 'evaluation_seeds': [1000,1001,1002,1003,1004],
                        'training_seed': 1, 'checkpoint': 'synthetic-test.pth', 'checkpoint_sha256': 'test'}
            (root / 'evaluation_manifest.json').write_text(json.dumps(manifest))
            for seed in manifest['evaluation_seeds']:
                successes = 19 if seed == 1003 else 20
                (root / f'seed_{seed}.log').write_text('Average Rollout Stats\n' + json.dumps(
                    {'Num_Success': successes, 'Success_Rate': successes/20, 'Horizon': 44.}))
            REPORT.evaluation_report(root)
            summary = json.loads((root / 'report/independent_summary.json').read_text())
            self.assertEqual(summary['successes'], 99)
            self.assertEqual(summary['episodes'], 100)
            self.assertEqual(summary['training_seed_count'], 1)
            (root / 'seed_1004.log').write_text('Evaluation started but not completed')
            with self.assertRaisesRegex(AssertionError, 'incomplete'):
                REPORT.evaluation_report(root)


if __name__ == '__main__':
    unittest.main()
