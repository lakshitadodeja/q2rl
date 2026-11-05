"""Matched-config and real train.py integration tests (synthetic HDF5, no simulator)."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

from robomimic.config import config_factory
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import torch_utils as TorchUtils

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'experiments/configs/flow'


class UpstreamIntegrationTest(unittest.TestCase):
    def test_shared_settings_match_template(self):
        template = json.loads((ROOT / 'docs/reference/diffusion_policy_user_template.json').read_text())
        diffusion = json.loads((CONFIGS / 'lift_diffusion_reference.json').read_text())
        flow = json.loads((CONFIGS / 'lift_low_dim.json').read_text())
        adjusted = copy.deepcopy(template)
        adjusted['train']['data'] = diffusion['train']['data']
        adjusted['train']['output_dir'] = diffusion['train']['output_dir']
        adjusted['experiment']['name'] = diffusion['experiment']['name']
        self.assertEqual(adjusted, diffusion)
        for c in (flow, diffusion):
            c.pop('algo_name')
            c['experiment'].pop('name')
            c['train'].pop('output_dir')
        flow['algo'].pop('flow')
        diffusion['algo'].pop('ddpm')
        diffusion['algo'].pop('ddim')
        self.assertEqual(flow, diffusion)

    def test_cosine_warmup_and_decay(self):
        config = config_factory('flow_policy')
        params = config.algo.optim_params.policy.to_dict()
        params.update(num_train_batches=100, num_epochs=2000)
        net = torch.nn.Linear(1, 1)
        optimizer = TorchUtils.optimizer_from_optim_params(params, net)
        scheduler = TorchUtils.lr_scheduler_from_optim_params(params, net, optimizer)
        curve = scheduler.lr_lambdas[0]
        self.assertEqual(curve(0), 0.)
        self.assertAlmostEqual(curve(250), .5)
        self.assertAlmostEqual(curve(500), 1.)
        self.assertAlmostEqual(curve(100250), .5)
        self.assertAlmostEqual(curve(200000), 0.)
        for _ in range(2):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-4 * 2 / 500)

    def test_official_train_and_reload_both_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            dataset = work / 'synthetic.hdf5'
            shapes = {'robot0_eef_pos': 3, 'robot0_eef_quat': 4,
                      'robot0_gripper_qpos': 2, 'object': 10}
            rng = np.random.default_rng(7)
            with h5py.File(dataset, 'w') as f:
                group = f.create_group('data')
                group.attrs['env_args'] = json.dumps({'env_name': 'Lift', 'type': 1, 'env_kwargs': {}})
                group.attrs['total'] = 48
                for i in range(2):
                    demo = group.create_group(f'demo_{i}')
                    demo.attrs['num_samples'] = 24
                    for key, dim in shapes.items():
                        demo.create_dataset('obs/' + key, data=rng.normal(size=(24, dim)).astype('float32'))
                    demo.create_dataset('actions', data=rng.uniform(-1, 1, (24, 7)).astype('float32'))
                    demo.create_dataset('rewards', data=np.zeros(24, dtype='float32'))
                    demo.create_dataset('dones', data=np.zeros(24, dtype='int64'))
            for name in ('lift_low_dim.json', 'lift_diffusion_reference.json'):
                cfg = json.loads((CONFIGS / name).read_text())
                algo = cfg['algo_name']
                cfg['train'].update(data=str(dataset), output_dir=str(work / algo),
                                    num_epochs=2, batch_size=2, cuda=False)
                cfg['experiment']['epoch_every_n_steps'] = 3
                cfg['experiment']['rollout']['enabled'] = False
                cfg['experiment']['render_video'] = False
                cfg['experiment']['save']['every_n_epochs'] = 1
                cfg['experiment']['logging']['log_tb'] = False
                cfg['algo']['unet'].update(down_dims=[16, 32, 64], diffusion_step_embed_dim=16)
                path = work / (algo + '.json')
                path.write_text(json.dumps(cfg))
                result = subprocess.run([sys.executable, str(ROOT / 'robomimic/robomimic/scripts/train.py'),
                                         '--config', str(path)], text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('finished run successfully!', result.stdout, result.stdout + result.stderr)
                self.assertNotIn('Traceback', result.stdout + result.stderr)
                checkpoint = next((work / algo).rglob('model_epoch_2.pth'))
                raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
                scheduler = raw['model']['lr_schedulers']['policy']
                self.assertEqual(scheduler['last_epoch'], 6)
                self.assertAlmostEqual(raw['model']['optimizers']['policy']['param_groups'][0]['lr'], 1e-4 * 6 / 500)
                policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(checkpoint), device=torch.device('cpu'))
                policy.start_episode()
                action = policy({k: np.zeros((2, d), dtype='float32') for k, d in shapes.items()})
                self.assertEqual(action.shape, (7,))
                self.assertTrue(np.isfinite(action).all())
                if algo == 'flow_policy':
                    # Resume through the official CLI and verify three more updates.
                    cfg["train"]["num_epochs"] = 3
                    path.write_text(json.dumps(cfg))
                    resumed = subprocess.run([sys.executable, str(ROOT / 'robomimic/robomimic/scripts/train.py'),
                                              '--config', str(path), '--resume'], text=True, capture_output=True, timeout=120)
                    self.assertIn('finished run successfully!', resumed.stdout, resumed.stdout + resumed.stderr)
                    self.assertIn('resuming training from epoch 3', resumed.stdout)
                    latest = torch.load(next((work / algo).rglob('last.pth')), map_location='cpu', weights_only=False)
                    self.assertEqual(latest['model']['lr_schedulers']['policy']['last_epoch'], 9)
                    self.assertEqual(latest['model']['ema_optimization_step'], 9)


if __name__ == '__main__':
    unittest.main()
