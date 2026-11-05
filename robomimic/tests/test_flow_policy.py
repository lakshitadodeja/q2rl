"""CPU checks; run from repo root with PYTHONPATH=$PWD/robomimic."""
from collections import OrderedDict
from copy import deepcopy
import unittest
import torch

from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.algo.flow_policy import flow_matching_targets
from robomimic.utils import obs_utils as ObsUtils


class FlowPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        self.config = config_factory("flow_policy")
        self.config.algo.unet.down_dims = (16, 32, 64)
        self.config.algo.unet.diffusion_step_embed_dim = 16
        self.config.algo.flow.num_inference_timesteps = 2
        with self.config.values_unlocked():
            self.config.algo.optim_params.policy["num_train_batches"] = 100
            self.config.algo.optim_params.policy["num_epochs"] = 2000
        self.config.algo.optim_params.policy.learning_rate.warmup_steps = 0
        ObsUtils.initialize_obs_utils_with_config(self.config)
        self.shapes = OrderedDict(robot0_eef_pos=(3,), robot0_eef_quat=(4,),
                                  robot0_gripper_qpos=(2,), object=(10,))
        self.model = self.make_model()

    def make_model(self):
        return algo_factory("flow_policy", self.config, self.shapes, 7, torch.device("cpu"))

    def test_targets_and_euler_direction(self):
        noise, actions = torch.randn(2, 16, 7), torch.randn(2, 16, 7)
        x, v = flow_matching_targets(actions, noise, torch.tensor([0., 1.]))
        torch.testing.assert_close(x[0], noise[0])
        torch.testing.assert_close(x[1], actions[1])
        recovered = noise.clone()
        for _ in range(10):
            recovered = recovered + v / 10
        torch.testing.assert_close(recovered, actions)

    def test_train_validation_reload_and_rollout(self):
        m = self.model
        batch = {"obs": {k: torch.randn(2, 16, *s) for k, s in self.shapes.items()},
                 "actions": torch.rand(2, 16, 7) * 2 - 1}
        batch = m.process_batch_for_training(batch)
        batch = m.postprocess_batch_for_training(batch, obs_normalization_stats=None)
        before = deepcopy(m.nets.state_dict())
        m.set_train()
        info = m.train_on_batch(batch, epoch=1)
        self.assertTrue(torch.isfinite(info["losses"]["l2_loss"]))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in m.nets.state_dict().items()))
        trained = deepcopy(m.serialize())
        m.set_eval()
        m.train_on_batch(batch, epoch=1, validate=True)
        for k, v in m.nets.state_dict().items():
            torch.testing.assert_close(v, trained["nets"][k])
        restored = self.make_model()
        restored.deserialize(trained)
        restored.set_eval()
        obs = {k: torch.randn(1, *s) for k, s in self.shapes.items()}
        torch.manual_seed(123)
        a = m.get_action(obs)
        torch.manual_seed(123)
        b = restored.get_action(obs)
        torch.testing.assert_close(a, b)
        self.assertEqual(a.shape, (1, 7))
        self.assertTrue(torch.isfinite(a).all() and (a.abs() <= 1).all())
        self.assertEqual(len(m.action_queue), 7)
        m.reset()
        self.assertEqual(len(m.action_queue), 0)
        rollout = RolloutPolicy(restored)
        rollout.start_episode()
        action = rollout({k: v[0].numpy() for k, v in obs.items()})
        self.assertEqual(action.shape, (7,))

    def test_history_padding_and_chunk_alignment(self):
        m = self.model
        obs0 = {k: torch.zeros(1, *shape) for k, shape in self.shapes.items()}
        obs1 = {k: torch.ones(1, *shape) for k, shape in self.shapes.items()}
        first = m._prepare_observation_history(obs0)
        self.assertEqual(first["object"].shape, (1, 2, 10))
        self.assertEqual(first["object"].sum().item(), 0)
        second = m._prepare_observation_history(obs1)
        torch.testing.assert_close(second["object"][:, 0], obs0["object"])
        torch.testing.assert_close(second["object"][:, 1], obs1["object"])
        m.reset()
        m._prepare_observation_history(second)  # also accept upstream-style stacks
        self.assertEqual(len(m.obs_queue), 2)
        # Identify exactly which index of a predicted chunk is executed first.
        seen = []
        def fake_trajectory(obs_dict):
            seen.append(obs_dict["object"].clone())
            return torch.arange(8.).reshape(1, 8, 1).expand(1, 8, 7)
        m._get_action_trajectory = fake_trajectory
        m.reset()
        for i in range(9):
            obs = {k: torch.full((1, *shape), float(i)) for k, shape in self.shapes.items()}
            action = m.get_action(obs)
            self.assertEqual(action[0, 0].item(), float(i % 8))
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[-1][0, :, 0].tolist(), [7., 8.])

    def test_restore_training_state(self):
        m = self.model
        batch = {"obs": {k: torch.randn(2, 17, *s) for k, s in self.shapes.items()},
                 "actions": torch.rand(2, 17, 7) * 2 - 1}
        batch = m.process_batch_for_training(batch)
        m.set_train()
        for _ in range(4):
            m.train_on_batch(batch, epoch=1)
        saved = deepcopy(m.serialize())
        restored = self.make_model()
        restored.deserialize(saved, load_optimizers=True)
        restored.set_train()
        self.assertEqual(restored.ema.optimization_step, m.ema.optimization_step)
        self.assertTrue(restored.optimizers["policy"].state)
        torch.manual_seed(99)
        m.train_on_batch(batch, epoch=2)
        torch.manual_seed(99)
        restored.train_on_batch(batch, epoch=2)
        for key, value in m.nets.state_dict().items():
            torch.testing.assert_close(value, restored.nets.state_dict()[key])
        for key, value in m.ema.averaged_model.state_dict().items():
            torch.testing.assert_close(value, restored.ema.averaged_model.state_dict()[key])

    def test_single_frame_without_ema(self):
        self.config.train.frame_stack = 1
        self.config.algo.horizon.observation_horizon = 1
        self.config.algo.ema.enabled = False
        model = self.make_model()
        model.set_eval()
        action = model.get_action({k: torch.zeros(1, *s) for k, s in self.shapes.items()})
        self.assertEqual(action.shape, (1, 7))
        self.assertIsNone(model.serialize()["ema"])

    def test_dataset_frame_and_action_alignment(self):
        import tempfile
        from pathlib import Path
        import h5py
        import numpy as np
        from robomimic.utils.dataset import SequenceDataset
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.hdf5"
            with h5py.File(path, "w") as f:
                demo = f.create_group("data/demo_0")
                demo.attrs["num_samples"] = 24
                for key, shape in self.shapes.items():
                    demo.create_dataset("obs/" + key, data=np.broadcast_to(
                        np.arange(24, dtype=np.float32)[:, None], (24, *shape)))
                demo.create_dataset("actions", data=np.broadcast_to(
                    np.arange(24, dtype=np.float32)[:, None] / 24, (24, 7)))
            dataset = SequenceDataset(str(path), list(self.shapes), ["actions"], ["actions"],
                                      {"actions": {"normalization": None}},
                                      frame_stack=2, seq_length=16, load_next_obs=False)
            # Index 3 contains observation history [2, 3]. Action index To-1
            # must be the action at 3, which is the first executed prediction.
            sample = dataset[3]
            self.assertEqual(sample["obs"]["object"][:2, 0].tolist(), [2., 3.])
            self.assertAlmostEqual(float(sample["actions"][1, 0]), 3 / 24)
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            batch = self.model.process_batch_for_training(batch)
            batch = self.model.postprocess_batch_for_training(batch, obs_normalization_stats=None)
            info = self.model.train_on_batch(batch, epoch=1)
            self.assertTrue(torch.isfinite(info["losses"]["l2_loss"]))
            dataset.close_and_delete_hdf5_handle()


if __name__ == "__main__":
    unittest.main()
