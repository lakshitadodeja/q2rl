"""Regression check for standalone evaluation of frame-stacked checkpoints."""
import unittest
from unittest.mock import MagicMock
import numpy as np

from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import FrameStackWrapper
from robomimic.scripts.run_trained_agent import rollout


class WrappedEvaluationTest(unittest.TestCase):
    def test_base_and_frame_stacked_rollouts(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                base = MagicMock(spec=EnvBase)
                base.action_dimension = 3
                base.rollout_exceptions = ()
                observation = {"state": np.zeros(4)}
                base.reset.return_value = observation.copy()
                base.reset_to.return_value = observation.copy()
                base.get_state.return_value = {"states": np.zeros(5)}
                base.step.side_effect = [
                    ({"state": np.ones(4)}, 0., False, {}),
                    ({"state": np.ones(4) * 2}, 1., False, {}),
                ]
                base.is_success.side_effect = [{"task": False}, {"task": True}]
                env = FrameStackWrapper(base, num_frames=2) if wrapped else base
                policy = MagicMock(spec=RolloutPolicy)
                policy.return_value = np.zeros(3)
                stats, trajectory = rollout(policy, env, horizon=10)
                self.assertEqual(stats["Success_Rate"], 1.)
                self.assertEqual(stats["Horizon"], 2)
                self.assertEqual(trajectory["actions"].shape, (2, 3))
                first_obs = policy.call_args_list[0].kwargs["ob"]["state"]
                self.assertEqual(first_obs.shape, (2, 4) if wrapped else (4,))
                policy.start_episode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
