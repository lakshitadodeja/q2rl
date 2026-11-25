"""Adapter that lets a robomimic Diffusion Policy checkpoint stand in for Q2RL's GMM BC policy.

Q2RL calls ``bc_query(ob=<single-frame robomimic obs>, log_prob=True)`` and expects
``(action, log_prob, entropy)``. A robomimic Diffusion Policy differs in three ways:

1. No log-prob / entropy. Phase 1 of the Diffusion/Flow extension uses V_BC only, so both
   are returned as 0.0 (TEMPORARY, see ``__call__``). With zeros, Q2RL's
   ``target_q_bc = mc_returns + bc_probs + bc_entropy`` and ``bc_q = v + bc_log_prob +
   bc_entropy`` (backend/agents/q2rl_robosuite.py) reduce to the Monte Carlo return and V.
2. Observation history. The policy was trained with ``observation_horizon`` (To) stacked
   frames; robomimic normally gets them from FrameStackWrapper. Here the last To frames are
   kept and the first frame of an episode is repeated, as FrameStackWrapper does on reset.
3. Action chunking. The policy predicts ``prediction_horizon`` actions (starting one step in
   the past) and executes ``action_horizon`` of them, starting at the current step, from a
   queue before replanning. ``exec_horizon`` overrides how many are executed at inference;
   ``exec_horizon=1`` predicts the chunk and executes only its first action, i.e. replans
   every step (the setting agreed for the Diffusion/Flow Q2RL experiments). ``drop_plan()``
   empties the queue so the next query replans (--dp_replan_on_rl_action).

Each instance must belong to exactly one environment: call ``reset()`` whenever that
environment resets (``ResetsBC`` does this automatically). The finetune env and the eval
env therefore need separate instances, each with its own loaded policy, because the action
queue lives inside the policy object.
"""
import collections

import numpy as np

from backend.envs.wrappers.robosuite_state import RobosuiteStateWrapper


class DiffusionBCQuery:
    def __init__(self, rollout_policy, observation_horizon, exec_horizon=None):
        self.policy = rollout_policy            # robomimic RolloutPolicy
        self.To = int(observation_horizon)
        self._frames = collections.deque(maxlen=self.To)
        algo_cfg = rollout_policy.policy.algo_config
        if exec_horizon is not None:
            # Only the number of executed actions changes; the network still predicts the
            # same chunk. The action queue is rebuilt with this length on every reset().
            assert 1 <= exec_horizon <= algo_cfg.horizon.action_horizon, exec_horizon
            with algo_cfg.values_unlocked():
                algo_cfg.horizon.action_horizon = int(exec_horizon)
        self.exec_horizon = int(algo_cfg.horizon.action_horizon)

    def reset(self):
        """Start a new episode: clear the action queue and the observation history."""
        self.policy.start_episode()
        self._frames.clear()

    def drop_plan(self):
        """Discard the remaining planned actions; the next query replans."""
        self.policy.policy.action_queue.clear()

    def _stacked(self, ob):
        if not self._frames:
            self._frames.extend([ob] * self.To)
        else:
            self._frames.append(ob)
        return {k: np.stack([f[k] for f in self._frames]) for k in ob}

    def __call__(self, ob, log_prob=True, goal=None):
        action = self.policy(ob=self._stacked(ob), goal=goal)
        if not log_prob:
            return action
        # TODO(phase 1, V_BC only): Diffusion Policy has no tractable action
        # log-probability or entropy, so both are set to 0. This is NOT the true value.
        # Replace with real estimates once the method for them is decided.
        return action, 0.0, 0.0


def raw_robosuite_env(rm_env):
    """Return the robosuite environment inside a robomimic env from ``env_from_checkpoint``.

    Newer robomimic (0.5.0 @ d309eae) wraps the env in FrameStackWrapper when the policy uses
    frame stacking (Diffusion Policy: 2 frames), so ``rm_env.env`` is robomimic's EnvRobosuite,
    not the robosuite env Q2RL expects. Q2RL's ignore_done / reward_shaping / horizon settings
    must go on the robosuite env, otherwise they silently have no effect. Frame stacking is
    handled by DiffusionBCQuery instead, so skipping the wrapper is safe.
    """
    from robomimic.envs.env_robosuite import EnvRobosuite

    e = rm_env
    while not isinstance(e, EnvRobosuite):
        e = e.env
    return e.env


class ResetsBC(RobosuiteStateWrapper):
    """RobosuiteStateWrapper that also resets its BC query object on every env reset."""

    def __init__(self, env, bc_query, **kwargs):
        self.bc_query = bc_query
        super().__init__(env, **kwargs)

    def reset(self, seed=None, options=None):
        self.bc_query.reset()
        return super().reset(seed=seed, options=options)
