"""Placeholder for 'mj_envs' (not installed in the q2rl-dp env).

backend/envs/env_common.py imports d4rl and mj_envs at module level only to register the
D4RL / Adroit gym environments. The robosuite-state Q2RL path never uses them, but it
imports env_common (via backend/data/replay_buffer.py for calc_return_to_go). This module
satisfies that import and fails loudly if anything actually uses it.
"""


def __getattr__(name):
    raise ImportError("'mj_envs' is a placeholder in q2rl_pypath; the real package is not installed "
                      f"(attribute '{name}' requested). It is only needed for D4RL/Adroit tasks.")
