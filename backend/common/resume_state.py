"""Save / load everything needed to continue a Q2RL online run in a later Slurm job.

``--resume_path_agent`` only restores the network parameters; it does not restore the online
replay buffer, and the main loop derives its step counter from ``agent.state.step`` (the number
of gradient updates, which is utd x online steps + Q-estimation steps), so it cannot continue an
online run. This module stores, in one pickle:

- the agent (flax state dict: networks, target networks, optimizer states, update count),
- the online step to execute next and the main loop's jax rng,
- the online replay buffer up to its last completed trajectory, with its sampling RNG,
- numpy and torch global RNG states (the Diffusion Policy samples its noise from torch).

The trajectory that is still running when the state is saved is left out: its transitions
were never sampleable (Q2RL's buffer only samples completed trajectories, because it needs
their Monte Carlo returns), and keeping a half trajectory would make the buffer join it to the
first trajectory of the resumed run. The resumed run starts with a fresh episode.
"""
import os
import pickle

import flax
import jax
import numpy as np
import torch


def _slice(d, n):
    if isinstance(d, dict):
        return {k: _slice(v, n) for k, v in d.items()}
    return np.asarray(d[:n]).copy()


def _fill(dst, src):
    if isinstance(dst, dict):
        for k in dst:
            _fill(dst[k], src[k])
    else:
        n = len(src)
        dst[:n] = src


def save_resume_state(path, next_step, rng, agent, buffer):
    n = int(buffer._traj_start_idx)            # end of the last completed trajectory
    state = dict(
        next_step=int(next_step),
        rng=np.asarray(jax.device_get(rng)),
        agent=jax.device_get(flax.serialization.to_state_dict(agent)),
        buffer_n=n,
        buffer_data=_slice(buffer.dataset_dict, n),
        buffer_allow_idxs=[i for i in buffer._allow_idxs if i < n],
        # state dict, not the Generator object: gym 0.23 patches Generator pickling so the object
        # cannot be unpickled again
        buffer_rng_state=buffer.np_random.bit_generator.state,
        dropped_partial_transitions=int(buffer._insert_index) - n,
        np_state=np.random.get_state(),
        torch_state=torch.random.get_rng_state(),
        torch_cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=4)
    os.replace(tmp, path)                     # never leave a half-written file behind
    return state["dropped_partial_transitions"]


def load_resume_state(path, agent, buffer):
    """Returns (agent, next_step, rng); fills ``buffer`` in place."""
    with open(path, "rb") as f:
        s = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, s["agent"])
    n = s["buffer_n"]
    _fill(buffer.dataset_dict, s["buffer_data"])
    buffer._size = buffer._insert_index = buffer._traj_start_idx = n
    buffer._allow_idxs = list(s["buffer_allow_idxs"])
    buffer.np_random.bit_generator.state = s["buffer_rng_state"]
    np.random.set_state(s["np_state"])
    torch.random.set_rng_state(s["torch_state"])
    if s["torch_cuda_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["torch_cuda_state"])
    print(f"RESUMED from {path}: next online-loop step {s['next_step']}, "
          f"agent updates {int(agent.state.step)}, buffer {n} transitions "
          f"({len(buffer._allow_idxs)} sampleable; dropped {s['dropped_partial_transitions']} "
          f"from the unfinished trajectory)", flush=True)
    return agent, s["next_step"], jax.numpy.asarray(s["rng"])
