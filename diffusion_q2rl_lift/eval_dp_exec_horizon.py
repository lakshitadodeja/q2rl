"""Success rate of Diffusion Policy checkpoints under a given execution horizon, run through the
same components Q2RL uses (DiffusionBCQuery + ResetsBC on the raw robosuite env).

Agreed setting for the Q2RL experiments: the policy predicts its chunk and executes only the
first action (exec_horizon=1), and episodes last 300 steps (Q2RL's --max_traj_length).
Writes one JSON per checkpoint (appending as it goes) so partial results survive.

Needs PYTHONPATH=<this repo>/diffusion_q2rl_lift/pypath and USE_FLAX=0 (see run_q2rl_dp.sbatch).
"""
import argparse
import collections
import json
import os
import time

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from backend.common.dp_bc_query import DiffusionBCQuery, ResetsBC, raw_robosuite_env

parser = argparse.ArgumentParser()
parser.add_argument("--ckpts", nargs="+", required=True)
parser.add_argument("--exec_horizon", type=int, default=1)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--horizon", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_rm(o):   # same conversion as q2rl_robosuite.py
    return ObsUtils.process_obs_dict(collections.OrderedDict(
        robot0_eef_pos=o[:3], robot0_eef_quat=o[3:7], robot0_gripper_qpos=o[7:9], object=o[9:]))


for ckpt in args.ckpts:
    name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ckpt)))) + "__" + \
        os.path.basename(ckpt).split("_low_dim")[0].replace(".pth", "")
    out = os.path.join(args.out_dir, f"{name}__exec{args.exec_horizon}.json")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent, ck = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
    cfg = json.loads(ck["config"]) if isinstance(ck["config"], str) else ck["config"]
    bc = DiffusionBCQuery(agent, cfg["algo"]["horizon"]["observation_horizon"], exec_horizon=args.exec_horizon)
    rm_env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ck, render=False, render_offscreen=False, verbose=False)
    suite = raw_robosuite_env(rm_env)
    suite.ignore_done = False
    suite.reward_shaping = False
    suite.horizon = args.horizon
    env = ResetsBC(suite, bc, reward_scale=1.0, reward_bias=0.0)

    rec = dict(ckpt=ckpt, exec_horizon=bc.exec_horizon,
               prediction_horizon=cfg["algo"]["horizon"]["prediction_horizon"],
               trained_action_horizon=cfg["algo"]["horizon"]["action_horizon"],
               horizon=args.horizon, seed=args.seed, episodes=[])
    t0 = time.time()
    for ep in range(args.episodes):
        o, _ = env.reset()
        success, steps = False, 0
        for steps in range(1, args.horizon + 1):
            a, _, _ = bc(ob=to_rm(o), log_prob=True)
            o, r, done, trunc, info = env.step(a)
            if suite._check_success():
                success = True
                break
            if done:
                break
        rec["episodes"].append(dict(success=success, steps=steps))
        k = sum(e["success"] for e in rec["episodes"])
        rec.update(n=ep + 1, successes=k, success_rate=k / (ep + 1), minutes=(time.time() - t0) / 60)
        json.dump(rec, open(out, "w"), indent=1)
    print(f"RESULT {name} exec{bc.exec_horizon}: {rec['successes']}/{rec['n']} = {rec['success_rate']:.2f} "
          f"({rec['minutes']:.0f} min)", flush=True)
