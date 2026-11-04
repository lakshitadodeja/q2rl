"""
Flow Matching adaptation of the official robomimic Diffusion Policy.
Original: https://diffusion-policy.cs.columbia.edu/ by Cheng Chi.
Source commit: d309eaecc18acf4152a830a895a6984b8ac71b05
Original method order, docstrings, observation encoding, action chunks, EMA,
and checkpoint interfaces retained. FLOW CHANGE marks algorithm replacements;
COMPATIBILITY marks adaptations to the older q2rl robomimic interfaces.
See FLOW_POLICY.md and models/FLOW_POLICY_LICENSE.txt.
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# COMPATIBILITY: vendor the original diffusers 0.11.1 EMA class only.
from robomimic.models.flow_ema import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.flow_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("flow_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return FlowPolicy, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class FlowPolicy(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self._validate_config()
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)

        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        velocity_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=list(self.algo_config.unet.down_dims),
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups,
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "velocity_net": velocity_net
            })
        })

        nets = nets.float().to(self.device)

        # FLOW CHANGE: no DDPM/DDIM noise scheduler is needed.

        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)

        # set attrs
        self.nets = nets
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.reset()

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        if batch["actions"].shape[1] < Tp:
            raise ValueError("Dataset action sequence is shorter than prediction horizon")
        if any(v.shape[1] < To for v in batch["obs"].values()):
            raise ValueError("Dataset observations are shorter than observation horizon")
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Flow Policy; supply normalized robosuite actions.")
            self.action_check_done = True

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]


        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(FlowPolicy, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]

            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)

            # FLOW CHANGE 1: straight-line conditional flow matching.
            # t=0 is Gaussian noise; t=1 is the demonstration action sequence.
            noise = torch.randn_like(actions)
            time = torch.rand((B,), device=self.device, dtype=actions.dtype)
            noisy_actions, target_velocity = flow_matching_targets(actions, noise, time)
            velocity_pred = self.nets["policy"]["velocity_net"](
                noisy_actions, time * self.algo_config.flow.time_scale,
                global_cond=obs_cond)
            loss = F.mse_loss(velocity_pred, target_velocity)

            # logging
            losses = {
                "l2_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )

                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)

                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(FlowPolicy, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue

    @torch.no_grad()
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        # COMPATIBILITY: q2rl's RolloutPolicy passes one observation, not a
        # frame stack. Build the same past->current ordering as SequenceDataset.
        # Also accept an already-stacked observation (upstream-style callers).
        if goal_dict is not None:
            raise ValueError("Goal-conditioned observations are not configured")
        obs_dict = self._prepare_observation_history(obs_dict)

        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)

            # put actions into the queue
            self.action_queue.extend(action_sequence[0])

        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()

        # [1,Da]
        action = action.unsqueeze(0)
        return action

    @torch.no_grad()
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        num_inference_timesteps = self.algo_config.flow.num_inference_timesteps

        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model

        # encode obs
        inputs = {
            "obs": dict(obs_dict),
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action

        # FLOW CHANGE 2: Euler integration from noise (0) to actions (1).
        dt = 1.0 / num_inference_timesteps
        for k in range(num_inference_timesteps):
            time = naction.new_full((B,), k * dt * self.algo_config.flow.time_scale)
            velocity_pred = nets["policy"]["velocity_net"](
                sample=naction, timestep=time, global_cond=obs_cond)
            naction = naction + dt * velocity_pred
        if self.algo_config.flow.clip_sample:
            naction = naction.clamp(-1, 1)

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
            # Also retain the warmup counter when restoring training state.
            "ema_optimization_step": self.ema.optimization_step if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            if self.ema is None:
                raise ValueError("Checkpoint uses EMA but the configuration disables it")
            self.ema.averaged_model.load_state_dict(model_dict["ema"])
            self.ema.optimization_step = model_dict.get("ema_optimization_step", 0) or 0

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])

        self.reset()

    def _validate_config(self):
        """Reject mismatched dataset/history/chunk settings before training."""
        cfg = self.algo_config
        To, Ta, Tp = (cfg.horizon.observation_horizon,
                      cfg.horizon.action_horizon, cfg.horizon.prediction_horizon)
        if To < 1 or Ta < 1 or To - 1 + Ta > Tp:
            raise ValueError("Need To >= 1, Ta >= 1 and To - 1 + Ta <= Tp")
        if self.global_config.train.frame_stack != To:
            raise ValueError("train.frame_stack must equal observation_horizon")
        if self.global_config.train.seq_length < Tp:
            raise ValueError("train.seq_length must cover prediction_horizon")
        dims = list(cfg.unet.down_dims)
        if len(dims) < 2 or Tp % (2 ** (len(dims) - 1)):
            raise ValueError("Prediction horizon must match U-Net downsampling factor")
        if cfg.flow.num_inference_timesteps < 1 or cfg.flow.time_scale <= 0:
            raise ValueError("Flow sampling steps and time scale must be positive")
        if self.goal_shapes:
            raise ValueError("The upstream observation encoder is not goal-conditioned")

    def _prepare_observation_history(self, obs_dict):
        """Pad initial history by repeating the first frame, as in the dataset."""
        To = self.algo_config.horizon.observation_horizon
        obs = {k: obs_dict[k] for k in self.obs_shapes}
        if any(v.shape[0] != 1 for v in obs.values()):
            raise ValueError("Action queues support one environment at a time")
        if all(v.ndim == len(self.obs_shapes[k]) + 2 for k, v in obs.items()):
            if any(v.shape[1] != To for v in obs.values()):
                raise ValueError("Stacked observation has wrong history length")
            self.obs_queue.clear()
            for i in range(To):
                self.obs_queue.append({k: v[:, i].detach().clone() for k, v in obs.items()})
            return obs
        if any(v.ndim != len(self.obs_shapes[k]) + 1 for k, v in obs.items()):
            raise ValueError("Expected a single observation or a consistent frame stack")
        self.obs_queue.append({k: v.detach().clone() for k, v in obs.items()})
        while len(self.obs_queue) < To:
            self.obs_queue.appendleft(self.obs_queue[0])
        return {k: torch.stack([frame[k] for frame in self.obs_queue], dim=1) for k in obs}


def flow_matching_targets(actions, noise, time):
    """Straight-line sample and velocity target, with 0=noise and 1=data."""
    t = time[:, None, None]
    return (1 - t) * noise + t * actions, actions - noise


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module
