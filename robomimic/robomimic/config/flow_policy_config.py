"""Low-dimensional, action-chunked flow-matching BC configuration."""
from robomimic.config.bc_config import BCConfig


class FlowPolicyConfig(BCConfig):
    ALGO_NAME = "flow_policy"

    def train_config(self):
        super().train_config()
        self.train.seq_length = 16
        self.train.frame_stack = 2
        self.train.batch_size = 64
        self.train.num_epochs = 100  # initial pilot, not a tuned reproduction
        self.train.dataset_keys = ("actions",)

    def algo_config(self):
        # Retain the optimizer schema expected by this version of robomimic.
        super().algo_config()
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.regularization.L2 = 1e-6
        # This older q2rl base lacks upstream per-batch cosine LR scheduling.
        # Keep its constant-LR (empty multistep schedule) configuration explicit.
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.prediction_horizon = 16
        self.algo.horizon.action_horizon = 8
        self.algo.flow.num_inference_timesteps = 10
        self.algo.flow.time_scale = 1000.0
        self.algo.flow.clip_sample = True
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75
        self.algo.unet.enabled = True
        self.algo.unet.down_dims = (256, 512, 1024)
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8
