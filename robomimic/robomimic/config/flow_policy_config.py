"""Flow configuration sharing the pinned upstream Diffusion training defaults."""
from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class FlowPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "flow_policy"

    def algo_config(self):
        super().algo_config()
        del self.algo["ddpm"]
        del self.algo["ddim"]
        self.algo.flow.num_inference_timesteps = 10
        self.algo.flow.time_scale = 1000.0
        self.algo.flow.clip_sample = True
