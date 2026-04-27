"""RL configuration for Unitree G1 tracking task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.alg.zopo import ZOPO

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  ZOPOAlgorithmCfg,
)

def unitree_g1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )

def unitree_g1_tracking_zopo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    class_name="tasks.tracking.rl.runner.MotionTrackingOnPolicyRolloutRunner",
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=None,
    ),
    critic={}, # type: ignore
    obs_groups= {"actor": ("actor",)},
    algorithm=ZOPOAlgorithmCfg(
      gamma=0.99,
      sigma=0.01,
      lr=0.003,
      lr_sigma=0.001,
      use_ranks=True,
      max_grad_norm=0.0,
      weight_decay=0.01,
      normalize_returns=True,
      antithetic=True,
      optimizer="adamw",
      class_name="mjlab.alg.zopo.ZOPO",
      last_activation=None
    ), # type: ignore
    experiment_name="g1_tracking",
    save_interval=20,
    num_steps_per_env=256,
    max_iterations=30_000,
  )
