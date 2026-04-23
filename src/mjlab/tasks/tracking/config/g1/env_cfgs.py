"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.tracking.mdp.actions import MotionTrackingJointPositionActionCfg

def no_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Remove randomization config terms inplace."""
  cfg.observations["actor"].enable_corruption = False
  cfg.events = {}
  motion_cmd : MotionCommandCfg = cfg.commands["motion"] # type: ignore
  motion_cmd.pose_range={}
  motion_cmd.joint_position_range=(0., 0.)
  motion_cmd.velocity_range={}

def improved_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Modify reward terms inplace."""
  cfg.rewards["self_collisions"].weight = -1.0
  cfg.rewards["motion_global_root_pos"].weight = 4.0
  
def use_residual_pd_targets(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.actions["joint_pos"] = MotionTrackingJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=G1_ACTION_SCALE,
    command_name="motion",
  )

def no_critic(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.observations.pop("critic")

def relaxed_termination(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.terminations["anchor_ori"].params["threshold"] = 0.8
  cfg.terminations["anchor_pos"].params["threshold"] = 0.4
  cfg.terminations["ee_body_pos"].func = mdp.bad_motion_body_pos
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.4

def make_domain_rand_antithetic(cfg: ManagerBasedRlEnvCfg) -> None:
  for event_name in cfg.events.keys():
    cfg.events[event_name].params["antithetic"] = True
    cfg.events[event_name].antithetic = True
  motion_command : MotionCommandCfg = cfg.commands["motion"] # type: ignore
  motion_command.antithetic = True

def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  with_critic: bool = True,
  randomize: bool = True,
  residual_action: bool = True,
  antithetic_domain_rand: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  improved_rewards(cfg)
  relaxed_termination(cfg)

  if antithetic_domain_rand:
    make_domain_rand_antithetic(cfg)

  if not randomize:
    no_randomization(cfg)

  if not with_critic:
    no_critic(cfg)

  if residual_action:
    use_residual_pd_targets(cfg)

  return cfg
