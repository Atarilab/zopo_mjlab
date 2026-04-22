import torch
from dataclasses import dataclass

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp.commands import MotionCommand

@dataclass(kw_only=True)
class MotionTrackingJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for motion tracking joint position control."""

    command_name: str = "motion"

    def __post_init__(self):
        self.transmission_type = TransmissionType.JOINT

    def build(self, env: ManagerBasedRlEnv) -> "MotionTrackingJointPositionAction":
        return MotionTrackingJointPositionAction(self, env)

class MotionTrackingJointPositionAction(JointPositionAction):
    """Control joints via position targets with motion tracking offset."""

    _motion_command: MotionCommand

    def __init__(
        self,
        cfg: MotionTrackingJointPositionActionCfg,
        env: ManagerBasedRlEnv,
    ):
        super().__init__(cfg=cfg, env=env)

        # Get motion command from command manager
        command_name = cfg.command_name

        from mjlab.tasks.tracking.mdp.commands import MotionCommand

        motion_command = env.command_manager.get_term(command_name)
        if not isinstance(motion_command, (MotionCommand)):
            raise TypeError(
                f"Command '{command_name}' must be a MotionCommand or MultiMotionCommand, got {type(motion_command)}"
            )
        self._motion_command = motion_command

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions by applying scale and offset from motion command."""
        # Update offset from motion tracking command if available
        if self._motion_command is not None:
            # Get current joint positions from motion command
            motion_joint_pos = self._motion_command.joint_pos  # (num_envs, num_joints)
            # Select the joints that this action uses
            self._offset = motion_joint_pos[:, self._target_ids].clone()

        # Process actions with updated offset
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = self._processed_actions - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)