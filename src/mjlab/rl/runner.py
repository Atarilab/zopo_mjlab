import os
import time
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self.zero_order = train_cfg["algorithm"].pop("zero_order", False)
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)

  def learn_zero_order(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    # Randomize initial episode lengths (for exploration)
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )

    # Start learning
    obs = self.env.get_observations().to(self.device)
    at_least_done_once = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
    dones_ = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
    self.alg.train_mode()  # switch to train mode (for dropout for example)

    # Ensure all parameters are in-synced
    if self.is_distributed:
        print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
        self.alg.broadcast_parameters()

    # Initialize the logging writer
    self.logger.init_logging_writer()
    
    # Start training
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
        start = time.time()
        # Rollout
        with torch.no_grad():
            # Rollout for num_steps_per_env.
            # Stop rollout if all envs has been terminated once.
            obs = self.env.reset()[0].to(self.device)
            for i in range(self.cfg["num_steps_per_env"]):
                # Sample actions
                actions = self.alg.act(obs)
                # Step the environment
                obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                # Move to device
                obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                # Process the step
                if i == self.cfg["num_steps_per_env"] - 1:
                    # Terminate if num_steps_per_env is reached
                    dones = (~at_least_done_once).long()
                    # self.alg.process_env_step(obs, rewards, (~at_least_done_once).long(), extras)

                self.alg.process_env_step(obs, rewards, dones, extras)
                    
                # Extract intrinsic rewards if RND is used (only for logging)
                intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                # Book keeping, only for the first episode done
                dones_ = dones.bool() & ~at_least_done_once
                self.logger.process_env_step(rewards, dones_, extras, intrinsic_rewards)
                if i > 0:
                    at_least_done_once = at_least_done_once | dones.bool()
                    if at_least_done_once.all():
                        at_least_done_once.fill_(False)
                        break

            stop = time.time()
            collect_time = stop - start
            start = stop

            # Compute returns
            self.alg.compute_returns(obs)

        # Update policy
        loss_dict = self.alg.update()

        stop = time.time()
        learn_time = stop - start
        self.current_learning_iteration = it

        # Log information
        policy = self.alg.get_policy()
        self.logger.log(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=self.alg.learning_rate,
            action_std=policy.output_std if policy.distribution is not None else torch.zeros(1),
            rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,  # type: ignore
        )

        # Reset env steps count
        dones_.fill_(True)
        self.logger.process_env_step(rewards, dones_, extras, intrinsic_rewards) # type: ignore
        # Reset buffers
        self.logger.rewbuffer.clear()
        self.logger.lenbuffer.clear()

        # Save model
        if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
            self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

    # Save the final model after training and stop the logging writer
    if self.logger.writer is not None:
        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
        self.logger.stop_logging_writer()

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if self.zero_order:
      self.learn_zero_order(num_learning_iterations, init_at_random_ep_len)
    else:
      super().learn(num_learning_iterations, init_at_random_ep_len)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
