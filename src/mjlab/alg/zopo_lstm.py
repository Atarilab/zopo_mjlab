import math
import torch
import torch.nn as nn
from torch.func import vmap, functional_call
from torch.distributions import Normal
from tensordict import TensorDict
from typing import Iterator, Tuple
from itertools import chain

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.models.rnn_model import RNNModel, RNN, HiddenState
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer, resolve_nn_activation

from .storage import Storage

# Force new API, avoids PyTorch/inductor conflict
torch.set_float32_matmul_precision("high")

# Optional: check current status
print("Matmul TF32 status:", torch.get_float32_matmul_precision())


import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input weights
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.02)
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))

        # Hidden weights
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.02)
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():

            if 'W_ih' in name:
                # Input → hidden
                torch.nn.init.xavier_uniform_(param)

            elif 'W_hh' in name:
                # Hidden → hidden (important for stability)
                torch.nn.init.orthogonal_(param)

            elif 'b_ih' in name or 'b_hh' in name:
                # Zero all biases first
                torch.nn.init.zeros_(param)

                # Set forget gate bias to 1
                # Layout: [i, f, g, o]
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

    def init_state(self, batch_size, device, dtype):
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return h, c

    def forward(self, x, state):
        """
        x: (B, input_size)
        state: (h, c) or None
        """
        h, c = state
        gates = (
            F.linear(x, self.W_ih, self.b_ih)
            + F.linear(h, self.W_hh, self.b_hh)
        )

        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, (h, c)

class ZOPO():
    actor: MLPModel | RNNModel
    """The actor model."""

    def __init__(
        self,
        actor: MLPModel | RNNModel,
        storage: Storage,
        gamma: float = 0.99,
        sigma: float = 0.01,
        lr: float = 1.0e-3,
        lr_sigma: float | None = 1.0e-4,
        use_ranks: bool = True,
        max_grad_norm: float = 1.0,
        antithetic: bool = True,
        optimizer: str = "adam",
        weight_decay: float = 0.01,
        normalize_returns: bool = True,
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        dtype : torch.dtype = torch.float32,
        # Dummy args to match ppo init
        rnd_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.normalize_returns = normalize_returns
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.symmetry = None
        self.rnd = None
        self.rnd_optimizer = None
        self.dtype=dtype

        # Add storage
        self.storage = storage
        self.transition = Storage.Transition()
        self.N = self.storage.num_envs

        # ZOPO parameters
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.learning_rate = lr
        self.learning_rate_sigma = lr_sigma if lr_sigma is not None else lr
        if lr_sigma is not None and lr_sigma <= 0.:
            self.learn_sigma = False
        else:
            self.learn_sigma = True

        self.use_ranks = use_ranks
        self.sigma = sigma
        if self.N % 2 == 0:
            self.antithetic = antithetic
            self.N_half = self.N // 2
        else:
            self.antithetic = False
            if antithetic:
                print("Number of envs shoud be a multiple of 2 to use antithetic sampling.")

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # ZOPO components
        self.actor = actor.to(self.device).to(dtype)
        self._init_weights()

        # Define vmap for MLP
        def mlp_forward(p, obs):
            return functional_call(self.actor.mlp, p, (obs,))
        self.vmap_mlp = torch.compile(vmap(mlp_forward, in_dims=(0, 0)))

        # Init samples
        self.params_samples = {
            k: torch.empty((self.N, *v.shape), device=self.device, dtype=self.dtype)
            for k, v in self.actor.mlp.named_parameters()
        }
        self.mlp_params_name = [k for k, _ in self.actor.mlp.named_parameters()]
        # Sigma parameters
        if self.learn_sigma:
            self._log_sig = {
                k: f"{k.replace('.', '')}_log_sigma" for k, _ in self.actor.mlp.named_parameters()
            }
            self.params_log_sigma = torch.nn.ParameterDict({
                self._log_sig[k]: torch.nn.Parameter(torch.tensor(self.sigma, device=self.device).log())
                for k, _ in self.actor.mlp.named_parameters()
            })

        # Define vmap for RNN, if exists
        self.with_rnn = hasattr(self.actor, "rnn")
        self.rnn_params_name = []
        if self.with_rnn:
            lstm : FunctionalLSTM = self.actor.rnn.rnn # type: ignore

            def lstm_forward(p, x, state):
                return functional_call(lstm, p, (x, state))
            self.vmap_rnn = torch.compile(vmap(lstm_forward, in_dims=(0, 0, 0)))

            # Init samples
            self.rnn_params_name = [k for k, _ in lstm.named_parameters()]
            self.params_samples.update({
                k: torch.empty((self.N, *v.shape), device=self.device, dtype=self.dtype)
                for k, v in lstm.named_parameters()
            })
            # Sigma parameters
            if self.learn_sigma:
                self._log_sig.update({
                    k: f"{k.replace('.', '')}_log_sigma" for k, _ in lstm.named_parameters()
                })
                self.params_log_sigma.update(torch.nn.ParameterDict({
                    self._log_sig[k]: torch.nn.Parameter(torch.tensor(self.sigma, device=self.device).log())
                    for k, _ in lstm.named_parameters()
                }))
        
        # Create optimizer
        optim_params = [
            {"params": self.actor.mlp.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]
        if self.with_rnn:
            lstm : FunctionalLSTM = self.actor.rnn.rnn # type: ignore
            optim_params.append(
                {"params": lstm.parameters(), "lr": lr, "weight_decay": weight_decay},
            )
        if self.learn_sigma:
            optim_params.append(
                {"params": self.params_log_sigma.parameters(), "lr": lr_sigma, "weight_decay": 0.0}
            )
        self.optimizer = resolve_optimizer(optimizer)( # type: ignore
            optim_params,
            maximize=True,
            )
        
        self.all_params_name = self.mlp_params_name + self.rnn_params_name
        self.sample_params()

    def _init_weights(self) -> None:
        layers = [
            m for _, m in
            self.actor.mlp.named_modules(remove_duplicate=False)
        ]

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                gain = 1.0
                # Look ahead to next activation
                if i + 1 < len(layers):
                    next_layer = layers[i + 1]
                    if isinstance(next_layer, nn.LeakyReLU):
                        slope = next_layer.negative_slope
                        gain = nn.init.calculate_gain("leaky_relu", slope)
                    if isinstance(next_layer, nn.ReLU):
                        gain = nn.init.calculate_gain("relu")
                    elif isinstance(next_layer, nn.Tanh):
                        gain = nn.init.calculate_gain("tanh")
                    elif isinstance(next_layer, nn.SELU) or isinstance(next_layer, nn.ELU):
                        gain = nn.init.calculate_gain("selu")
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def get_latent(self, obs: TensorDict) -> torch.Tensor:
        """Overrides RSL-RL logic to handle vmap and RNN."""
        # Select and concatenate observations
        obs_list = [obs[obs_group] for obs_group in self.actor.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        # Normalize observations
        latent = self.actor.obs_normalizer(latent)

        if self.with_rnn:
            self.actor: RNNModel
            # Pass through RNN and update hidden state
            if self.actor.rnn.hidden_state is None:
                lstm : FunctionalLSTM = self.actor.rnn.rnn # type: ignore
                self.actor.rnn.hidden_state = lstm.init_state(latent.shape[0], device=obs.device, dtype=obs.dtype)
            latent, self.actor.rnn.hidden_state = self.vmap_rnn(
                {k: self.params_samples[k] for k in self.rnn_params_name},
                latent,
                self.actor.rnn.hidden_state
                )

        return latent
    
    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        # Compute the actions and values
        if self.actor.training:
            # Get obs processed
            latent = self.get_latent(obs)
            # vmap MLP forward pass
            action = self.vmap_mlp(self.params_samples, latent).squeeze()
            return action
        else:
            # Use the standard forward actor
            return self.actor(obs, stochastic_output=False)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "ZOPO":
        """Construct ZOPO algorithm."""
        # Resolve class callables
        alg_class: type[ZOPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        
        if cfg["actor"].pop("rnn_type", None) == "lstm":
            lstm = FunctionalLSTM(actor.obs_dim, cfg["actor"]["rnn_hidden_dim"])
            actor.rnn.rnn = lstm # type: ignore

        last_activation_str = cfg["algorithm"].pop("last_activation")
        if last_activation_str is not None:
            last_activation = resolve_nn_activation(last_activation_str)
            actor.add_module("last_activation", last_activation)
        print(f"Actor Model: {actor}")

        # Initialize the storage
        storage = Storage(env.num_envs, cfg["num_steps_per_env"], device)
        
        # Initialize the algorithm
        alg: ZOPO = alg_class(actor, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        return alg
        
    def train_mode(self) -> None:
        self.actor.train()
    
    def eval_mode(self) -> None:
        self.actor.eval()

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.actor.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards
        # Add rewards of the last episode if not done
        self.transition.dones = dones

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        pass

    def sample_params(self):
        for k, p in self.all_parameters():
            if self.learn_sigma:
                sigma = torch.exp(self.params_log_sigma[self._log_sig[k]]).item()
            else:
                sigma = self.sigma
            
            # in-place param update
            if not self.antithetic:
                self.params_samples[k].normal_(std=sigma).add_(p)
            else:
                self.params_samples[k][:self.N_half].normal_(std=sigma)
                self.params_samples[k][self.N_half:].copy_(-self.params_samples[k][:self.N_half])
                self.params_samples[k].add_(p)

    def all_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        if self.with_rnn:
            return chain(self.actor.mlp.named_parameters(), self.actor.rnn.rnn.named_parameters())
        else:
            return self.actor.mlp.named_parameters()
        
    def set_grad(self, returns: torch.Tensor) -> float:
        total_elems = 0.
        l2_sq_params = 0.

        for k, p in self.all_parameters():
            # Recover epsilon WITHOUT modifying buffer
            self.params_samples[k].sub_(p.unsqueeze(0))
            r = returns.view(-1, *([1] * p.ndim))

            if self.learn_sigma:
                sigma = torch.exp(self.params_log_sigma[self._log_sig[k]])
            else:
                sigma = self.sigma
            
            p.grad = torch.mean(r * self.params_samples[k], dim=0) / sigma
            if self.learn_sigma:
                # chain rule: d/d logσ = σ * d/dσ
                # grad = r * sigma * (eps ** 2 - sigma ** 2) / sigma ** 3
                # grad = r * (eps ** 2 / sigma ** 2 - 1)
                self.params_samples[k].div_(self.sigma).square_().sub_(1.)
                self.params_log_sigma[self._log_sig[k]].grad = torch.mean(r * self.params_samples[k])

            # bookkeeping
            l2_sq_params += p.square().sum().item()
            total_elems += p.grad.numel()

        return l2_sq_params / total_elems
    
    @torch.no_grad()
    def update(self) -> dict[str, float]:
        # No update if no episode done
        dones_frac = self.storage.has_episode_done.float().mean().item()
        std_r = self.storage.returns.std()
        mean_r = self.storage.returns.mean()
        
        self.optimizer.zero_grad()

        # Remove baseline

        if self.use_ranks:
            # Get ranks (0 = worst, N-1 = best)
            ranks = torch.argsort(torch.argsort(self.storage.returns))
            # Convert to [0, 1]
            ranks = ranks.float() / (len(self.storage.returns) - 1)
            # Center to [-0.5, 0.5]
            r = ranks - 0.5
        else:
            if self.normalize_returns:
                r = (self.storage.returns - mean_r) / (std_r + 1e-8)
            else:
                r = (self.storage.returns - mean_r)

        # Compute gradients
        param_norm2 = self.set_grad(r)
        # Clip grads
        grads = [p.grad for p in self.actor.mlp.parameters() if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(grads)

        if self.max_grad_norm > 0.:
            torch.nn.utils.clip_grads_with_norm_(
                self.actor.mlp.parameters(),
                self.max_grad_norm,
                total_norm
            )

        # torch.nn.utils.clip_grad_norm_(self.actor.mlp.parameters(), self.max_grad_norm)
        # Gradient descene step
        self.optimizer.step()
        # Sample new params for next rollout
        self.sample_params()
        # Clear storage
        self.storage.clear()
        self.actor.reset(torch.ones(self.N, dtype=torch.long, device=self.device))

        info_dict = {
            "grad_total_norm": total_norm.item(),
            "param_l2": param_norm2,
            "mean_r": mean_r.item(),
            "std_r": std_r.item(),
            "dones_frac": dones_frac,
        }

        if self.learn_sigma:
            for p in self.params_log_sigma.values():
                p.data.clamp_(-7.0, -3.0)
            sigma_dict = {
                f"sigma/{k}": v.exp().item()
                for k, v in self.params_log_sigma.items()
            }
            info_dict.update(sigma_dict)

        return info_dict
    
    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.mlp.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self.actor.mlp.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self.actor

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.actor.mlp.state_dict()]
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.actor.mlp.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = list(self.actor.mlp.parameters())
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
