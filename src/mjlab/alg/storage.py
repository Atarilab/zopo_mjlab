import torch

class Storage:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    class Transition:
        """Storage for a single state transition."""

        def __init__(self) -> None:
            self.rewards: torch.Tensor
            self.dones: torch.Tensor

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        device: str = "cpu",
    ) -> None:
        self.step = 0
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.returns = torch.zeros(num_envs, device=self.device)
        self.has_episode_done = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

    def add_transition(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        
        # If has one episode done, do not add reward anymore
        self.returns[~self.has_episode_done] += transition.rewards[~self.has_episode_done]

        if self.step > 0:
            self.has_episode_done |= transition.dones.bool()

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        self.returns.zero_()
        self.has_episode_done.zero_()
        self.step = 0
