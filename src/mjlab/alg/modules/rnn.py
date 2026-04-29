# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Union, Tuple

from rsl_rl.utils import unpad_trajectories


class FunctionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.ih.weight)
        nn.init.zeros_(self.ih.bias)

        nn.init.orthogonal_(self.hh.weight)
        nn.init.zeros_(self.hh.bias)

        # Forget gate bias = 1
        H = self.hidden_size
        with torch.no_grad():
            self.ih.bias[H:2*H].fill_(1.0)
            self.hh.bias[H:2*H].fill_(1.0)

    def init_state(self, batch_size, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return h, c

    def forward(self, x: torch.Tensor, state: HiddenState) -> Tuple[torch.Tensor, HiddenState]:
        """
        x: (B, input_size)
        state: tuple
            - (B, hidden_size)
            - (B, hidden_size)
        """
        if state is None:
            B = x.shape[0]
            h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        gates = self.ih(x) + self.hh(h)

        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)
    

HiddenState = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor], None]  # Using Union due to Python <3.10
"""Type alias for the hidden state of RNNs (GRU/LSTM).

For GRUs, this is a single tensor while for LSTMs, this is a tuple of two tensors (hidden state and cell state).
"""


class RNN(nn.Module):
    """Recurrent Neural Network.

    This network is used to store the hidden state of the policy. It currently supports GRU and LSTM.
    """

    def __init__(self, input_size: int, hidden_dim: int = 256, num_layers: int = 1, type: str = "lstm") -> None:
        """Initialize a GRU or LSTM module with internal hidden-state storage."""
        super().__init__()
        self.rnn = FunctionalLSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.hidden_state = None
        self.num_layers = num_layers

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Run recurrent inference in rollout mode or batched update mode."""
        batch_mode = masks is not None
        if batch_mode:
            # Batch mode needs saved hidden states
            if hidden_state is None:
                raise ValueError("Hidden states not passed to RNN module during policy update")
            out, _ = self.rnn(input, hidden_state)
            out = unpad_trajectories(out, masks)
        else:
            # Inference/distillation mode uses hidden state of last step
            out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
        return out

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset hidden states for all or done environments."""
        if dones is None:  # Reset hidden state
            if hidden_state is None:
                self.hidden_state = None
            else:
                self.hidden_state = hidden_state
        elif self.hidden_state is not None:  # Reset hidden state of done environments
            if hidden_state is None:
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = 0.0  # type: ignore
                else:
                    self.hidden_state[..., dones == 1, :] = 0.0
            else:
                raise NotImplementedError(
                    "Resetting the hidden state of done environments with a custom hidden state is not implemented"
                )

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden states for all or done environments from the computation graph."""
        if self.hidden_state is not None:
            if dones is None:  # Detach hidden state
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    self.hidden_state = tuple(hidden_state.detach() for hidden_state in self.hidden_state)
                else:
                    self.hidden_state = self.hidden_state.detach()
            else:  # Detach hidden state of done environments
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_state[..., dones == 1, :] = self.hidden_state[..., dones == 1, :].detach()
