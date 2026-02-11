"""
MuZero Neural Networks — THE HIVE

Ported from Muzero_Pro_Trader/MuZero/models/muzero_network.py.

Architecture:
  - RepresentationNetwork: Observation (142) → Hidden State (64)
  - DynamicsNetwork:       Hidden State + Action → Next State + Reward
  - PredictionNetwork:     Hidden State → Policy (5) + Value (1)
  - MuZeroNet:             Façade combining all three networks
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class RepresentationNetwork(nn.Module):
    """Encode raw observation → compact hidden state (latent space)."""

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Tanh())  # Normalize to [-1, 1] for dynamics stability
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DynamicsNetwork(nn.Module):
    """Predict next hidden state + reward from (state, action)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        in_dim = state_dim + action_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.common = nn.Sequential(*layers)

        self.next_state_head = nn.Sequential(
            nn.Linear(in_dim, state_dim),
            nn.Tanh(),
        )
        self.reward_head = nn.Linear(in_dim, 1)

    def forward(self, hidden_state: torch.Tensor, action_onehot: torch.Tensor):
        x = torch.cat([hidden_state, action_onehot], dim=1)
        common_out = self.common(x)
        next_state = self.next_state_head(common_out)
        reward = self.reward_head(common_out)
        return next_state, reward


class PredictionNetwork(nn.Module):
    """Predict policy distribution + value from hidden state."""

    def __init__(self, state_dim: int, hidden_dims: list, action_dim: int):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.common = nn.Sequential(*layers)

        self.policy_head = nn.Sequential(
            nn.Linear(in_dim, action_dim),
            nn.Softmax(dim=1),
        )
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, hidden_state: torch.Tensor):
        common_out = self.common(hidden_state)
        policy = self.policy_head(common_out)
        value = self.value_head(common_out)
        return policy, value


class MuZeroNet(nn.Module):
    """
    Complete MuZero network: Representation + Dynamics + Prediction.

    Two inference modes:
      - initial_inference(obs)         → hidden_state, policy, value
      - recurrent_inference(state, a)  → next_state, reward, policy, value
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        obs_dim = config.observation_shape[0]
        h_dim = config.hidden_state_size
        act_dim = config.action_space_size
        hidden_layers = config.network_hidden_dims

        self.representation = RepresentationNetwork(obs_dim, hidden_layers, h_dim)
        self.dynamics = DynamicsNetwork(h_dim, act_dim, hidden_layers)
        self.prediction = PredictionNetwork(h_dim, hidden_layers, act_dim)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"[MuZero:Net] Initialized — {total_params:,} parameters "
                    f"(obs={obs_dim}, hidden={h_dim}, actions={act_dim})")

    def initial_inference(self, observation: torch.Tensor):
        """First step: encode observation→latent, predict policy+value."""
        hidden_state = self.representation(observation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value

    def recurrent_inference(self, hidden_state: torch.Tensor, action_onehot: torch.Tensor):
        """Subsequent steps: simulate dynamics in imagination."""
        next_state, reward = self.dynamics(hidden_state, action_onehot)
        policy, value = self.prediction(next_state)
        return next_state, reward, policy, value
