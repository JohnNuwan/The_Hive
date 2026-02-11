"""
Monte Carlo Tree Search — THE HIVE MuZero Engine

Ported from Muzero_Pro_Trader/MuZero/agents/muzero_mcts.py.

Algorithm:
  1. Selection:  UCB score to pick best child node
  2. Expansion:  Dynamics network predicts next state + reward
  3. Backprop:   Value propagated up with discount factor
  4. Noise:      Dirichlet noise at root for exploration
"""

import math
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class Node:
    """A single node in the MCTS search tree."""

    __slots__ = ["visit_count", "prior", "value_sum", "children", "hidden_state", "reward"]

    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children = {}
        self.hidden_state = None
        self.reward = 0.0

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MinMaxStats:
    """Track min/max values for UCB normalization."""

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MuZeroMCTS:
    """
    Monte Carlo Tree Search for MuZero.

    Uses the Prediction network for initial expansion and the Dynamics
    network for subsequent expansions (imagining future states).
    """

    def __init__(self, config, network):
        self.config = config
        self.network = network

    def run(
        self,
        root_state: torch.Tensor,
        legal_actions: list = None,
        add_exploration_noise: bool = False,
    ) -> Node:
        """
        Run MCTS simulations from root_state.

        Returns the root Node with visit counts reflecting the search.
        """
        root = Node(0)
        root.hidden_state = root_state

        # Initial expansion using Prediction network
        policy, value = self.network.prediction(root_state)
        self._expand_node(root, policy, legal_actions)

        if add_exploration_noise:
            self._add_exploration_noise(root)

        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            # ── Selection ──
            node = root
            search_path = [node]
            actions_path = []

            while node.expanded:
                action, child = self._select_child(node, min_max_stats)
                search_path.append(child)
                actions_path.append(action)
                node = child

            # ── Expansion via Dynamics Network ──
            parent = search_path[-2]
            last_action = actions_path[-1]

            action_onehot = torch.zeros(
                (1, self.config.action_space_size), device=parent.hidden_state.device
            )
            action_onehot[0, last_action] = 1.0

            next_state, reward, policy, value = self.network.recurrent_inference(
                parent.hidden_state, action_onehot
            )

            node.hidden_state = next_state
            node.reward = reward.item()
            self._expand_node(node, policy, legal_actions)

            # ── Backpropagation ──
            self._backpropagate(search_path, value.item(), min_max_stats)

        return root

    def _select_child(self, node: Node, min_max_stats: MinMaxStats):
        """Select child with highest UCB score."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = self._ucb_score(node, child, min_max_stats)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        """Upper Confidence Bound score (AlphaZero/MuZero variant)."""
        pb_c = (
            math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base)
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = (
            min_max_stats.normalize(child.reward + self.config.discount * child.value)
            if child.visit_count > 0
            else 0
        )

        return prior_score + value_score

    def _expand_node(self, node: Node, policy_logits: torch.Tensor, legal_actions: list = None):
        """Create child nodes from policy distribution."""
        policy = policy_logits[0].detach().cpu().numpy()
        for action in range(self.config.action_space_size):
            if legal_actions and action not in legal_actions:
                continue
            node.children[action] = Node(policy[action])

    def _backpropagate(self, search_path: list, value: float, min_max_stats: MinMaxStats):
        """Propagate value estimates back up the tree."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value)
            value = node.reward + self.config.discount * value

    def _add_exploration_noise(self, node: Node):
        """Add Dirichlet noise at root for exploration diversity."""
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for i, action in enumerate(actions):
            node.children[action].prior = (
                node.children[action].prior * (1 - frac) + noise[i] * frac
            )
