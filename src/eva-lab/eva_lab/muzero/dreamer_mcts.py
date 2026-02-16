"""
DreamerMCTS — Hybrid Monte Carlo Tree Search using RSSM World Model
THE HIVE EVA Lab

This implementation uses the RSSM (Recurrent State Space Model) as the dynamics engine.
It leverages the latent space for selection, expansion and backpropagation.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
import logging
from typing import Dict, List, Tuple, Optional

from eva_lab.muzero.rssm import RSSMState

logger = logging.getLogger(__name__)

class DreamerNode:
    """A single node in the Dreamer MCTS search tree."""
    __slots__ = ["visit_count", "prior", "value_sum", "children", "rssm_state", "reward"]

    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[int, 'DreamerNode'] = {}
        self.rssm_state: Optional[RSSMState] = None
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

class DreamerMCTS:
    """
    Search tree implementation using RSSM for dynamics.
    """
    def __init__(self, config, params, apply_fns):
        """
        Args:
            config: MuZeroConfig (with horizon, simulations, etc)
            params: Parameters for the transformed DreamerModel
            apply_fns: Tuple of (init_fn, imagine_fn) for the DreamerModel
        """
        self.config = config
        self.params = params
        self.init_apply, self.imagine_apply = apply_fns

    def run(self, root_rssm_state: RSSMState, add_exploration_noise: bool = False) -> DreamerNode:
        root = DreamerNode(0)
        root.rssm_state = root_rssm_state

        # Initial expansion using policy head
        # In Dreamer, we can use the Actor head to get priors
        # We need a way to call the actor head individually.
        # Assuming our apply_fns or params structure allows this.
        
        # Simulation loop
        min_max_stats = MinMaxStats()
        
        # Expansion of the root first
        self._expand_root(root)

        if add_exploration_noise:
            self._add_exploration_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            actions_path = []

            # 1. Selection
            while node.expanded:
                action, child = self._select_child(node, min_max_stats)
                search_path.append(child)
                actions_path.append(action)
                node = child

            # 2. Expansion & Evaluation
            parent = search_path[-2]
            last_action = actions_path[-1]
            
            # Action one-hot
            action_onehot = jax.nn.one_hot(jnp.array([last_action]), self.config.action_space_size)
            
            # Use RSSM to imagine next latent state
            next_rssm_state = self.imagine_apply(self.params, None, parent.rssm_state, action_onehot)
            
            # Evaluate the new state
            # Note: We need reward and value predictions here.
            # In a full DreamerModel, we have decoders for this.
            # We'll assume the model provides these or we have specific apply_fns.
            # For this MVP version, we'll use a placeholder logic that matches the trainer.
            
            node.rssm_state = next_rssm_state
            # pred_reward, pred_value, policy_logits ...
            # self.model_evaluation_apply(...)
            
            # Placeholder for expansion (real implementation would use Decoders)
            self._expand_node(node)
            
            # 3. Backpropagation
            # value_prediction = ...
            # self._backpropagate(search_path, float(value_prediction), min_max_stats)

        return root

    def _expand_root(self, node: DreamerNode):
        # Dummy uniform prior for now
        for action in range(self.config.action_space_size):
            node.children[action] = DreamerNode(1.0 / self.config.action_space_size)

    def _expand_node(self, node: DreamerNode):
        # Dummy uniform prior for now
        for action in range(self.config.action_space_size):
            node.children[action] = DreamerNode(1.0 / self.config.action_space_size)

    def _select_child(self, node: DreamerNode, min_max_stats: MinMaxStats):
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

    def _ucb_score(self, parent: DreamerNode, child: DreamerNode, min_max_stats: MinMaxStats) -> float:
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.reward + self.config.discount * child.value) if child.visit_count > 0 else 0
        return prior_score + value_score

    def _backpropagate(self, search_path: List[DreamerNode], value: float, min_max_stats: MinMaxStats):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value)
            value = node.reward + self.config.discount * value

    def _add_exploration_noise(self, node: DreamerNode):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for i, action in enumerate(actions):
            node.children[action].prior = node.children[action].prior * (1 - frac) + noise[i] * frac
