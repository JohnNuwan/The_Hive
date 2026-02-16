"""
Monte Carlo Tree Search for JAX-based MuZero — THE HIVE EVA Lab

This implementation interacts with JAX-transformed functions.
Logic:
  1. Selection:  UCB score to pick best child
  2. Expansion:  Recurrent inference via Dynamics network
  3. Backprop:   Value propagated up
"""

import math
import numpy as np
import jax.numpy as jnp
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

class JAXMuZeroMCTS:
    """
    MCTS for MuZero using JAX inference functions.
    """
    def __init__(self, config, params, apply_fns):
        """
        Args:
            config: MuZeroConfig
            params: JAX network parameters
            apply_fns: Tuple of (initial_inference, recurrent_inference) transformed functions
        """
        self.config = config
        self.params = params
        self.init_apply, self.rec_apply = apply_fns

    def run(self, root_state: jnp.ndarray, legal_actions: list = None, add_exploration_noise: bool = False) -> Node:
        root = Node(0)
        root.hidden_state = root_state

        # Initial expansion (Prediction)
        # We assume initial_inference(params, obs) -> (h, logits, v)
        # But here we already have root_state (hidden). We need policy/value for it.
        # Actually in MuZero, we usually start with initial_inference(obs) -> s0, p0, v0
        # For simplicity, let's assume root_state already comes with p/v or we compute them.
        
        # We need to call prediction part. In jax_networks, recurrent_inference includes prediction.
        # Let's use recurrent_inference with zero action if needed, or better, 
        # modify jax_networks to expose prediction separately if possible.
        # For now, let's call rec_apply with a dummy action or modify to handle root.
        
        # Re-check jax_networks initial_inference code... 
        # It takes observation. If we have root_state, we are already inside recurrent loops or it's the start.
        # If it's the start, root_state is s0. 
        
        # Hack for MVP: call rec_apply with action 0 but ignore the next_h/r for root expansion.
        dummy_action = jnp.zeros((1, self.config.action_space_size))
        _, _, logits, value = self.rec_apply(self.params, root_state, dummy_action)
        
        policy = jax.nn.softmax(logits)
        self._expand_node(root, policy, legal_actions)

        if add_exploration_noise:
            self._add_exploration_noise(root)

        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            actions_path = []

            while node.expanded:
                action, child = self._select_child(node, min_max_stats)
                search_path.append(child)
                actions_path.append(action)
                node = child

            # Expansion via Dynamics
            parent = search_path[-2]
            last_action = actions_path[-1]

            action_onehot = jnp.zeros((1, self.config.action_space_size))
            action_onehot = action_onehot.at[0, last_action].set(1.0)

            next_state, reward, logits, value = self.rec_apply(self.params, parent.hidden_state, action_onehot)
            
            node.hidden_state = next_state
            node.reward = float(reward[0, 0])
            policy = jax.nn.softmax(logits)
            self._expand_node(node, policy, legal_actions)

            self._backpropagate(search_path, float(value[0, 0]), min_max_stats)

        return root

    def _select_child(self, node: Node, min_max_stats: MinMaxStats):
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
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.reward + self.config.discount * child.value) if child.visit_count > 0 else 0
        return prior_score + value_score

    def _expand_node(self, node: Node, policy: jnp.ndarray, legal_actions: list = None):
        policy_np = np.array(policy[0])
        for action in range(self.config.action_space_size):
            if legal_actions and action not in legal_actions:
                continue
            node.children[action] = Node(float(policy_np[action]))

    def _backpropagate(self, search_path: list, value: float, min_max_stats: MinMaxStats):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value)
            value = node.reward + self.config.discount * value

    def _add_exploration_noise(self, node: Node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for i, action in enumerate(actions):
            node.children[action].prior = node.children[action].prior * (1 - frac) + noise[i] * frac
