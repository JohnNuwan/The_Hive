"""MCTS MuZero pour les reseaux JAX, avec expansion et retropropagation."""

from __future__ import annotations

import logging
import math

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class Node:
    """Represente un noeud de l'arbre MCTS."""

    __slots__ = ["visit_count", "prior", "value_sum", "children", "hidden_state", "reward"]

    def __init__(self, prior: float):
        """Initialise les compteurs et la prior d'un noeud."""
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children = {}
        self.hidden_state = None
        self.reward = 0.0

    @property
    def expanded(self) -> bool:
        """Indique si le noeud a deja ete developpe."""
        return len(self.children) > 0

    @property
    def value(self) -> float:
        """Retourne la valeur moyenne du noeud."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MinMaxStats:
    """Normalise les valeurs utilisees par le score UCB."""

    def __init__(self):
        """Initialise les bornes min/max."""
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float) -> None:
        """Met a jour les bornes avec une nouvelle valeur."""
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """Normalise une valeur si les bornes sont connues."""
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class JAXMuZeroMCTS:
    """Execute la recherche d'arbre MuZero a partir des fonctions JAX jittees."""

    def __init__(self, config, params, apply_fns):
        """Memorise la configuration, les poids et les hooks d'inference."""
        self.config = config
        self.params = params
        self.init_apply, self.rec_apply = apply_fns

    def run(
        self,
        root_state: jnp.ndarray,
        legal_actions: list | None = None,
        add_exploration_noise: bool = False,
    ) -> Node:
        """Lance la recherche MCTS depuis un etat latent racine."""
        root = Node(0.0)
        root.hidden_state = root_state

        dummy_action = jnp.zeros((1, self.config.action_space_size))
        _, _, logits, value = self.rec_apply(self.params, root_state, dummy_action)
        policy = jax.nn.softmax(logits)
        self._expand_node(root, policy, legal_actions)
        root.value_sum = float(value[0, 0])
        root.visit_count = 1

        if add_exploration_noise:
            self._add_exploration_noise(root)

        min_max_stats = MinMaxStats()
        min_max_stats.update(root.value)

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            actions_path = []

            while node.expanded:
                action, child = self._select_child(node, min_max_stats)
                search_path.append(child)
                actions_path.append(action)
                node = child

            parent = search_path[-2]
            last_action = actions_path[-1]
            action_onehot = jnp.zeros((1, self.config.action_space_size))
            action_onehot = action_onehot.at[0, last_action].set(1.0)

            next_state, reward, logits, value = self.rec_apply(
                self.params,
                parent.hidden_state,
                action_onehot,
            )

            node.hidden_state = next_state
            node.reward = float(reward[0, 0])
            policy = jax.nn.softmax(logits)
            self._expand_node(node, policy, legal_actions)
            self._backpropagate(search_path, float(value[0, 0]), min_max_stats)

        return root

    def _select_child(self, node: Node, min_max_stats: MinMaxStats):
        """Selectionne l'enfant avec le meilleur score UCB."""
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
        """Calcule le score UCB d'un enfant."""
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base)
        pb_c += self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(
                child.reward + self.config.discount * child.value,
            )
        else:
            value_score = 0.0
        return prior_score + value_score

    def _expand_node(self, node: Node, policy: jnp.ndarray, legal_actions: list | None = None) -> None:
        """Cree les enfants d'un noeud a partir de la politique predite."""
        policy_np = np.array(policy[0])
        for action in range(self.config.action_space_size):
            if legal_actions and action not in legal_actions:
                continue
            node.children[action] = Node(float(policy_np[action]))

    def _backpropagate(self, search_path: list[Node], value: float, min_max_stats: MinMaxStats) -> None:
        """Retropropage la valeur le long du chemin visite."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value)
            value = node.reward + self.config.discount * value

    def _add_exploration_noise(self, node: Node) -> None:
        """Ajoute le bruit de Dirichlet a la racine pour le self-play."""
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        fraction = self.config.root_exploration_fraction
        for index, action in enumerate(actions):
            child = node.children[action]
            child.prior = child.prior * (1 - fraction) + noise[index] * fraction
