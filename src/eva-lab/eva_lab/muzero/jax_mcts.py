"""MCTS MuZero pour les reseaux JAX, avec expansion et retropropagation."""

from __future__ import annotations

import logging
import math

import jax
import jax.numpy as jnp
import numpy as np

from eva_lab.muzero.jax_networks import support_to_scalar

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
        root_policy_logits: jnp.ndarray,
        root_value_logits: jnp.ndarray,
        root_legal_actions: list[int] | None = None,
        add_exploration_noise: bool = False,
    ) -> Node:
        """Lance la recherche MCTS depuis un etat latent racine.

        Args:
            root_state (jnp.ndarray): Etat latent initial.
            root_policy_logits (jnp.ndarray): Logits de politique de
                l'inference initiale.
            root_value_logits (jnp.ndarray): Logits de valeur de
                l'inference initiale.
            root_legal_actions (list[int] | None): Masque des actions legales
                pour la racine uniquement.
            add_exploration_noise (bool): Active le bruit de Dirichlet.

        Returns:
            Node: Racine de l'arbre MCTS.
        """
        root = Node(0.0)
        root.hidden_state = root_state

        policy = jax.nn.softmax(root_policy_logits, axis=-1)
        value = support_to_scalar(root_value_logits, self.config.support_size)

        self._expand_node(root, policy, root_legal_actions)
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

            next_state, reward_logits, logits, value_logits = self.rec_apply(
                self.params,
                parent.hidden_state,
                action_onehot,
            )

            reward = support_to_scalar(reward_logits, self.config.support_size)
            value = support_to_scalar(value_logits, self.config.support_size)

            node.hidden_state = next_state
            node.reward = float(reward[0, 0])
            policy = jax.nn.softmax(logits, axis=-1)
            self._expand_node(node, policy)
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

    def _expand_node(
        self,
        node: Node,
        policy: jnp.ndarray,
        legal_actions: list[int] | None = None,
    ) -> None:
        """Cree les enfants d'un noeud a partir de la politique predite.

        Args:
            node (Node): Noeud a developper.
            policy (jnp.ndarray): Probabilites de politique du noeud.
            legal_actions (list[int] | None): Actions autorisees pour ce noeud.
        """
        normalized_policy, allowed_actions = self._normalize_policy(policy, legal_actions)
        for action in allowed_actions:
            node.children[action] = Node(float(normalized_policy[action]))

    def _normalize_policy(
        self,
        policy: jnp.ndarray,
        legal_actions: list[int] | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """Normalise une politique et applique un masque legal si fourni.

        Args:
            policy (jnp.ndarray): Politique a normaliser.
            legal_actions (list[int] | None): Actions autorisees.

        Returns:
            tuple[np.ndarray, list[int]]: Politique normalisee et actions
                retenues.
        """
        policy_np = np.asarray(policy[0], dtype=np.float32)
        if legal_actions is None:
            total = float(policy_np.sum())
            if total <= 0.0:
                policy_np = np.full(
                    self.config.action_space_size,
                    1.0 / float(self.config.action_space_size),
                    dtype=np.float32,
                )
            else:
                policy_np = policy_np / total
            return policy_np, list(range(self.config.action_space_size))

        allowed_actions: list[int] = []
        for action in legal_actions:
            action_int = int(action)
            if 0 <= action_int < self.config.action_space_size and action_int not in allowed_actions:
                allowed_actions.append(action_int)
        if not allowed_actions:
            allowed_actions = [0]

        masked_policy = np.zeros_like(policy_np, dtype=np.float32)
        masked_policy[allowed_actions] = policy_np[allowed_actions]
        total = float(masked_policy.sum())
        if total <= 0.0:
            masked_policy[allowed_actions] = 1.0 / float(len(allowed_actions))
        else:
            masked_policy /= total
        return masked_policy, allowed_actions

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
        if not actions:
            return
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        fraction = self.config.root_exploration_fraction
        for index, action in enumerate(actions):
            child = node.children[action]
            child.prior = child.prior * (1 - fraction) + noise[index] * fraction
