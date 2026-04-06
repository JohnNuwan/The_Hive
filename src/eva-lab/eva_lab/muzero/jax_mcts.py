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

    def __init__(self, config, params=None, apply_fns=None, recurrent_inference_fn=None):
        """Memorise la configuration et les hooks d'inference.

        Args:
            config (Any): Configuration MuZero.
            params (Any | None): Poids JAX pour le mode local.
            apply_fns (tuple | None): Hooks ``(init, recurrent)`` jittes pour
                le mode local.
            recurrent_inference_fn (Callable | None): Callback recurrente
                distante pour le mode collecteur.
        """
        self.config = config
        self.params = params
        self.init_apply = None
        self.rec_apply = None
        if apply_fns is not None:
            self.init_apply, self.rec_apply = apply_fns
        self.recurrent_inference_fn = recurrent_inference_fn

    def run(
        self,
        root_state: jnp.ndarray,
        legal_actions: list | None = None,
        add_exploration_noise: bool = False,
        root_logits: jnp.ndarray | np.ndarray | None = None,
        root_value: jnp.ndarray | np.ndarray | None = None,
    ) -> Node:
        """Lance la recherche MCTS depuis un etat latent racine."""
        root = Node(0.0)
        root.hidden_state = root_state

        if root_logits is None or root_value is None:
            if self.rec_apply is None or self.params is None:
                raise ValueError(
                    "MCTS MuZero requiert des logits de racine ou une inference locale valide."
                )
            dummy_action = jnp.zeros((1, self.config.action_space_size))
            _, _, logits, value = self.rec_apply(self.params, root_state, dummy_action)
        else:
            logits = jnp.asarray(root_logits)
            value = jnp.asarray(root_value)
        policy = jax.nn.softmax(logits)
        self._expand_node(root, policy, legal_actions)
        root.value_sum = self._scalar_from_tensor(value)
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

            if self.recurrent_inference_fn is not None:
                next_state, reward, logits, value = self.recurrent_inference_fn(
                    parent.hidden_state,
                    action_onehot,
                )
            elif self.rec_apply is not None and self.params is not None:
                next_state, reward, logits, value = self.rec_apply(
                    self.params,
                    parent.hidden_state,
                    action_onehot,
                )
            else:
                raise ValueError("Aucun chemin d'inference recurrente MuZero n'est disponible.")

            node.hidden_state = next_state
            node.reward = self._scalar_from_tensor(reward)
            policy = jax.nn.softmax(jnp.asarray(logits))
            self._expand_node(node, policy, legal_actions)
            self._backpropagate(search_path, self._scalar_from_tensor(value), min_max_stats)

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
        """Cree les enfants d'un noeud a partir de la politique predite.

        Args:
            node (Node): Noeud a etendre.
            policy (jnp.ndarray): Politique predite, acceptee en forme
                ``[actions]`` ou ``[batch, actions]``.
            legal_actions (list | None): Sous-ensemble d'actions autorisees.
        """

        policy_np = np.asarray(policy, dtype=np.float32)
        if policy_np.ndim == 0:
            policy_np = np.full(
                int(self.config.action_space_size),
                1.0 / max(int(self.config.action_space_size), 1),
                dtype=np.float32,
            )
        elif policy_np.ndim >= 2:
            policy_np = np.asarray(policy_np[0], dtype=np.float32)
        else:
            policy_np = policy_np.reshape(-1)
        if policy_np.size != int(self.config.action_space_size):
            policy_np = np.resize(policy_np, int(self.config.action_space_size))
        for action in range(self.config.action_space_size):
            if legal_actions and action not in legal_actions:
                continue
            node.children[action] = Node(float(policy_np[action]))

    @staticmethod
    def _scalar_from_tensor(value: jnp.ndarray | np.ndarray) -> float:
        """Materialise un scalaire JAX/Numpy en float Python."""

        value_np = np.asarray(jax.device_get(value), dtype=np.float32).reshape(-1)
        if value_np.size <= 0:
            return 0.0
        return float(value_np[0])

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
