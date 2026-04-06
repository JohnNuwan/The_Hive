"""Collecte MuZero parallele avec inference GPU centralisee et batchee.

Ce module garde un seul proprietaire GPU pour les reseaux JAX et
delegue la simulation des episodes a des workers CPU. Les workers
envoient leurs requetes d'inference via IPC afin d'eviter la
duplication des poids GPU.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from eva_lab.muzero.environment import TradingEnvironment
from eva_lab.muzero.jax_mcts import JAXMuZeroMCTS
from eva_lab.muzero.replay_buffer import GameHistory

logger = logging.getLogger(__name__)


@dataclass
class CollectorEnvironmentPayload:
    """Decrit un symbole pret a etre collecte par un worker CPU.

    Attributes:
        symbol (str): Symbole cible.
        market_data (np.ndarray): Historique enrichi deja materialise.
        max_steps (int): Nombre maximal de pas de l'episode.
        dataset_source (str | None): Source dataset observee.
    """

    symbol: str
    market_data: np.ndarray
    max_steps: int
    dataset_source: str | None = None


def build_collector_config(config: Any) -> dict[str, Any]:
    """Serialise la configuration MuZero utile aux workers.

    Args:
        config (Any): Configuration MuZero source.

    Returns:
        dict[str, Any]: Vue serialisable et minimale pour les workers.
    """

    return {
        "horizon": str(getattr(config, "horizon", "intraday") or "intraday"),
        "model_family": getattr(config, "model_family", None),
        "feature_profile": dict(getattr(config, "feature_profile", {}) or {}),
        "position_mechanics_profile": dict(
            getattr(config, "position_mechanics_profile", {}) or {}
        ),
        "mechanics_profile_version": str(
            getattr(config, "mechanics_profile_version", "v1") or "v1"
        ),
        "observation_shape": tuple(getattr(config, "observation_shape", (32,)) or (32,)),
        "action_space_size": int(getattr(config, "action_space_size", 5) or 5),
        "hidden_state_size": int(getattr(config, "hidden_state_size", 256) or 256),
        "max_moves": int(getattr(config, "max_moves", 300) or 300),
        "num_simulations": int(getattr(config, "num_simulations", 100) or 100),
        "discount": float(getattr(config, "discount", 0.99) or 0.99),
        "root_dirichlet_alpha": float(
            getattr(config, "root_dirichlet_alpha", 0.3) or 0.3
        ),
        "root_exploration_fraction": float(
            getattr(config, "root_exploration_fraction", 0.5) or 0.5
        ),
        "pb_c_base": int(getattr(config, "pb_c_base", 19652) or 19652),
        "pb_c_init": float(getattr(config, "pb_c_init", 1.25) or 1.25),
        "quality_trade_bonus": float(
            getattr(config, "quality_trade_bonus", 10.0) or 10.0
        ),
        "final_growth_bonus": float(
            getattr(config, "final_growth_bonus", 50.0) or 50.0
        ),
        "final_growth_threshold": float(
            getattr(config, "final_growth_threshold", 0.10) or 0.10
        ),
        "slbe_activation_bonus": float(
            getattr(config, "slbe_activation_bonus", 6.0) or 6.0
        ),
        "split_with_profit_bonus": float(
            getattr(config, "split_with_profit_bonus", 10.0) or 10.0
        ),
        "close_big_winner_bonus": float(
            getattr(config, "close_big_winner_bonus", 15.0) or 15.0
        ),
        "drawdown_time_penalty_rate": float(
            getattr(config, "drawdown_time_penalty_rate", 0.2) or 0.2
        ),
        "max_drawdown_penalty": float(
            getattr(config, "max_drawdown_penalty", 10.0) or 10.0
        ),
        "loss_penalty_multiplier": float(
            getattr(config, "loss_penalty_multiplier", 2.0) or 2.0
        ),
    }


class RemoteInferenceError(RuntimeError):
    """Represente une panne d'inference distante dans la collecte."""


class RemoteInferenceClient:
    """Client synchrone de requetes d'inference pour un worker CPU."""

    def __init__(
        self,
        *,
        worker_id: int,
        request_queue: mp.queues.Queue,
        response_queue: mp.queues.Queue,
    ) -> None:
        """Initialise les files IPC du worker.

        Args:
            worker_id (int): Identifiant stable du worker.
            request_queue (mp.queues.Queue): File des requetes vers le GPU.
            response_queue (mp.queues.Queue): File des reponses dediee.
        """

        self._worker_id = int(worker_id)
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._request_seq = 0

    def initial_inference(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute une inference initiale distante."""

        payload = {
            "observation": np.asarray(observation, dtype=np.float32).reshape(-1),
        }
        response = self._round_trip("initial", payload)
        return (
            np.asarray(response["hidden_state"], dtype=np.float32),
            np.asarray(response["logits"], dtype=np.float32),
            np.asarray(response["value"], dtype=np.float32),
        )

    def recurrent_inference(
        self,
        hidden_state: np.ndarray,
        action_onehot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute une inference recurrente distante."""

        payload = {
            "hidden_state": np.asarray(hidden_state, dtype=np.float32).reshape(-1),
            "action_onehot": np.asarray(action_onehot, dtype=np.float32).reshape(-1),
        }
        response = self._round_trip("recurrent", payload)
        return (
            np.asarray(response["next_state"], dtype=np.float32),
            np.asarray(response["reward"], dtype=np.float32),
            np.asarray(response["logits"], dtype=np.float32),
            np.asarray(response["value"], dtype=np.float32),
        )

    def _round_trip(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Envoie une requete puis attend sa reponse."""

        self._request_seq += 1
        request_id = f"{self._worker_id}:{self._request_seq}"
        self._request_queue.put(
            {
                "worker_id": self._worker_id,
                "request_id": request_id,
                "kind": kind,
                "payload": payload,
            }
        )
        response = self._response_queue.get()
        if str(response.get("request_id") or "") != request_id:
            raise RemoteInferenceError(
                "Reponse d'inference incoherente recue par le worker."
            )
        if not bool(response.get("ok", False)):
            raise RemoteInferenceError(
                str(response.get("error") or "Inference distante indisponible.")
            )
        return dict(response)


class BatchedInferenceCoordinator:
    """Agrege les requetes workers et execute les batchs GPU."""

    def __init__(
        self,
        *,
        request_queue: mp.queues.Queue,
        response_queues: dict[int, mp.queues.Queue],
        initial_inference_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        recurrent_inference_fn: Callable[
            [np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
        batch_max: int,
        batch_timeout_ms: int,
    ) -> None:
        """Initialise le coordinateur de micro-batching."""

        self._request_queue = request_queue
        self._response_queues = response_queues
        self._initial_inference_fn = initial_inference_fn
        self._recurrent_inference_fn = recurrent_inference_fn
        self._batch_max = max(int(batch_max or 1), 1)
        self._batch_timeout_ms = max(int(batch_timeout_ms or 1), 1)
        self._pending_requests: deque[dict[str, Any]] = deque()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats: dict[str, Any] = {
            "mode": "micro_batch",
            "batch_max": self._batch_max,
            "batch_timeout_ms": self._batch_timeout_ms,
            "total_requests": 0,
            "total_batches": 0,
            "initial_batches": 0,
            "recurrent_batches": 0,
            "max_observed_batch_size": 0,
            "average_batch_size": 0.0,
            "average_batch_latency_ms": 0.0,
            "average_initial_batch_size": 0.0,
            "average_recurrent_batch_size": 0.0,
        }

    def start(self) -> None:
        """Demarre la boucle du coordinateur dans un thread dedie."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._serve,
            name="muzero-batched-inference",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Arrete le coordinateur et attend la fin du thread."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def snapshot(self) -> dict[str, Any]:
        """Retourne un resume stable de l'activite d'inference."""

        return dict(self._stats)

    def _serve(self) -> None:
        """Boucle principale d'agregation des requetes workers."""

        while True:
            first_request = self._pop_next_request()
            if first_request is None:
                if self._stop_event.is_set():
                    break
                continue
            batch = self._gather_batch(first_request)
            try:
                self._process_batch(batch)
            except Exception as exc:  # pragma: no cover - securite runtime
                logger.exception("Coordinateur MuZero en erreur: %s", exc)
                error_message = (
                    "Inference GPU batchee en erreur: "
                    f"{type(exc).__name__}: {exc}"
                )
                for request in batch:
                    self._response_queues[int(request["worker_id"])].put(
                        {
                            "request_id": request["request_id"],
                            "ok": False,
                            "error": error_message,
                        }
                    )

    def _pop_next_request(self) -> dict[str, Any] | None:
        """Retourne la prochaine requete disponible."""

        if self._pending_requests:
            return self._pending_requests.popleft()
        try:
            return self._request_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def _gather_batch(self, first_request: dict[str, Any]) -> list[dict[str, Any]]:
        """Agrege un micro-batch homogene."""

        batch = [first_request]
        batch_kind = str(first_request.get("kind") or "")
        deadline = time.perf_counter() + (self._batch_timeout_ms / 1000.0)

        while len(batch) < self._batch_max:
            timeout = max(deadline - time.perf_counter(), 0.0)
            if timeout <= 0.0:
                break
            try:
                request = self._request_queue.get(timeout=timeout)
            except queue.Empty:
                break
            if str(request.get("kind") or "") != batch_kind:
                self._pending_requests.append(request)
                continue
            batch.append(request)
        return batch

    def _process_batch(self, batch: list[dict[str, Any]]) -> None:
        """Execute un micro-batch d'inference sur le GPU."""

        if not batch:
            return
        batch_size = len(batch)
        batch_kind = str(batch[0].get("kind") or "")
        started_at = time.perf_counter()
        if batch_kind == "initial":
            observations = np.stack(
                [
                    np.asarray(item["payload"]["observation"], dtype=np.float32).reshape(-1)
                    for item in batch
                ],
                axis=0,
            )
            hidden_state, logits, value = self._initial_inference_fn(observations)
            hidden_state = _coerce_batched_output(
                hidden_state,
                batch_size=batch_size,
                tensor_name="hidden_state",
            )
            logits = _coerce_batched_output(
                logits,
                batch_size=batch_size,
                tensor_name="logits",
            )
            value = _coerce_batched_output(
                value,
                batch_size=batch_size,
                tensor_name="value",
            )
            outputs = [
                {
                    "hidden_state": hidden_state[index],
                    "logits": logits[index],
                    "value": value[index],
                }
                for index in range(len(batch))
            ]
        else:
            hidden_states = np.stack(
                [
                    np.asarray(item["payload"]["hidden_state"], dtype=np.float32).reshape(-1)
                    for item in batch
                ],
                axis=0,
            )
            action_onehots = np.stack(
                [
                    np.asarray(item["payload"]["action_onehot"], dtype=np.float32).reshape(-1)
                    for item in batch
                ],
                axis=0,
            )
            next_state, reward, logits, value = self._recurrent_inference_fn(
                hidden_states,
                action_onehots,
            )
            next_state = _coerce_batched_output(
                next_state,
                batch_size=batch_size,
                tensor_name="next_state",
            )
            reward = _coerce_batched_output(
                reward,
                batch_size=batch_size,
                tensor_name="reward",
            )
            logits = _coerce_batched_output(
                logits,
                batch_size=batch_size,
                tensor_name="logits",
            )
            value = _coerce_batched_output(
                value,
                batch_size=batch_size,
                tensor_name="value",
            )
            outputs = [
                {
                    "next_state": next_state[index],
                    "reward": reward[index],
                    "logits": logits[index],
                    "value": value[index],
                }
                for index in range(len(batch))
            ]

        duration_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
        self._record_batch_metrics(batch_kind, len(batch), duration_ms)

        for request, output in zip(batch, outputs):
            payload = {
                "request_id": request["request_id"],
                "ok": True,
                **output,
            }
            self._response_queues[int(request["worker_id"])].put(payload)

    def _record_batch_metrics(self, batch_kind: str, batch_size: int, duration_ms: float) -> None:
        """Met a jour les statistiques agregees du coordinateur."""

        previous_batches = int(self._stats["total_batches"])
        previous_requests = int(self._stats["total_requests"])
        self._stats["total_batches"] = previous_batches + 1
        self._stats["total_requests"] = previous_requests + batch_size
        self._stats["max_observed_batch_size"] = max(
            int(self._stats["max_observed_batch_size"]),
            int(batch_size),
        )
        self._stats["average_batch_size"] = round(
            float(self._stats["total_requests"]) / float(self._stats["total_batches"]),
            3,
        )
        previous_latency = float(self._stats["average_batch_latency_ms"])
        total_batches = float(self._stats["total_batches"])
        self._stats["average_batch_latency_ms"] = round(
            ((previous_latency * (total_batches - 1.0)) + duration_ms) / total_batches,
            3,
        )
        if batch_kind == "initial":
            previous_kind_batches = int(self._stats["initial_batches"])
            self._stats["initial_batches"] = previous_kind_batches + 1
            self._stats["average_initial_batch_size"] = round(
                (
                    float(self._stats["average_initial_batch_size"]) * float(previous_kind_batches)
                    + float(batch_size)
                )
                / float(self._stats["initial_batches"]),
                3,
            )
        else:
            previous_kind_batches = int(self._stats["recurrent_batches"])
            self._stats["recurrent_batches"] = previous_kind_batches + 1
            self._stats["average_recurrent_batch_size"] = round(
                (
                    float(self._stats["average_recurrent_batch_size"]) * float(previous_kind_batches)
                    + float(batch_size)
                )
                / float(self._stats["recurrent_batches"]),
                3,
            )


def _coerce_batched_output(
    value: Any,
    *,
    batch_size: int,
    tensor_name: str,
) -> np.ndarray:
    """Normalise une sortie d'inference en tenseur explicitement batche.

    Args:
        value (Any): Sortie brute materialisee cote Python.
        batch_size (int): Taille de batch attendue.
        tensor_name (str): Nom logique du tenseur pour les erreurs.

    Returns:
        np.ndarray: Tableau avec un axe batch explicite en position 0.

    Raises:
        ValueError: Si la sortie ne peut pas etre alignee sur le batch.
    """

    array_value = np.asarray(value, dtype=np.float32)
    expected_batch = max(int(batch_size), 1)
    if array_value.ndim == 0:
        return array_value.reshape(1, 1)
    if array_value.shape[0] == expected_batch:
        return array_value
    if array_value.ndim == 1 and expected_batch == 1:
        return array_value.reshape(1, -1)
    if array_value.ndim == 1 and array_value.size == expected_batch:
        return array_value.reshape(expected_batch, 1)
    raise ValueError(
        "Sortie d'inference incoherente pour "
        f"{tensor_name}: shape={array_value.shape}, batch_attendu={expected_batch}"
    )


def _policy_from_logits(logits: np.ndarray, action_space_size: int) -> np.ndarray:
    """Convertit des logits en distribution stable."""

    logits_np = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits_np.size != action_space_size:
        logits_np = np.resize(logits_np, action_space_size)
    logits_np = logits_np - np.max(logits_np)
    probs = np.exp(logits_np).astype(np.float32)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(
            action_space_size,
            1.0 / max(action_space_size, 1),
            dtype=np.float32,
        )
    return (probs / total).astype(np.float32)


def _select_action_from_policy(policy: np.ndarray, exploration: bool) -> int:
    """Choisit une action discrete depuis une politique."""

    if exploration:
        return int(np.random.choice(np.arange(len(policy)), p=policy))
    return int(np.argmax(policy))


def _select_action_from_root(root: Any, exploration: bool) -> int:
    """Choisit une action a partir des visites MCTS."""

    visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
    actions = [item[0] for item in visit_counts]
    counts = np.asarray([item[1] for item in visit_counts], dtype=np.float32)
    if exploration:
        total = float(np.sum(counts))
        if not np.isfinite(total) or total <= 0.0:
            return int(actions[0])
        probs = counts / total
        return int(np.random.choice(actions, p=probs))
    return int(actions[int(np.argmax(counts))])


def _get_policy_distribution(root: Any, action_space_size: int) -> np.ndarray:
    """Construit une politique a partir des visites MCTS."""

    policy = np.zeros(action_space_size, dtype=np.float32)
    for action, child in root.children.items():
        policy[action] = float(child.visit_count)
    total = float(np.sum(policy))
    if total > 0.0:
        policy = policy / total
    return policy.astype(np.float32)


def _json_safe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Materialise un resume d'episode en types JSON simples."""

    safe_summary: dict[str, Any] = {}
    for key, value in dict(summary or {}).items():
        if isinstance(value, np.generic):
            safe_summary[key] = value.item()
        elif isinstance(value, np.ndarray):
            safe_summary[key] = value.tolist()
        else:
            safe_summary[key] = value
    return safe_summary


def _play_game_remote(
    *,
    env_payload: CollectorEnvironmentPayload,
    config_payload: dict[str, Any],
    client: RemoteInferenceClient,
    collection_mode: str,
    exploration: bool,
    max_wall_time_seconds: float | None,
) -> tuple[GameHistory, dict[str, Any]]:
    """Joue un episode complet en delocalisant l'inference GPU."""

    config = SimpleNamespace(**dict(config_payload or {}))
    env = TradingEnvironment(
        data=np.asarray(env_payload.market_data, dtype=np.float32),
        symbol=env_payload.symbol,
        config=config,
        max_steps=int(env_payload.max_steps),
    )
    setattr(env, "dataset_source", env_payload.dataset_source or "unknown")

    game = GameHistory()
    game.metadata.update(
        {
            "symbol": env_payload.symbol,
            "dataset_source": env_payload.dataset_source or "unknown",
            "collection_mode": collection_mode,
        }
    )
    observation, _ = env.reset()
    done = False
    steps = 0
    started_at = time.perf_counter()

    while not done and steps < int(config.max_moves):
        if (
            max_wall_time_seconds is not None
            and max_wall_time_seconds > 0.0
            and (time.perf_counter() - started_at) >= max_wall_time_seconds
        ):
            logger.warning(
                "Collecte distante interrompue apres %.1fs sur %s.",
                max_wall_time_seconds,
                env_payload.symbol,
            )
            break

        steps += 1
        hidden_state, logits, value_tensor = client.initial_inference(observation)

        if collection_mode == "mcts":
            mcts = JAXMuZeroMCTS(
                config,
                recurrent_inference_fn=client.recurrent_inference,
            )
            root = mcts.run(
                hidden_state,
                add_exploration_noise=exploration,
                root_logits=logits,
                root_value=value_tensor,
            )
            action = _select_action_from_root(root, exploration)
            policy = _get_policy_distribution(root, int(config.action_space_size))
            value = float(root.value)
        elif collection_mode == "policy_only":
            policy = _policy_from_logits(logits, int(config.action_space_size))
            action = _select_action_from_policy(policy, exploration)
            value = float(np.asarray(value_tensor, dtype=np.float32).reshape(-1)[0])
        else:
            raise ValueError(f"Mode de collecte MuZero inconnu: {collection_mode}")

        next_observation, reward, done, _, _ = env.step(action)
        game.store(
            np.asarray(observation, dtype=np.float32).reshape(-1),
            int(action),
            float(reward),
            np.asarray(policy, dtype=np.float32),
            float(value),
        )
        observation = next_observation

    return game, _json_safe_summary(env.get_summary())


def _collection_worker(
    *,
    worker_id: int,
    job_queue: mp.queues.Queue,
    request_queue: mp.queues.Queue,
    response_queue: mp.queues.Queue,
    result_queue: mp.queues.Queue,
    config_payload: dict[str, Any],
    games_per_symbol: int,
    collection_mode: str,
    max_wall_time_seconds: float | None,
) -> None:
    """Execute la collecte CPU pour un ou plusieurs symboles."""

    client = RemoteInferenceClient(
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
    )
    while True:
        env_payload = job_queue.get()
        if env_payload is None:
            break
        result_queue.put(
            {
                "type": "symbol_start",
                "worker_id": worker_id,
                "symbol": env_payload.symbol,
            }
        )
        try:
            for game_index in range(1, games_per_symbol + 1):
                game, summary = _play_game_remote(
                    env_payload=env_payload,
                    config_payload=config_payload,
                    client=client,
                    collection_mode=collection_mode,
                    exploration=True,
                    max_wall_time_seconds=max_wall_time_seconds,
                )
                result_queue.put(
                    {
                        "type": "game_result",
                        "worker_id": worker_id,
                        "symbol": env_payload.symbol,
                        "game_index": game_index,
                        "summary": summary,
                        "game": game,
                    }
                )
            result_queue.put(
                {
                    "type": "symbol_done",
                    "worker_id": worker_id,
                    "symbol": env_payload.symbol,
                }
            )
        except Exception as exc:  # pragma: no cover - securite runtime
            result_queue.put(
                {
                    "type": "worker_error",
                    "worker_id": worker_id,
                    "symbol": env_payload.symbol,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exception(
                        type(exc),
                        exc,
                        exc.__traceback__,
                    )[-6:],
                }
            )
            return
    result_queue.put({"type": "worker_done", "worker_id": worker_id})


def collect_games_parallel(
    *,
    agent: Any,
    config: Any,
    environments: list[CollectorEnvironmentPayload],
    games_per_symbol: int,
    collection_mode: str,
    max_wall_time_seconds: float | None,
    collector_workers: int,
    queue_depth: int,
    inference_batch_max: int,
    inference_batch_timeout_ms: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Collecte des episodes MuZero en parallele par symbole."""

    if not environments:
        return {
            "collector_mode": "batched_symbol_workers",
            "collector_workers": 0,
            "collector_queue_depth": int(queue_depth),
            "collector_active_symbols": [],
            "inference_batch_profile": {
                "mode": "micro_batch",
                "batch_max": int(inference_batch_max),
                "batch_timeout_ms": int(inference_batch_timeout_ms),
                "total_requests": 0,
                "total_batches": 0,
            },
            "valid_symbols": [],
            "total_games": 0,
            "symbol_game_counts": {},
        }

    worker_count = min(max(int(collector_workers or 1), 1), len(environments))
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue(maxsize=max(int(queue_depth or 8), 8))
    result_queue = ctx.Queue()
    job_queue = ctx.Queue()
    response_queues = {
        worker_id: ctx.Queue(maxsize=max(int(queue_depth or 8), 8))
        for worker_id in range(worker_count)
    }

    for env_payload in environments:
        job_queue.put(env_payload)
    for _ in range(worker_count):
        job_queue.put(None)

    coordinator = BatchedInferenceCoordinator(
        request_queue=request_queue,
        response_queues=response_queues,
        initial_inference_fn=agent.initial_inference_batch,
        recurrent_inference_fn=agent.recurrent_inference_batch,
        batch_max=inference_batch_max,
        batch_timeout_ms=inference_batch_timeout_ms,
    )
    coordinator.start()

    config_payload = build_collector_config(config)
    processes: list[mp.Process] = []
    symbol_order = {payload.symbol: index + 1 for index, payload in enumerate(environments)}
    symbol_game_counts = {payload.symbol: 0 for payload in environments}
    pending_symbols = [payload.symbol for payload in environments]
    total_games = 0
    completed_workers = 0
    errors: list[dict[str, Any]] = []
    result_poll_timeout_seconds = 5.0
    heartbeat_interval_seconds = 30.0
    last_heartbeat_at = time.perf_counter()

    try:
        for worker_id in range(worker_count):
            process = ctx.Process(
                target=_collection_worker,
                kwargs={
                    "worker_id": worker_id,
                    "job_queue": job_queue,
                    "request_queue": request_queue,
                    "response_queue": response_queues[worker_id],
                    "result_queue": result_queue,
                    "config_payload": config_payload,
                    "games_per_symbol": int(games_per_symbol),
                    "collection_mode": collection_mode,
                    "max_wall_time_seconds": max_wall_time_seconds,
                },
                name=f"muzero-collector-{worker_id}",
            )
            process.start()
            processes.append(process)

        while completed_workers < worker_count:
            try:
                message = result_queue.get(timeout=result_poll_timeout_seconds)
            except queue.Empty:
                now = time.perf_counter()
                alive_workers = sum(1 for process in processes if process.is_alive())
                live_snapshot = coordinator.snapshot()
                if progress_callback is not None and (
                    (now - last_heartbeat_at) >= heartbeat_interval_seconds
                ):
                    progress_callback(
                        {
                            "event": "collector_heartbeat",
                            "collector_active_symbols": list(pending_symbols),
                            "collector_alive_workers": int(alive_workers),
                            "collector_completed_workers": int(completed_workers),
                            "symbol_game_counts": dict(symbol_game_counts),
                            "total_games": int(total_games),
                            "inference_batch_profile": dict(live_snapshot),
                        }
                    )
                    last_heartbeat_at = now
                if alive_workers <= 0:
                    raise RuntimeError(
                        "Collecte MuZero parallele interrompue: aucun worker actif "
                        "sans message de progression."
                    )
                continue
            message_type = str(message.get("type") or "")
            symbol = str(message.get("symbol") or "").strip()
            live_snapshot = coordinator.snapshot()

            if message_type == "symbol_start":
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "symbol_start",
                            "symbol": symbol,
                            "collector_active_symbols": list(pending_symbols),
                            "inference_batch_profile": dict(live_snapshot),
                        }
                    )
                continue

            if message_type == "game_result":
                game = message.get("game")
                if isinstance(game, GameHistory) and len(game) > 0:
                    agent.replay_buffer.save_game(game)
                total_games += 1
                symbol_game_counts[symbol] = int(symbol_game_counts.get(symbol, 0)) + 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "game_result",
                            "symbol": symbol,
                            "symbol_index": symbol_order.get(symbol, 0),
                            "symbol_total": len(environments),
                            "part_index": int(message.get("game_index") or 0),
                            "part_total": int(games_per_symbol),
                            "replay_entries": int(agent.replay_buffer.size),
                            "collector_active_symbols": list(pending_symbols),
                            "inference_batch_profile": dict(live_snapshot),
                        }
                    )
                if log_callback is not None:
                    summary = dict(message.get("summary") or {})
                    log_callback(
                        "[%s] %s partie %s/%s | return=%.2f%% | trades=%s | buffer=%s"
                        % (
                            getattr(config, "horizon", "n/a"),
                            symbol,
                            int(message.get("game_index") or 0),
                            int(games_per_symbol),
                            float(summary.get("return_pct", 0.0) or 0.0),
                            int(summary.get("total_trades", 0) or 0),
                            int(agent.replay_buffer.size),
                        )
                    )
                continue

            if message_type == "symbol_done":
                pending_symbols = [item for item in pending_symbols if item != symbol]
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "symbol_done",
                            "symbol": symbol,
                            "collector_active_symbols": list(pending_symbols),
                            "inference_batch_profile": dict(live_snapshot),
                        }
                    )
                continue

            if message_type == "worker_error":
                errors.append(dict(message))
                break

            if message_type == "worker_done":
                completed_workers += 1

        if errors:
            first_error = errors[0]
            traceback_tail = [
                str(line).strip()
                for line in list(first_error.get("traceback_tail") or [])
                if str(line).strip()
            ]
            diagnostic_suffix = ""
            if traceback_tail:
                diagnostic_suffix = " | traceback=" + " || ".join(traceback_tail)
            raise RuntimeError(
                "Collecte MuZero parallele en echec sur "
                f"{first_error.get('symbol') or 'symbole_inconnu'}: "
                f"{first_error.get('error_type')}: {first_error.get('error')}"
                f"{diagnostic_suffix}"
            )
    finally:
        for process in processes:
            if process.is_alive() and errors:
                process.terminate()
        for process in processes:
            process.join(timeout=5.0)
        coordinator.stop()

    return {
        "collector_mode": "batched_symbol_workers",
        "collector_workers": worker_count,
        "collector_queue_depth": int(queue_depth),
        "collector_active_symbols": list(pending_symbols),
        "inference_batch_profile": coordinator.snapshot(),
        "valid_symbols": [payload.symbol for payload in environments],
        "total_games": int(total_games),
        "symbol_game_counts": dict(symbol_game_counts),
    }
