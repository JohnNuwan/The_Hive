"""
Circuit Breaker — Pattern de résilience inter-services
═══════════════════════════════════════════════════════

Empêche les défaillances en cascade dans THE HIVE.

États :
  CLOSED    → Normal, les requêtes passent
  OPEN      → Service défaillant, requêtes rejetées immédiatement
  HALF_OPEN → Test de récupération, N requêtes autorisées

Transitions :
  CLOSED  --[failures >= threshold]--> OPEN
  OPEN    --[timeout écoulé]---------> HALF_OPEN
  HALF_OPEN --[succès]--------------> CLOSED
  HALF_OPEN --[échec]---------------> OPEN
"""

import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """État du circuit breaker"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    """Levée quand le Circuit Breaker est ouvert (service indisponible)."""
    pass


class CircuitBreaker:
    """
    Implémentation du pattern Circuit Breaker pour THE HIVE.

    Usage comme décorateur:
        cb = CircuitBreaker("banker_mt5")

        @cb
        async def call_mt5_service():
            ...

    Usage programmatique:
        cb = CircuitBreaker("redis")
        try:
            result = await cb.execute(some_async_func, *args)
        except CircuitBreakerOpenError:
            # Fallback
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_requests: int = 2,
    ):
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        self.failures = 0
        self.successes_in_half_open = 0
        self.half_open_requests = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()
        self.total_calls = 0
        self.total_failures = 0
        self.total_rejected = 0

        logger.info(
            f"⚡ Circuit Breaker '{name}' initialisé "
            f"(seuil={failure_threshold}, recovery={recovery_timeout}s)"
        )

    def _transition(self, new_state: CircuitState) -> None:
        """Transition d'état avec logging"""
        old = self.state
        self.state = new_state
        self.last_state_change = time.time()

        if new_state == CircuitState.OPEN:
            logger.error(f"🔴 CB '{self.name}': {old} → OPEN (service défaillant)")
        elif new_state == CircuitState.HALF_OPEN:
            logger.warning(f"🟡 CB '{self.name}': {old} → HALF_OPEN (test récupération)")
            self.half_open_requests = 0
            self.successes_in_half_open = 0
        elif new_state == CircuitState.CLOSED:
            logger.info(f"🟢 CB '{self.name}': {old} → CLOSED (service rétabli)")
            self.failures = 0

    def _check_state(self) -> None:
        """Vérifie les transitions automatiques (OPEN → HALF_OPEN)"""
        if self.state == CircuitState.OPEN and self.last_failure_time:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)

    def _record_success(self) -> None:
        """Enregistre un succès"""
        if self.state == CircuitState.HALF_OPEN:
            self.successes_in_half_open += 1
            if self.successes_in_half_open >= self.half_open_max_requests:
                self._transition(CircuitState.CLOSED)
        elif self.state == CircuitState.CLOSED:
            # Reset progressif des failures sur succès
            self.failures = max(0, self.failures - 1)

    def _record_failure(self) -> None:
        """Enregistre un échec"""
        self.failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.OPEN)
        elif self.state == CircuitState.CLOSED:
            if self.failures >= self.failure_threshold:
                self._transition(CircuitState.OPEN)

    async def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Exécute une fonction à travers le circuit breaker"""
        self._check_state()
        self.total_calls += 1

        if self.state == CircuitState.OPEN:
            self.total_rejected += 1
            raise CircuitBreakerOpenError(
                f"Circuit Breaker '{self.name}' est OPEN. "
                f"Retry dans {self.recovery_timeout}s."
            )

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_requests += 1
            if self.half_open_requests > self.half_open_max_requests:
                self.total_rejected += 1
                raise CircuitBreakerOpenError(
                    f"Circuit Breaker '{self.name}' HALF_OPEN — quota de test atteint."
                )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def __call__(self, func: Callable) -> Callable:
        """Utilisable comme décorateur"""
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.execute(func, *args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état complet du circuit breaker"""
        self._check_state()
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.failures,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": (
                datetime.fromtimestamp(self.last_failure_time).isoformat()
                if self.last_failure_time
                else None
            ),
            "time_in_state_seconds": round(time.time() - self.last_state_change, 1),
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejected": self.total_rejected,
        }
