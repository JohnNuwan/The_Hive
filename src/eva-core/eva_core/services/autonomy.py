"""Service d'autonomie lecture seule pour EVA Core."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable

import httpx

from shared.config import Settings
from shared.internal_auth import get_internal_headers
from shared.redis_client import get_redis_client

from eva_core.services.docker_monitor import SystemMonitor

logger = logging.getLogger(__name__)


class AutonomyService:
    """Agrège l'etat de la ruche pour l'autonomie en lecture seule.

    Cette couche ne declenche aucune action destructive. Elle consolide
    simplement les signaux utiles a EVA pour comprendre l'etat courant de la
    ruche: agents, trading, lab, monitoring et dependances memoire.
    """

    def __init__(
        self,
        settings: Settings,
        system_monitor: SystemMonitor,
        refresh_interval_seconds: int = 60,
    ) -> None:
        """Initialise le service d'autonomie.

        Args:
            settings (Settings): Configuration globale du projet.
            system_monitor (SystemMonitor): Moniteur systeme local au conteneur.
            refresh_interval_seconds (int): Periode de rafraichissement auto.
        """
        self.settings = settings
        self.system_monitor = system_monitor
        self.refresh_interval_seconds = max(15, refresh_interval_seconds)
        self._snapshot: dict[str, Any] = {
            "generated_at": None,
            "mode": "read_only_autonomy",
            "posture": {
                "status": "initializing",
                "recommended_mode": "research_only",
                "blockers": ["snapshot_not_ready"],
            },
        }

    @staticmethod
    def _build_http_url(host: str, port: int, path: str) -> str:
        """Construit une URL HTTP simple.

        Args:
            host (str): Hote cible.
            port (int): Port cible.
            path (str): Chemin HTTP.

        Returns:
            str: URL HTTP complete.
        """
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"http://{host}:{port}{normalized_path}"

    @staticmethod
    def _normalize_http_probe(
        name: str,
        ok: bool,
        status_code: int | None = None,
        payload: Any = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Normalise la sortie d'une sonde HTTP.

        Args:
            name (str): Nom logique de la sonde.
            ok (bool): Etat de succes.
            status_code (int | None): Code HTTP si disponible.
            payload (Any): Charge utile JSON ou texte.
            error (str | None): Message d'erreur si echec.

        Returns:
            dict[str, Any]: Resultat standardise.
        """
        return {
            "name": name,
            "ok": ok,
            "status": "online" if ok else "offline",
            "status_code": status_code,
            "payload": payload,
            "error": error,
        }

    @staticmethod
    def _extract_json_payload(response: httpx.Response) -> Any:
        """Retourne un contenu JSON si possible, sinon le texte brut.

        Args:
            response (httpx.Response): Reponse HTTP recue.

        Returns:
            Any: Charge utile JSON ou texte.
        """
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        text = response.text.strip()
        return text[:500] if text else None

    async def _probe_http(self, name: str, url: str, timeout: float = 4.0) -> dict[str, Any]:
        """Interroge un endpoint HTTP en lecture seule.

        Args:
            name (str): Nom logique de la sonde.
            url (str): URL cible.
            timeout (float): Timeout en secondes.

        Returns:
            dict[str, Any]: Resultat normalise.
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=get_internal_headers("core"))
            payload = self._extract_json_payload(response)
            return self._normalize_http_probe(
                name=name,
                ok=response.status_code == 200,
                status_code=response.status_code,
                payload=payload,
                error=None if response.status_code == 200 else f"http_{response.status_code}",
            )
        except Exception as exc:
            return self._normalize_http_probe(name=name, ok=False, error=str(exc))

    async def _probe_vllm(self) -> dict[str, Any]:
        """Interroge le serveur vLLM.

        Returns:
            dict[str, Any]: Etat de l'API d'inference.
        """
        url = self._build_http_url(self.settings.vllm_host, self.settings.vllm_port, "/v1/models")
        return await self._probe_http("vllm", url, timeout=6.0)

    async def _probe_tcp(self, name: str, host: str, port: int, timeout: float = 2.0) -> dict[str, Any]:
        """Teste une dependance TCP sans action destructive.

        Args:
            name (str): Nom logique de la sonde.
            host (str): Hote cible.
            port (int): Port cible.
            timeout (float): Timeout en secondes.

        Returns:
            dict[str, Any]: Resultat normalise.
        """
        try:
            connect_coro = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(connect_coro, timeout=timeout)
            writer.close()
            await writer.wait_closed()
            return {
                "name": name,
                "ok": True,
                "status": "online",
                "host": host,
                "port": port,
            }
        except Exception as exc:
            return {
                "name": name,
                "ok": False,
                "status": "offline",
                "host": host,
                "port": port,
                "error": str(exc),
            }

    def _service_endpoints(self) -> dict[str, str]:
        """Construit la table des endpoints critiques exposes a EVA.

        Returns:
            dict[str, str]: Mapping service -> URL de lecture seule.
        """
        return {
            "banker_health": self._build_http_url(
                self.settings.banker_api_host,
                self.settings.banker_api_port,
                "/health",
            ),
            "banker_trading": self._build_http_url(
                self.settings.banker_api_host,
                self.settings.banker_api_port,
                "/trading/status",
            ),
            "banker_performance": self._build_http_url(
                self.settings.banker_api_host,
                self.settings.banker_api_port,
                "/performance/models?days=7&limit=5",
            ),
            "sentinel_health": self._build_http_url(
                self.settings.sentinel_api_host,
                self.settings.sentinel_api_port,
                "/health",
            ),
            "sentinel_metrics": self._build_http_url(
                self.settings.sentinel_api_host,
                self.settings.sentinel_api_port,
                "/system/metrics",
            ),
            "lab_health": self._build_http_url(
                self.settings.lab_api_host,
                self.settings.lab_api_port,
                "/health",
            ),
            "lab_champions": self._build_http_url(
                self.settings.lab_api_host,
                self.settings.lab_api_port,
                "/champions/status",
            ),
            "lab_universe": self._build_http_url(
                self.settings.lab_api_host,
                self.settings.lab_api_port,
                "/live/universe?horizon=intraday",
            ),
        }

    @staticmethod
    def _docker_summary(containers: list[dict[str, Any]]) -> dict[str, Any]:
        """Agrege une liste de conteneurs Docker.

        Args:
            containers (list[dict[str, Any]]): Conteneurs remontes par le monitor.

        Returns:
            dict[str, Any]: Resume exploitable pour EVA.
        """
        total = len(containers)
        running = sum(1 for container in containers if str(container.get("status", "")).lower() in {"running", "healthy"})
        unhealthy = sum(1 for container in containers if str(container.get("status", "")).lower() in {"unhealthy", "exited", "dead"})
        top_unhealthy = [
            {
                "name": container.get("name"),
                "status": container.get("status"),
                "cpu": container.get("cpu_percent"),
                "memory_mb": container.get("memory_mb"),
            }
            for container in containers
            if str(container.get("status", "")).lower() not in {"running", "healthy"}
        ][:5]
        return {
            "total": total,
            "running": running,
            "unhealthy": unhealthy,
            "top_unhealthy": top_unhealthy,
        }

    @staticmethod
    def _resolve_champion_state(champions_probe: dict[str, Any]) -> dict[str, Any]:
        """Extrait l'etat champion intraday pour le pilotage live.

        Args:
            champions_probe (dict[str, Any]): Sonde HTTP du service Lab.

        Returns:
            dict[str, Any]: Synthese de l'etat champion.
        """
        payload = champions_probe.get("payload") if champions_probe.get("ok") else {}
        if not isinstance(payload, dict):
            return {
                "selection": "unknown",
                "engine": "unknown",
                "gate": {"allowed": False, "reason": "lab_unavailable"},
            }

        horizon_payload = ((payload.get("horizons") or {}).get("intraday") or {})
        promotion_gate = horizon_payload.get("promotion_gate") or {}
        return {
            "selection": horizon_payload.get("selection", "unknown"),
            "engine": horizon_payload.get("engine_label") or payload.get("dreamer_gate", {}).get("engine", "unknown"),
            "gate": promotion_gate,
            "candidate_id": horizon_payload.get("candidate_id"),
            "champion_id": horizon_payload.get("champion_id"),
            "live_checkpoint": horizon_payload.get("live_checkpoint"),
        }

    @staticmethod
    def _build_posture(
        agents: dict[str, Any],
        banker_trading: dict[str, Any],
        champion_state: dict[str, Any],
        monitoring_ready: bool,
        dependencies: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Derive la posture operative d'EVA.

        Args:
            agents (dict[str, Any]): Etat des agents.
            banker_trading (dict[str, Any]): Etat de trading du banker.
            champion_state (dict[str, Any]): Etat de promotion intraday.
            monitoring_ready (bool): Etat global du monitoring.
            dependencies (dict[str, dict[str, Any]]): Etat des dependances memoire/infra.

        Returns:
            dict[str, Any]: Posture d'autonomie et blocages.
        """
        blockers: list[str] = []
        banker_connection = (banker_trading.get("connection") or {}) if isinstance(banker_trading, dict) else {}
        banker_risk = (banker_trading.get("risk") or {}) if isinstance(banker_trading, dict) else {}
        mt5_connected = bool(banker_connection.get("mt5_connected", False))
        trading_allowed = bool(banker_risk.get("trading_allowed", False))
        champion_allowed = bool((champion_state.get("gate") or {}).get("allowed", False))

        if agents.get("banker", {}).get("status") != "online":
            blockers.append("banker_offline")
        if not mt5_connected:
            blockers.append("mt5_offline")
        if not trading_allowed:
            blockers.append("risk_gate_closed")
        if not champion_allowed:
            blockers.append(str((champion_state.get("gate") or {}).get("reason", "champion_blocked")))
        if not monitoring_ready:
            blockers.append("monitoring_degraded")
        for dependency_name, dependency_state in dependencies.items():
            if not dependency_state.get("ok", False):
                blockers.append(f"{dependency_name}_offline")

        can_trade = mt5_connected and trading_allowed and champion_allowed
        if can_trade:
            recommended_mode = "assisted_live"
            status = "ready"
        elif champion_allowed:
            recommended_mode = "safe_hold"
            status = "degraded"
        else:
            recommended_mode = "research_only"
            status = "blocked"

        return {
            "status": status,
            "can_trade": can_trade,
            "monitoring_ready": monitoring_ready,
            "memory_ready": all(item.get("ok", False) for item in dependencies.values()),
            "recommended_mode": recommended_mode,
            "blockers": blockers,
        }

    async def collect_snapshot(self, agents_snapshot: dict[str, Any]) -> dict[str, Any]:
        """Construit un snapshot complet d'autonomie.

        Args:
            agents_snapshot (dict[str, Any]): Etat Redis des agents de la ruche.

        Returns:
            dict[str, Any]: Snapshot agrege pret a etre expose ou publie.
        """
        endpoints = self._service_endpoints()
        probes = await asyncio.gather(
            self._probe_http("banker_health", endpoints["banker_health"]),
            self._probe_http("banker_trading", endpoints["banker_trading"]),
            self._probe_http("banker_performance", endpoints["banker_performance"]),
            self._probe_http("sentinel_health", endpoints["sentinel_health"]),
            self._probe_http("sentinel_metrics", endpoints["sentinel_metrics"]),
            self._probe_http("lab_health", endpoints["lab_health"]),
            self._probe_http("lab_champions", endpoints["lab_champions"], timeout=8.0),
            self._probe_http("lab_universe", endpoints["lab_universe"]),
            self._probe_vllm(),
            self._probe_tcp("redis", self.settings.redis_host, self.settings.redis_port),
            self._probe_tcp("qdrant", self.settings.qdrant_host, self.settings.qdrant_port),
            self._probe_tcp("neo4j", self.settings.neo4j_host, self.settings.neo4j_port),
        )
        probe_map = {probe["name"]: probe for probe in probes}

        system_metrics = await self.system_monitor.get_system_metrics()
        docker_containers = await self.system_monitor.get_docker_containers()
        champion_state = self._resolve_champion_state(probe_map["lab_champions"])
        banker_trading_payload = probe_map["banker_trading"].get("payload") or {}
        sentinel_metrics_payload = probe_map["sentinel_metrics"].get("payload") or {}
        lab_universe_payload = probe_map["lab_universe"].get("payload") or {}

        dependencies = {
            "redis": probe_map["redis"],
            "qdrant": probe_map["qdrant"],
            "neo4j": probe_map["neo4j"],
            "vllm": probe_map["vllm"],
        }
        monitoring_ready = bool(probe_map["sentinel_health"].get("ok") or system_metrics.get("real_data", False))
        posture = self._build_posture(
            agents=agents_snapshot,
            banker_trading=banker_trading_payload,
            champion_state=champion_state,
            monitoring_ready=monitoring_ready,
            dependencies=dependencies,
        )

        return {
            "generated_at": datetime.now().isoformat(),
            "mode": "read_only_autonomy",
            "posture": posture,
            "agents": agents_snapshot,
            "services": {
                "banker": {
                    "health": probe_map["banker_health"],
                    "trading": banker_trading_payload,
                    "performance": probe_map["banker_performance"].get("payload"),
                },
                "sentinel": {
                    "health": probe_map["sentinel_health"],
                    "metrics": sentinel_metrics_payload,
                },
                "lab": {
                    "health": probe_map["lab_health"],
                    "champion_state": champion_state,
                    "live_universe": lab_universe_payload,
                },
            },
            "monitoring": {
                "system_metrics": system_metrics,
                "docker": self._docker_summary(docker_containers),
            },
            "dependencies": dependencies,
        }

    async def refresh_snapshot(self, agents_snapshot: dict[str, Any]) -> dict[str, Any]:
        """Regenere et publie un snapshot sur Redis.

        Args:
            agents_snapshot (dict[str, Any]): Etat Redis des agents.

        Returns:
            dict[str, Any]: Snapshot regenere.
        """
        snapshot = await self.collect_snapshot(agents_snapshot)
        self._snapshot = snapshot
        try:
            redis_client = get_redis_client()
            await redis_client.cache_set(
                "eva.core.autonomy.snapshot",
                snapshot,
                ttl_seconds=max(120, self.refresh_interval_seconds * 2),
            )
        except Exception as exc:
            logger.warning("Publication Redis du snapshot d'autonomie impossible: %s", exc)
        return snapshot

    def get_snapshot(self) -> dict[str, Any]:
        """Retourne le dernier snapshot connu.

        Returns:
            dict[str, Any]: Snapshot d'autonomie mis en cache.
        """
        return self._snapshot

    async def start_monitoring(
        self,
        agents_provider: Callable[[], Awaitable[dict[str, Any]]],
    ) -> None:
        """Demarre la boucle de publication periodique.

        Args:
            agents_provider (Callable[[], Awaitable[dict[str, Any]]]): Fonction
                asynchrone qui retourne l'etat courant des agents.
        """
        logger.info(
            "Moniteur d'autonomie EVA active (intervalle: %ss).",
            self.refresh_interval_seconds,
        )
        while True:
            try:
                agents_snapshot = await agents_provider()
                await self.refresh_snapshot(agents_snapshot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Cycle d'autonomie EVA interrompu: %s", exc)
            await asyncio.sleep(self.refresh_interval_seconds)
