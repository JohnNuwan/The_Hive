"""
Moteur Hydra de replication master/slaves.

Ce service transforme un fill confirme du compte maitre en jobs de copie
vers des executeurs MT5 distants. Il reste agnostique du terminal reel :
chaque slave expose une petite API HTTP dediee, souvent lancee sous Wine
avec son propre `WINEPREFIX`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from typing import Any
from uuid import UUID

import httpx

from eva_banker.services.multi_account import MultiAccountManager
from shared import (
    CopyTradeRequest,
    CopyTradeResult,
    HydraEventType,
    HydraJobStatus,
    HydraReplicationJob,
    HydraScalingMode,
    HydraTerminalHealth,
    OrderSource,
    PropFirmAccount,
    TradeOrder,
    TradeReplicationEvent,
)
from shared.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class HydraCopyEngine:
    """
    Orchestre la replication Hydra a partir des fills du maitre.

    Le moteur:
    - filtre les comptes esclaves eligibles ;
    - applique mapping symbole et scaling ;
    - envoie la requete a l'executeur distant du compte cible ;
    - historise jobs, sante et mesures de latence pour Nexus.
    """

    REDIS_JOBS_PREFIX = "hydra:job:"
    REDIS_HEALTH_PREFIX = "hydra:health:"
    REDIS_METRICS_PREFIX = "hydra:metrics:"

    def __init__(
        self,
        account_manager: MultiAccountManager,
        *,
        master_source_id: str,
        request_timeout_seconds: float = 5.0,
        order_path: str = "/hydra/terminal/order",
        close_path: str = "/hydra/terminal/close",
        health_path: str = "/hydra/terminal/health",
    ) -> None:
        self.account_manager = account_manager
        self.master_source_id = str(master_source_id or "local-master").strip() or "local-master"
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.order_path = str(order_path or "/hydra/terminal/order")
        self.close_path = str(close_path or "/hydra/terminal/close")
        self.health_path = str(health_path or "/hydra/terminal/health")
        self._jobs: dict[UUID, HydraReplicationJob] = {}
        self._job_index: dict[tuple[str, UUID], UUID] = {}
        self._metrics: dict[str, dict[str, Any]] = {}
        self._health: dict[UUID, HydraTerminalHealth] = {}

    async def initialize(self) -> None:
        """
        Initialise le moteur Hydra a partir du registre comptes.
        """
        await self.account_manager.initialize()

    async def _persist_job(self, job: HydraReplicationJob) -> None:
        """
        Persiste un job Hydra en Redis si possible.

        Args:
            job (HydraReplicationJob): Job a sauvegarder.
        """
        self._jobs[job.id] = job
        self._job_index[(str(job.event_id), job.target_account_id)] = job.id
        try:
            redis = get_redis_client()
            await redis.cache_set(
                f"{self.REDIS_JOBS_PREFIX}{job.id}",
                job.model_dump(mode="json"),
                ttl_seconds=86400,
            )
        except Exception as exc:
            logger.warning("Hydra: echec de persistance du job %s (%s).", job.id, exc)

    async def _find_existing_job(
        self,
        *,
        event_id: UUID,
        target_account_id: UUID,
    ) -> HydraReplicationJob | None:
        """
        Recherche un job deja cree pour un couple evenement/cible.

        Args:
            event_id (UUID): Identifiant de l'evenement maitre.
            target_account_id (UUID): Compte esclave cible.

        Returns:
            HydraReplicationJob | None: Job existant si deja connu.
        """
        indexed_id = self._job_index.get((str(event_id), target_account_id))
        if indexed_id is not None:
            return self._jobs.get(indexed_id)

        for job in self._jobs.values():
            if job.event_id == event_id and job.target_account_id == target_account_id:
                self._job_index[(str(event_id), target_account_id)] = job.id
                return job
        return None

    async def _persist_health(self, health: HydraTerminalHealth) -> None:
        """
        Persiste l'etat de sante d'un terminal esclave.

        Args:
            health (HydraTerminalHealth): Snapshot a memoriser.
        """
        self._health[health.account_id] = health
        try:
            redis = get_redis_client()
            await redis.cache_set(
                f"{self.REDIS_HEALTH_PREFIX}{health.account_id}",
                health.model_dump(mode="json"),
                ttl_seconds=3600,
            )
        except Exception as exc:
            logger.warning("Hydra: echec de persistance de la sante %s (%s).", health.account_id, exc)

    async def _persist_metrics(self, account: PropFirmAccount, patch: dict[str, Any]) -> None:
        """
        Met a jour les metriques compactes d'un compte esclave.

        Args:
            account (PropFirmAccount): Compte cible.
            patch (dict[str, Any]): Delta de metriques.
        """
        key = str(account.id)
        current = dict(self._metrics.get(key) or {})
        current.update(patch)
        current.setdefault("account_id", key)
        current.setdefault("account_name", account.name)
        current.setdefault("account_login", account.login)
        current["updated_at"] = datetime.now().isoformat()
        self._metrics[key] = current
        try:
            redis = get_redis_client()
            await redis.cache_set(
                f"{self.REDIS_METRICS_PREFIX}{account.id}",
                current,
                ttl_seconds=86400,
            )
        except Exception as exc:
            logger.warning("Hydra: echec de persistance des metriques %s (%s).", account.name, exc)

    @staticmethod
    def _normalize_symbol(account: PropFirmAccount, symbol: str) -> str | None:
        """
        Applique le mapping symbole du compte cible.

        Args:
            account (PropFirmAccount): Compte esclave cible.
            symbol (str): Symbole maitre.

        Returns:
            str | None: Symbole final ou `None` si le symbole est interdit.
        """
        normalized_symbol = str(symbol or "").strip()
        if not normalized_symbol:
            return None
        mapped_symbol = account.symbol_map.get(normalized_symbol, normalized_symbol)
        if account.allowed_symbols and mapped_symbol not in set(account.allowed_symbols):
            return None
        return mapped_symbol

    @staticmethod
    def _quantize_volume(volume: Decimal, step: Decimal, minimum: Decimal, maximum: Decimal) -> Decimal:
        """
        Contraint un volume selon les bornes du compte cible.

        Args:
            volume (Decimal): Volume souhaite.
            step (Decimal): Pas de lot.
            minimum (Decimal): Lot minimal.
            maximum (Decimal): Lot maximal.

        Returns:
            Decimal: Volume exploitable sur le compte cible.
        """
        safe_step = step if step > 0 else Decimal("0.01")
        safe_min = minimum if minimum > 0 else safe_step
        safe_max = maximum if maximum >= safe_min else safe_min
        bounded = min(max(volume, safe_min), safe_max)
        steps = ((bounded - safe_min) / safe_step).to_integral_value(rounding=ROUND_FLOOR)
        normalized = safe_min + (steps * safe_step)
        precision = max(0, -safe_step.normalize().as_tuple().exponent)
        return normalized.quantize(Decimal("1").scaleb(-precision))

    def _compute_slave_volume(
        self,
        account: PropFirmAccount,
        event: TradeReplicationEvent,
    ) -> Decimal:
        """
        Calcule le volume final d'un slave apres scaling.

        Args:
            account (PropFirmAccount): Compte esclave.
            event (TradeReplicationEvent): Fill maitre.

        Returns:
            Decimal: Volume final pret a envoyer.
        """
        source_volume = Decimal(str(event.volume or 0))
        factor = Decimal(str(account.scaling_factor or 1))
        if account.scaling_mode == HydraScalingMode.PROPORTIONAL:
            master_balance = Decimal(str(event.master_balance or 0))
            if master_balance > 0:
                ratio = Decimal(str(account.current_balance or 0)) / master_balance
            else:
                ratio = Decimal("1")
            requested = source_volume * factor * ratio
        else:
            requested = source_volume * factor
        return self._quantize_volume(
            requested,
            Decimal(str(account.lot_step or "0.01")),
            Decimal(str(account.lot_min or "0.01")),
            Decimal(str(account.lot_max or "10.0")),
        )

    def _build_job(self, account: PropFirmAccount, event: TradeReplicationEvent, symbol: str, volume: Decimal) -> HydraReplicationJob:
        """
        Construit un job Hydra en etat `pending`.

        Args:
            account (PropFirmAccount): Compte cible.
            event (TradeReplicationEvent): Evenement maitre.
            symbol (str): Symbole mappe.
            volume (Decimal): Volume final.

        Returns:
            HydraReplicationJob: Job initialise.
        """
        return HydraReplicationJob(
            event_id=event.event_id,
            source_account_id=event.source_account_id,
            target_account_id=account.id,
            target_login=account.login,
            event_type=event.event_type,
            symbol=symbol,
            action=event.action,
            volume=volume,
            source_ticket=event.ticket,
            scaling_mode=account.scaling_mode,
            scaling_factor=Decimal(str(account.scaling_factor or 1)),
        )

    async def _dispatch_order_job(
        self,
        account: PropFirmAccount,
        job: HydraReplicationJob,
        event: TradeReplicationEvent,
        *,
        dry_run: bool,
    ) -> HydraReplicationJob:
        """
        Envoie un job d'ouverture vers un executeur Hydra distant.

        Args:
            account (PropFirmAccount): Compte cible.
            job (HydraReplicationJob): Job a mettre a jour.
            event (TradeReplicationEvent): Fill maitre source.
            dry_run (bool): Si vrai, ne contacte pas l'executeur.

        Returns:
            HydraReplicationJob: Job finalise.
        """
        if dry_run:
            return job.model_copy(
                update={
                    "status": HydraJobStatus.DISPATCHED,
                    "updated_at": datetime.now(),
                }
            )

        if not account.executor_url:
            return job.model_copy(
                update={
                    "status": HydraJobStatus.REJECTED,
                    "error_message": "Aucun executeur Hydra configure pour ce compte.",
                    "updated_at": datetime.now(),
                }
            )

        order = TradeOrder(
            symbol=job.symbol,
            action=job.action,
            volume=job.volume,
            entry_price=event.entry_price,
            stop_loss_price=event.stop_loss_price,
            take_profit_price=event.take_profit_price,
            source=OrderSource.COPY,
            comment=f"Hydra copy {event.ticket}",
        )
        payload = order.model_dump(mode="json")
        payload["master_ticket"] = event.ticket

        started_at = datetime.now()
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                response = await client.post(f"{account.executor_url}{self.order_path}", json=payload)
                response.raise_for_status()
                data = response.json()
            latency_ms = int((datetime.now() - started_at).total_seconds() * 1000)
            if not bool(data.get("success")):
                return job.model_copy(
                    update={
                        "status": HydraJobStatus.FAILED,
                        "latency_ms": latency_ms,
                        "error_message": str(data.get("message") or "Execution distante refusee."),
                        "updated_at": datetime.now(),
                    }
                )
            return job.model_copy(
                update={
                    "status": HydraJobStatus.EXECUTED,
                    "target_ticket": data.get("ticket"),
                    "latency_ms": latency_ms,
                    "updated_at": datetime.now(),
                }
            )
        except Exception as exc:
            return job.model_copy(
                update={
                    "status": HydraJobStatus.FAILED,
                    "error_message": f"Echec executeur distant: {exc}",
                    "updated_at": datetime.now(),
                }
            )

    async def _dispatch_close_job(
        self,
        account: PropFirmAccount,
        job: HydraReplicationJob,
        *,
        dry_run: bool,
    ) -> HydraReplicationJob:
        """
        Envoie un job de cloture vers un executeur Hydra distant.

        Args:
            account (PropFirmAccount): Compte cible.
            job (HydraReplicationJob): Job a mettre a jour.
            dry_run (bool): Si vrai, ne contacte pas l'executeur.

        Returns:
            HydraReplicationJob: Job finalise.
        """
        if dry_run:
            return job.model_copy(
                update={
                    "status": HydraJobStatus.DISPATCHED,
                    "updated_at": datetime.now(),
                }
            )

        if not account.executor_url:
            return job.model_copy(
                update={
                    "status": HydraJobStatus.REJECTED,
                    "error_message": "Aucun executeur Hydra configure pour ce compte.",
                    "updated_at": datetime.now(),
                }
            )

        payload = {
            "source_ticket": job.source_ticket,
            "symbol": job.symbol,
            "action": job.action.value,
        }
        started_at = datetime.now()
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                response = await client.post(f"{account.executor_url}{self.close_path}", json=payload)
                response.raise_for_status()
                data = response.json()
            latency_ms = int((datetime.now() - started_at).total_seconds() * 1000)
            if not bool(data.get("success")):
                return job.model_copy(
                    update={
                        "status": HydraJobStatus.FAILED,
                        "latency_ms": latency_ms,
                        "error_message": str(data.get("message") or "Cloture distante refusee."),
                        "updated_at": datetime.now(),
                    }
                )
            return job.model_copy(
                update={
                    "status": HydraJobStatus.EXECUTED,
                    "target_ticket": data.get("ticket"),
                    "latency_ms": latency_ms,
                    "updated_at": datetime.now(),
                }
            )
        except Exception as exc:
            return job.model_copy(
                update={
                    "status": HydraJobStatus.FAILED,
                    "error_message": f"Echec fermeture distante: {exc}",
                    "updated_at": datetime.now(),
                }
            )

    async def _dispatch_job(
        self,
        account: PropFirmAccount,
        job: HydraReplicationJob,
        event: TradeReplicationEvent,
        *,
        dry_run: bool,
    ) -> HydraReplicationJob:
        """
        Route un job vers le bon executeur selon le type d'evenement.

        Args:
            account (PropFirmAccount): Compte cible.
            job (HydraReplicationJob): Job a executer.
            event (TradeReplicationEvent): Evenement maitre.
            dry_run (bool): Mode simulation.

        Returns:
            HydraReplicationJob: Job final.
        """
        if event.event_type == HydraEventType.CLOSE:
            return await self._dispatch_close_job(account, job, dry_run=dry_run)
        return await self._dispatch_order_job(account, job, event, dry_run=dry_run)

    async def replicate(self, request: CopyTradeRequest) -> CopyTradeResult:
        """
        Replique un fill maitre vers tous les slaves eligibles.

        Args:
            request (CopyTradeRequest): Demande de replication.

        Returns:
            CopyTradeResult: Resultat consolide des jobs generes.
        """
        event = request.event
        result = CopyTradeResult(event_id=event.event_id)
        target_filter = {str(account_id) for account_id in request.target_accounts or []}
        slaves = await self.account_manager.get_slave_accounts(event.source_account_id)

        for account in slaves:
            if target_filter and str(account.id) not in target_filter:
                continue
            mapped_symbol = self._normalize_symbol(account, event.symbol)
            if not mapped_symbol:
                result.skipped_accounts.append(f"{account.name}: symbole interdit")
                continue
            if not account.risk_enabled:
                result.skipped_accounts.append(f"{account.name}: risque local desactive")
                continue
            volume = self._compute_slave_volume(account, event)
            if volume <= 0:
                result.skipped_accounts.append(f"{account.name}: volume nul apres scaling")
                continue
            existing_job = await self._find_existing_job(
                event_id=event.event_id,
                target_account_id=account.id,
            )
            if existing_job is not None:
                result.jobs.append(existing_job)
                result.skipped_accounts.append(f"{account.name}: fill deja replique")
                continue

            job = self._build_job(account, event, mapped_symbol, volume)
            await self._persist_job(job)
            final_job = await self._dispatch_job(account, job, event, dry_run=request.dry_run)
            await self._persist_job(final_job)
            await self._persist_metrics(
                account,
                {
                    "last_job_id": str(final_job.id),
                    "last_status": final_job.status.value,
                    "last_error": final_job.error_message,
                    "last_latency_ms": final_job.latency_ms,
                    "last_symbol": final_job.symbol,
                },
            )
            result.jobs.append(final_job)

        return result

    async def handle_execution_event(self, payload: dict[str, Any] | TradeReplicationEvent) -> CopyTradeResult:
        """
        Point d'entree appele par le compte maitre apres fill confirme.

        Args:
            payload (dict[str, Any] | TradeReplicationEvent): Evenement brut ou deja valide.

        Returns:
            CopyTradeResult: Resultat consolide de la replication.
        """
        event = payload if isinstance(payload, TradeReplicationEvent) else TradeReplicationEvent(**payload)
        return await self.replicate(CopyTradeRequest(event=event))

    async def set_terminal_health(self, health: HydraTerminalHealth) -> HydraTerminalHealth:
        """
        Enregistre la sante d'un terminal Hydra.

        Args:
            health (HydraTerminalHealth): Snapshot venant du terminal.

        Returns:
            HydraTerminalHealth: Snapshot persiste.
        """
        await self._persist_health(health)
        return health

    async def probe_account_health(self, account: PropFirmAccount) -> HydraTerminalHealth:
        """
        Sonde un executeur distant pour verifier la sante du terminal.

        Args:
            account (PropFirmAccount): Compte cible.

        Returns:
            HydraTerminalHealth: Etat courant estime ou sonde.
        """
        if not account.executor_url:
            health = HydraTerminalHealth(
                account_id=account.id,
                process_alive=False,
                mt5_connected=False,
                autotrading_enabled=False,
                terminal_path=account.terminal_path,
                wineprefix=account.wineprefix,
            )
            await self._persist_health(health)
            return health

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                response = await client.get(f"{account.executor_url}{self.health_path}")
                response.raise_for_status()
                data = response.json()
            health = HydraTerminalHealth(
                account_id=account.id,
                process_alive=bool(data.get("process_alive", True)),
                mt5_connected=bool(data.get("mt5_connected", False)),
                autotrading_enabled=bool(data.get("autotrading_enabled", False)),
                symbols_available=list(data.get("symbols_available") or []),
                latency_ms=data.get("latency_ms"),
                terminal_path=data.get("terminal_path") or account.terminal_path,
                wineprefix=data.get("wineprefix") or account.wineprefix,
            )
        except Exception as exc:
            health = HydraTerminalHealth(
                account_id=account.id,
                process_alive=False,
                mt5_connected=False,
                autotrading_enabled=False,
                terminal_path=account.terminal_path,
                wineprefix=account.wineprefix,
            )
            await self._persist_metrics(
                account,
                {
                    "last_health_error": str(exc),
                },
            )
        await self._persist_health(health)
        return health

    async def get_job(self, job_id: UUID) -> HydraReplicationJob | None:
        """
        Retourne un job Hydra par identifiant.
        """
        if job_id in self._jobs:
            return self._jobs[job_id]
        try:
            redis = get_redis_client()
            payload = await redis.cache_get(f"{self.REDIS_JOBS_PREFIX}{job_id}")
            if not payload:
                return None
            job = HydraReplicationJob(**payload)
            self._jobs[job.id] = job
            return job
        except Exception:
            return None

    async def list_jobs(self, *, limit: int = 50, account_id: UUID | None = None) -> list[HydraReplicationJob]:
        """
        Retourne les derniers jobs Hydra connus.

        Args:
            limit (int): Nombre maximal de jobs retournes.
            account_id (UUID | None): Filtre cible optionnel.

        Returns:
            list[HydraReplicationJob]: Jobs tries du plus recent au plus ancien.
        """
        jobs = list(self._jobs.values())
        if account_id is not None:
            jobs = [job for job in jobs if job.target_account_id == account_id]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[: max(1, limit)]

    async def get_account_metrics(self, account_id: UUID) -> dict[str, Any] | None:
        """
        Retourne les metriques et jobs recents d'un compte Hydra.

        Args:
            account_id (UUID): Compte cible.

        Returns:
            dict[str, Any] | None: Vue compacte du compte, ou `None` si absent.
        """
        account = await self.account_manager.get_account(account_id)
        if account is None:
            return None

        jobs = await self.list_jobs(limit=25, account_id=account_id)
        executed = [job for job in jobs if job.status == HydraJobStatus.EXECUTED]
        failed = [job for job in jobs if job.status in {HydraJobStatus.FAILED, HydraJobStatus.REJECTED}]
        latencies = [int(job.latency_ms) for job in executed if job.latency_ms is not None]
        average_latency = round(sum(latencies) / len(latencies), 1) if latencies else None
        return {
            "account": account.model_dump(mode="json"),
            "health": self._health.get(account_id).model_dump(mode="json") if account_id in self._health else None,
            "metrics": dict(self._metrics.get(str(account_id)) or {}),
            "jobs": [job.model_dump(mode="json") for job in jobs],
            "summary": {
                "total_jobs": len(jobs),
                "executed_jobs": len(executed),
                "failed_jobs": len(failed),
                "average_latency_ms": average_latency,
            },
        }

    async def get_aggregate(self, master_runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Construit une vue consolidee Hydra pour l'API et Nexus.

        Args:
            master_runtime (dict[str, Any] | None): Etat compact du master courant.

        Returns:
            dict[str, Any]: Vue consolidee comptes/jobs/sante.
        """
        accounts = await self.account_manager.get_all_accounts()
        slaves = [account for account in accounts if account.role.value == "slave"]
        jobs = await self.list_jobs(limit=20)
        for account in slaves:
            if account.id not in self._health:
                await self.probe_account_health(account)
        executed_jobs = [job for job in jobs if job.status == HydraJobStatus.EXECUTED]
        failed_jobs = [job for job in jobs if job.status in {HydraJobStatus.FAILED, HydraJobStatus.REJECTED}]
        pending_jobs = [job for job in jobs if job.status in {HydraJobStatus.PENDING, HydraJobStatus.DISPATCHED}]
        latency_values = [int(job.latency_ms) for job in executed_jobs if job.latency_ms is not None]
        average_latency = round(sum(latency_values) / len(latency_values), 1) if latency_values else None
        return {
            "enabled": True,
            "mode": "master_local_transitoire",
            "master_source_id": self.master_source_id,
            "master": master_runtime or {},
            "registry": self.account_manager.get_status(),
            "jobs": [job.model_dump(mode="json") for job in jobs],
            "health": [health.model_dump(mode="json") for health in self._health.values()],
            "metrics": list(self._metrics.values()),
            "summary": {
                "total_jobs": len(jobs),
                "executed_jobs": len(executed_jobs),
                "failed_jobs": len(failed_jobs),
                "pending_jobs": len(pending_jobs),
                "average_latency_ms": average_latency,
            },
        }
