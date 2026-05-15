"""Orchestrateur multi-comptes pour agents followers locaux."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from eva_banker.follower.agent import EventCallback, FollowerAgent
from eva_banker.follower.config import FollowerAccountConfig, FollowerFleetConfig
from eva_banker.follower.models import FollowerRuntimeStatus

AgentFactory = Callable[[FollowerAccountConfig, EventCallback | None], Any]


class FollowerFleetManager:
    """Pilote plusieurs agents followers dans une seule application locale."""

    def __init__(
        self,
        config: FollowerFleetConfig,
        *,
        agent_factory: AgentFactory | None = None,
        event_callback: EventCallback | None = None,
    ) -> None:
        """Initialise l'orchestrateur de flotte.

        Args:
            config (FollowerFleetConfig): Configuration multi-comptes.
            agent_factory (AgentFactory | None): Fabrique injectee pour tests.
            event_callback (EventCallback | None): Callback UI/log commun.
        """

        self.config = config
        self.event_callback = event_callback
        self.agent_factory = agent_factory or self._default_agent_factory
        self.running = False
        self.agents: dict[str, Any] = {}
        self.tasks: dict[str, asyncio.Task] = {}

    async def run_forever(self) -> None:
        """Demarre tous les comptes actifs et maintient la flotte vivante."""

        self.running = True
        await self.start_all()
        try:
            while self.running:
                await asyncio.sleep(0.5)
        finally:
            await self.stop_all()

    async def start_all(self) -> None:
        """Demarre tous les comptes actifs non deja lances."""

        for account in self.config.accounts:
            if account.enabled:
                await self.start_account(account.client_id)

    async def stop_all(self) -> None:
        """Arrete tous les agents de la flotte."""

        for client_id in list(self.agents.keys()):
            await self.stop_account(client_id)
        self.running = False

    async def start_account(self, client_id: str) -> bool:
        """Demarre un compte precis si sa configuration existe.

        Args:
            client_id (str): Identifiant du compte follower.

        Returns:
            bool: True si un agent est actif apres l'appel.
        """

        account = self.get_account(client_id)
        if account is None:
            self._emit(f"Compte inconnu: {client_id}")
            return False
        if not account.enabled:
            self._emit(f"Compte desactive ignore: {account.account_label}")
            return False
        if client_id in self.agents:
            self._emit(f"Compte deja actif: {account.account_label}")
            return True

        agent = self.agent_factory(account, self._account_event_callback(account))
        self.agents[client_id] = agent
        self.tasks[client_id] = asyncio.create_task(agent.run_forever())
        self._emit(f"Compte demarre: {account.account_label}")
        return True

    async def stop_account(self, client_id: str) -> bool:
        """Arrete un compte precis.

        Args:
            client_id (str): Identifiant du compte follower.

        Returns:
            bool: True si un agent a ete arrete.
        """

        agent = self.agents.pop(client_id, None)
        task = self.tasks.pop(client_id, None)
        if agent is None:
            return False
        if hasattr(agent, "running"):
            agent.running = False
        if hasattr(agent, "close"):
            await agent.close()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
        self._emit(f"Compte arrete: {client_id}")
        return True

    def get_account(self, client_id: str) -> FollowerAccountConfig | None:
        """Retrouve une configuration de compte.

        Args:
            client_id (str): Identifiant recherche.

        Returns:
            FollowerAccountConfig | None: Compte trouve ou None.
        """

        for account in self.config.accounts:
            if account.client_id == client_id:
                return account
        return None

    def get_statuses(self) -> list[FollowerRuntimeStatus]:
        """Retourne le statut de tous les comptes de la flotte.

        Returns:
            list[FollowerRuntimeStatus]: Statuts actifs et inactifs.
        """

        statuses: list[FollowerRuntimeStatus] = []
        for account in self.config.accounts:
            agent = self.agents.get(account.client_id)
            if agent is not None and hasattr(agent, "get_status"):
                statuses.append(agent.get_status())
            else:
                statuses.append(
                    FollowerRuntimeStatus(
                        client_id=account.client_id,
                        account_label=account.account_label,
                        running=False,
                        paused=not account.enabled,
                        dry_run=account.dry_run,
                    )
                )
        return statuses

    @staticmethod
    def _default_agent_factory(
        account: FollowerAccountConfig,
        event_callback: EventCallback | None,
    ) -> FollowerAgent:
        """Cree un agent follower standard.

        Args:
            account (FollowerAccountConfig): Compte a piloter.
            event_callback (EventCallback | None): Callback UI/log.

        Returns:
            FollowerAgent: Agent configure.
        """

        return FollowerAgent(account, event_callback=event_callback)

    def _account_event_callback(self, account: FollowerAccountConfig) -> EventCallback:
        """Construit un callback prefixe par compte.

        Args:
            account (FollowerAccountConfig): Compte source.

        Returns:
            EventCallback: Callback prefixe.
        """

        def callback(message: str) -> None:
            self._emit(f"{account.account_label}: {message}")

        return callback

    def _emit(self, message: str) -> None:
        """Publie un evenement de flotte."""

        if self.event_callback:
            self.event_callback(message)
