"""
Gestionnaire Hydra des comptes master/slaves.

Ce module centralise le registre des comptes multi-MT5 :
- persistance Redis ;
- filtrage master/slaves ;
- quarantaine et pause operateur ;
- etat compact exploitable par Nexus.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from shared import HydraAccountRole, PropFirmAccount

logger = logging.getLogger(__name__)


class MultiAccountManager:
    """
    Gere les comptes Hydra et leur etat operateur.

    Chaque compte reste autonome, mais ce registre fournit :
    - la configuration metier de copie ;
    - les filtres master/slave ;
    - la quarantaine temporaire ;
    - la persistance Redis pour reprise rapide.
    """

    REDIS_PREFIX = "hydra:account:"

    def __init__(self) -> None:
        self.accounts: dict[UUID, PropFirmAccount] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """
        Charge les comptes Hydra depuis Redis au demarrage.
        """
        if self._initialized:
            return
        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            keys = await redis._client.keys(f"{self.REDIS_PREFIX}*")
            for key in keys:
                data = await redis.cache_get(key)
                if not data:
                    continue
                account = PropFirmAccount(**data)
                self.accounts[account.id] = account

            logger.info(
                "Hydra: %s compte(s) charge(s) depuis Redis.",
                len(self.accounts),
            )
        except Exception as exc:
            logger.warning("Hydra: Redis indisponible pour le registre comptes (%s).", exc)
        finally:
            self._initialized = True

    async def _persist_account(self, account: PropFirmAccount) -> None:
        """
        Persiste un compte Hydra en Redis si possible.

        Args:
            account (PropFirmAccount): Compte a serialiser.
        """
        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            await redis.cache_set(
                f"{self.REDIS_PREFIX}{account.id}",
                account.model_dump(mode="json"),
                ttl_seconds=None,
            )
        except Exception as exc:
            logger.warning("Hydra: persistance Redis echee pour %s (%s).", account.name, exc)

    async def add_account(self, account: PropFirmAccount) -> PropFirmAccount:
        """
        Ajoute un compte Hydra dans le registre.

        Args:
            account (PropFirmAccount): Nouveau compte a enregistrer.

        Returns:
            PropFirmAccount: Compte enregistre.

        Raises:
            ValueError: Si l'identifiant existe deja.
        """
        if account.id in self.accounts:
            raise ValueError(f"Le compte Hydra {account.id} existe deja.")

        self.accounts[account.id] = account
        await self._persist_account(account)
        logger.info("Hydra: compte ajoute (%s / %s).", account.name, account.role.value)
        return account

    async def update_account(self, account_id: UUID, patch: dict[str, Any]) -> PropFirmAccount | None:
        """
        Met a jour partiellement un compte Hydra.

        Args:
            account_id (UUID): Compte cible.
            patch (dict[str, Any]): Champs a modifier.

        Returns:
            PropFirmAccount | None: Compte mis a jour ou `None` si absent.
        """
        account = self.accounts.get(account_id)
        if account is None:
            return None
        updated = account.model_copy(update=patch)
        self.accounts[account_id] = updated
        await self._persist_account(updated)
        return updated

    async def get_account(self, account_id: UUID) -> PropFirmAccount | None:
        """
        Retourne un compte Hydra par identifiant.
        """
        return self.accounts.get(account_id)

    async def get_all_accounts(self) -> list[PropFirmAccount]:
        """
        Retourne tous les comptes Hydra connus.
        """
        return list(self.accounts.values())

    async def get_master_accounts(self) -> list[PropFirmAccount]:
        """
        Retourne les comptes marques comme maitres.
        """
        return [account for account in self.accounts.values() if account.role == HydraAccountRole.MASTER]

    async def get_slave_accounts(self, master_source_id: str | None = None) -> list[PropFirmAccount]:
        """
        Retourne les comptes esclaves eligibles pour un maitre donne.

        Args:
            master_source_id (str | None): Identifiant logique du maitre.

        Returns:
            list[PropFirmAccount]: Liste filtree de comptes esclaves actifs.
        """
        now = datetime.now()
        slaves: list[PropFirmAccount] = []
        for account in self.accounts.values():
            if account.role != HydraAccountRole.SLAVE:
                continue
            if not account.active or not account.copy_enabled:
                continue
            if account.quarantined_until and account.quarantined_until > now:
                continue
            if master_source_id and account.master_source_id and account.master_source_id != master_source_id:
                continue
            slaves.append(account)
        return slaves

    async def pause_account(self, account_id: UUID) -> PropFirmAccount | None:
        """
        Suspend la copie vers un compte Hydra.

        Args:
            account_id (UUID): Compte cible.

        Returns:
            PropFirmAccount | None: Compte mis a jour ou `None`.
        """
        return await self.update_account(account_id, {"copy_enabled": False})

    async def resume_account(self, account_id: UUID) -> PropFirmAccount | None:
        """
        Reprend la copie vers un compte Hydra.

        Args:
            account_id (UUID): Compte cible.

        Returns:
            PropFirmAccount | None: Compte mis a jour ou `None`.
        """
        return await self.update_account(
            account_id,
            {
                "copy_enabled": True,
                "quarantined_until": None,
                "quarantine_reason": None,
            },
        )

    async def quarantine_account(
        self,
        account_id: UUID,
        *,
        hours: int = 4,
        reason: str = "manual",
    ) -> PropFirmAccount | None:
        """
        Place un compte en quarantaine temporaire.

        Args:
            account_id (UUID): Compte cible.
            hours (int): Duree de quarantaine.
            reason (str): Motif operateur ou automatique.

        Returns:
            PropFirmAccount | None: Compte mis a jour ou `None`.
        """
        until = datetime.now() + timedelta(hours=max(1, hours))
        return await self.update_account(
            account_id,
            {
                "copy_enabled": False,
                "quarantined_until": until,
                "quarantine_reason": reason,
            },
        )

    async def remove_account(self, account_id: UUID) -> bool:
        """
        Supprime un compte Hydra.

        Args:
            account_id (UUID): Compte a retirer.

        Returns:
            bool: `True` si le compte a ete supprime.
        """
        if account_id not in self.accounts:
            return False

        del self.accounts[account_id]
        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            await redis._client.delete(f"{self.REDIS_PREFIX}{account_id}")
        except Exception:
            pass

        logger.info("Hydra: compte supprime (%s).", account_id)
        return True

    def get_status(self) -> dict[str, Any]:
        """
        Retourne une synthese compacte du registre Hydra.

        Returns:
            dict[str, Any]: Resume exploitable par l'API.
        """
        now = datetime.now()
        masters = 0
        slaves = 0
        quarantined = 0
        accounts_payload: list[dict[str, Any]] = []
        for account in self.accounts.values():
            if account.role == HydraAccountRole.MASTER:
                masters += 1
            else:
                slaves += 1
            if account.quarantined_until and account.quarantined_until > now:
                quarantined += 1
            accounts_payload.append(
                {
                    "id": str(account.id),
                    "name": account.name,
                    "role": account.role.value,
                    "login": account.login,
                    "broker": account.broker,
                    "server": account.server,
                    "active": account.active,
                    "copy_enabled": account.copy_enabled,
                    "scaling_mode": account.scaling_mode.value,
                    "scaling_factor": float(account.scaling_factor),
                    "executor_url": account.executor_url,
                    "master_source_id": account.master_source_id,
                    "quarantined_until": account.quarantined_until.isoformat() if account.quarantined_until else None,
                    "quarantine_reason": account.quarantine_reason,
                }
            )
        return {
            "total_accounts": len(self.accounts),
            "masters": masters,
            "slaves": slaves,
            "quarantined_accounts": quarantined,
            "accounts": accounts_payload,
        }
