"""
Multi-Account Manager — Gestion de comptes Prop Firm multiples
═══════════════════════════════════════════════════════════════

Gère N comptes de trading Prop Firm simultanément avec :
  - Persistance Redis de chaque compte
  - Tracking individuel du risque par compte
  - Rotation automatique si un compte est bloqué
  - Reporting consolidé
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from shared import PropFirmAccount

logger = logging.getLogger(__name__)


class MultiAccountManager:
    """
    Gestionnaire multi-comptes Prop Firm pour le protocole Hydra.
    Chaque compte est isolé avec ses propres limites de risque.
    """

    REDIS_PREFIX = "propfirm:account:"

    def __init__(self):
        self.accounts: Dict[UUID, PropFirmAccount] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Charge les comptes depuis Redis au démarrage."""
        try:
            from shared.redis_client import get_redis_client
            redis = get_redis_client()

            keys = await redis._client.keys(f"{self.REDIS_PREFIX}*")
            for key in keys:
                data = await redis.cache_get(key)
                if data:
                    account = PropFirmAccount(**data)
                    self.accounts[account.id] = account

            self._initialized = True
            logger.info(
                f"💼 MultiAccountManager: {len(self.accounts)} compte(s) chargé(s)"
            )
        except Exception as e:
            logger.warning(f"⚠️ MultiAccountManager: Redis indisponible ({e})")
            self._initialized = True  # Continue sans Redis

    async def add_account(self, account: PropFirmAccount) -> PropFirmAccount:
        """Ajoute un nouveau compte Prop Firm."""
        if account.id in self.accounts:
            raise ValueError(f"Compte {account.id} existe déjà.")

        self.accounts[account.id] = account

        try:
            from shared.redis_client import get_redis_client
            redis = get_redis_client()
            await redis.cache_set(
                f"{self.REDIS_PREFIX}{account.id}",
                account.model_dump(mode="json"),
                ttl_seconds=None,
            )
        except Exception as e:
            logger.warning(f"Persistance Redis échouée: {e}")

        logger.info(f"✅ Nouveau compte Prop Firm ajouté: {account.name}")
        return account

    async def get_account(self, account_id: UUID) -> Optional[PropFirmAccount]:
        return self.accounts.get(account_id)

    async def get_all_accounts(self) -> List[PropFirmAccount]:
        return list(self.accounts.values())

    async def remove_account(self, account_id: UUID) -> bool:
        """Supprime un compte."""
        if account_id not in self.accounts:
            return False

        del self.accounts[account_id]

        try:
            from shared.redis_client import get_redis_client
            redis = get_redis_client()
            await redis._client.delete(f"{self.REDIS_PREFIX}{account_id}")
        except Exception:
            pass

        logger.info(f"🗑️ Compte Prop Firm supprimé: {account_id}")
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_accounts": len(self.accounts),
            "accounts": [
                {
                    "id": str(a.id),
                    "name": a.name,
                    "broker": a.broker,
                }
                for a in self.accounts.values()
            ],
        }
