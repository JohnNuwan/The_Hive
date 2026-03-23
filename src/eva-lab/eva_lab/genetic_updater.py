"""Gere le registre genetique des champions par horizon."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HORIZONS = ["scalp", "intraday", "swing"]


class GeneticUpdater:
    """Maintient le registre ADN des generations MuZero.

    Le registre historique a existe sous deux schemas:
    - un schema legacy avec ``current_champion`` unique
    - un schema multi-horizon avec ``champions`` par horizon

    Cette classe normalise automatiquement le registre afin que l'Arena,
    la promotion live et Nexus lisent toujours la meme source de verite.
    """

    def __init__(self, registry_path: str = "data/models_registry.json") -> None:
        """Initialise le registre genetique.

        Args:
            registry_path (str): Chemin du registre JSON.
        """
        self.registry_path = Path(registry_path)
        self._ensure_registry()

    def _baseline_generation(self) -> dict[str, Any]:
        """Construit l'entree baseline compatible multi-horizon.

        Returns:
            dict[str, Any]: Metriques initiales du champion baseline.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "win_rate": {horizon: 50.0 for horizon in HORIZONS},
            "return_pct": {horizon: 0.0 for horizon in HORIZONS},
            "battles_won": {horizon: 0 for horizon in HORIZONS},
            "horizon_accuracy": {horizon: 0.33 for horizon in HORIZONS},
        }

    def _initial_registry(self) -> dict[str, Any]:
        """Retourne un registre multi-horizon valide.

        Returns:
            dict[str, Any]: Registre ADN initialise.
        """
        baseline_id = "gen_000_baseline"
        champions = {horizon: baseline_id for horizon in HORIZONS}
        return {
            "version": "2.1-MTF",
            "description": "Registre genetique multi-horizon THE HIVE",
            "current_champion": baseline_id,
            "champions": champions,
            "generations": {
                baseline_id: self._baseline_generation(),
            },
        }

    def _ensure_registry(self) -> None:
        """Cree le registre si absent puis force le schema courant."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save_registry(self._initial_registry())
            return

        registry = self._load_registry()
        self._save_registry(registry)

    def _load_registry(self) -> dict[str, Any]:
        """Charge et normalise le registre ADN.

        Returns:
            dict[str, Any]: Registre multi-horizon normalise.
        """
        try:
            raw_data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Registre genetique illisible, recreation du baseline: %s", exc)
            return self._initial_registry()

        baseline_id = "gen_000_baseline"
        registry = dict(raw_data or {})
        registry.setdefault("version", "2.1-MTF")
        registry.setdefault("description", "Registre genetique multi-horizon THE HIVE")

        generations = registry.get("generations")
        if not isinstance(generations, dict):
            generations = {}
        registry["generations"] = generations
        if baseline_id not in generations:
            generations[baseline_id] = self._baseline_generation()

        legacy_champion = str(registry.get("current_champion") or baseline_id)
        champions = registry.get("champions")
        if not isinstance(champions, dict):
            champions = {}

        normalized_champions: dict[str, str] = {}
        for horizon in HORIZONS:
            champion_id = str(champions.get(horizon) or legacy_champion or baseline_id)
            normalized_champions[horizon] = champion_id
        registry["champions"] = normalized_champions

        # Compatibilite legacy: on conserve la clef historique, mais elle suit
        # desormais le champion intraday au lieu de forcer tout le systeme.
        registry["current_champion"] = normalized_champions.get("intraday", baseline_id)
        return registry

    def _save_registry(self, registry: dict[str, Any]) -> None:
        """Ecrit le registre ADN sur disque.

        Args:
            registry (dict[str, Any]): Registre normalise a persister.
        """
        self.registry_path.write_text(
            json.dumps(registry, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_champion(self, horizon: str = "intraday") -> str:
        """Retourne le champion ADN courant d'un horizon.

        Args:
            horizon (str): Horizon cible.

        Returns:
            str: Identifiant de generation du champion.
        """
        registry = self._load_registry()
        return str(registry["champions"].get(horizon, "gen_000_baseline"))

    def get_all_champions(self) -> dict[str, str]:
        """Retourne la table des champions ADN par horizon.

        Returns:
            dict[str, str]: Identifiants de champion par horizon.
        """
        registry = self._load_registry()
        return {horizon: str(registry["champions"].get(horizon, "gen_000_baseline")) for horizon in HORIZONS}

    def register_new_generation(
        self,
        gen_id: str,
        metrics: dict[str, Any],
        is_champion: bool = False,
        horizon: str | None = None,
    ) -> None:
        """Enregistre une nouvelle generation MuZero.

        Args:
            gen_id (str): Identifiant unique de generation.
            metrics (dict[str, Any]): Metriques de la generation.
            is_champion (bool): Indique si la generation devient championne.
            horizon (str | None): Horizon cible. ``None`` signifie tous les horizons.
        """
        registry = self._load_registry()
        registry["generations"][gen_id] = {
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        if is_champion:
            if horizon:
                previous = registry["champions"].get(horizon, "gen_000_baseline")
                registry["champions"][horizon] = gen_id
                logger.info(
                    "Mutation ADN %s: champion %s -> %s.",
                    horizon.upper(),
                    previous,
                    gen_id,
                )
            else:
                for current_horizon in HORIZONS:
                    registry["champions"][current_horizon] = gen_id
                logger.info("Mutation ADN globale: %s devient champion multi-horizon.", gen_id)

            registry["current_champion"] = registry["champions"].get("intraday", gen_id)

        self._save_registry(registry)

    def get_performance_summary(self) -> dict[str, Any]:
        """Retourne un resume de performance des champions ADN.

        Returns:
            dict[str, Any]: Metriques de synthese par horizon.
        """
        registry = self._load_registry()
        summary: dict[str, Any] = {}

        for horizon, gen_id in registry["champions"].items():
            generation = registry["generations"].get(gen_id, {})
            win_rate = generation.get("win_rate", {})
            return_pct = generation.get("return_pct", {})

            summary[horizon] = {
                "champion": gen_id,
                "win_rate": win_rate.get(horizon, win_rate) if isinstance(win_rate, dict) else win_rate,
                "return_pct": return_pct.get(horizon, return_pct) if isinstance(return_pct, dict) else return_pct,
            }

        return summary

    def check_for_updates(self) -> dict[str, Any]:
        """Expose une reponse legacy attendue par d'anciens appels.

        Returns:
            dict[str, Any]: Reponse de compatibilite.
        """
        return {
            "updates_found": 0,
            "type": "TRADING_STRATEGY",
            "action": "NO_ACTION",
            "safety_check": "PASSED",
        }
