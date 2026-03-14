"""
Cerveau de l'Expert Banker (The Brain).
Contient la logique dÃ©cisionnelle (Manager), l'exÃ©cution (Worker) et la boucle d'autonomie.
"""

import asyncio
import logging
from decimal import Decimal
from uuid import UUID
from datetime import datetime, timedelta
import os
import aiohttp
import random
import uuid

from shared.redis_client import get_redis_client

from shared import (
    TradeAction,
    TradeOrder,
    symlog,
    calculate_var,
    calculate_cvar,
    get_settings,
)
from eva_banker.services.mt5 import MT5Service
from eva_banker.skill_library import SkillLibrary, SkilledBehavior
from eva_banker.models.gnn_model import TFTGNNModel
from eva_banker.services.risk import RiskValidator  # Type hinting only if needed at runtime
from eva_banker.strategist import Strategist
from eva_banker.nemesis import get_nemesis_system # Import the Nemesis System
from eva_banker.services.news_filter import NewsFilterService # Import News Filter

logger = logging.getLogger(__name__)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MANAGER (DECISION)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class BankerManager:
    """
    NIVEAU HAUT : Le Manager (Abstract World Model).
    Planifie les stratÃ©gies en utilisant TFT-GNN et la conscience du risque.
    """
    def __init__(self, library: SkillLibrary):
        self.library = library
        # Initialisation du modÃ¨le (dims fictives pour l'exemple)
        self.brain = TFTGNNModel(asset_dim=5, temporal_dim=64, hidden_dim=128)

    def plan_strategy(self, market_history: dict) -> SkilledBehavior:
        """
        Analyse le marchÃ© via TFT-GNN et injecte VaR/CVaR.
        """
        # 1. Calcul des mÃ©triques de risque adaptatives (Inhibiteur interne)
        returns = market_history.get("returns", [])
        var = calculate_var(returns)
        cvar = calculate_cvar(returns)
        
        # 2. PrÃ©paration des donnÃ©es pour le modÃ¨le (NormalisÃ©es via Symlog)
        price = symlog(market_history.get("price", 0))
        
        # Si le risque (VaR) est trop Ã©levÃ©, on bascule en mode conservateur
        if var < -0.02: # Perte potentielle > 2% attendue
            logger.warning("High VaR detected. Selecting HEDGING skill.")
            return SkilledBehavior.HEDGING
            
        return SkilledBehavior.SCALPING


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# WORKER (EXECUTION)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class BankerWorker:
    """
    NIVEAU BAS : L'ExÃ©cutant (Worker).
    Support de GhostShield pour l'invisibilitÃ© HFT.
    """
    def __init__(self, mt5_service: MT5Service, ghost_shield=None):
        self.mt5 = mt5_service
        self.ghost = ghost_shield

    async def execute_skill(self, skill: SkilledBehavior, order: TradeOrder):
        logger.info(f"Worker executing skill: {skill}")
        if self.ghost and skill != SkilledBehavior.HEDGING: # Le hedging doit Ãªtre direct
            return await self.ghost.execute_obfuscated_order(order)
        return await self.mt5.execute_skill(skill, order)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ENGINE (AUTONOMY)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class AutoTradingEngine:
    """
    Moteur de Trading Automatique ("Weekend Drift").
    Orchestre la boucle : Analyse -> Planification -> ExÃ©cution.
    """
    def __init__(self, manager: BankerManager, worker: BankerWorker, mt5: MT5Service, risk: RiskValidator):
        self.manager = manager
        self.worker = worker
        self.mt5 = mt5
        self.risk = risk
        self.settings = get_settings()
        self.is_active = False
        self._loop_task = None
        self._daily_report_task = None
        self._news_task = None
        self.symbols = list(dict.fromkeys(self.settings.banker_symbols))
        self.risk.register_symbol_universe({symbol: self.mt5.classify_symbol(symbol) or "unknown" for symbol in self.symbols})
        self.latest_decisions = {} # Stores latest analysis per symbol
        self._symbol_cursor = 0
        self._last_universe_refresh = None
        self._dynamic_universe_enabled = self._env_flag("BANKER_DYNAMIC_UNIVERSE", True)
        self._scan_forex = self._env_flag("BANKER_SCAN_FOREX", True)
        self._scan_cfd = self._env_flag("BANKER_SCAN_CFD", True)
        self._scan_crypto = self._env_flag("BANKER_SCAN_CRYPTO", True)
        self._scan_batch_size = max(1, self._env_int("BANKER_SCAN_BATCH_SIZE", 40))
        self._universe_refresh_minutes = max(5, self._env_int("BANKER_UNIVERSE_REFRESH_MINUTES", 240))
        self._universe_max_symbols = max(0, self._env_int("BANKER_UNIVERSE_MAX_SYMBOLS", 0))
        self._lab_universe_enabled = self._env_flag("BANKER_LIMIT_TO_LAB_UNIVERSE", True)
        self._lab_universe_refresh_minutes = max(
            5,
            self._env_int("BANKER_LAB_UNIVERSE_REFRESH_MINUTES", 30),
        )
        self._lab_universe_horizon = self._env_text("BANKER_LAB_UNIVERSE_HORIZON", "intraday").lower()
        self._live_inference_horizon = self._env_text("BANKER_LIVE_HORIZON", "auto").lower()
        self._require_valid_champion = self._env_flag("BANKER_REQUIRE_VALID_CHAMPION", True)
        self._startup_alert_cooldown = timedelta(
            minutes=max(1, self._env_int("BANKER_STARTUP_ALERT_COOLDOWN_MINUTES", 20))
        )
        self._veto_alert_cooldown = timedelta(
            minutes=max(1, self._env_int("BANKER_VETO_ALERT_COOLDOWN_MINUTES", 15))
        )
        self._symbol_entry_cooldown = timedelta(
            minutes=max(1, self._env_int("BANKER_SYMBOL_ENTRY_COOLDOWN_MINUTES", 30))
        )
        self._drift_interval_seconds = max(15, self._env_int("BANKER_DRIFT_INTERVAL_SECONDS", 60))
        self._shepherd_min_age_seconds = max(15, self._env_int("BANKER_SHEPHERD_MIN_AGE_SECONDS", 45))
        self._scalp_stale_minutes = max(3, self._env_int("BANKER_SCALP_STALE_MINUTES", 20))
        self._startup_alert_state_file = os.getenv(
            "BANKER_STARTUP_ALERT_STATE_FILE",
            os.path.join(os.getcwd(), ".banker_startup_alert"),
        )
        self._pause_log_state = {}

        # Sprint 7: The Cortex
        self.cortex = Strategist(mt5_service=mt5)

        # Sprint 8.5: Telegram Notification
        from shared.telegram_client import TelegramClient
        self.telegram = TelegramClient()

        # Sprint 9: Close Detection & Anti-Spam
        self._known_tickets = set()         # Tickets currently open (for close detection)
        self._last_veto_sent = {}           # symbol -> datetime (anti-spam)
        self._trade_open_info = {}          # ticket -> {symbol, action, entry_price, open_time, comment}
        self._last_symbol_entry_at = {}     # symbol -> datetime de la derniere entree executee
        self._lab_universe_symbols = list(self.symbols)
        self._lab_universe_source = "local_fallback"
        self._lab_universe_last_refresh = None
        self._lab_universe_gate_allowed = False
        self._lab_universe_selection = "none"
        self._lab_universe_gate_reason = "unknown"

        # Sprint 10: News Filter
        self.news = NewsFilterService(filter_minutes=30)

    async def start(self):
        """Demarre le pilote automatique."""
        if self.is_active:
            return
        self.is_active = True
        await self.refresh_symbol_universe()
        await self._sync_open_positions()

        self._loop_task = asyncio.create_task(self._drift_loop())
        self._daily_report_task = asyncio.create_task(self._half_day_report_loop())
        self._news_task = asyncio.create_task(self.news.start_monitoring())
        logger.info(
            "AUTO-TRADING ENGINE STARTED: universe=%s batch=%s",
            len(self.symbols),
            len(self.get_symbol_batch(advance=False)),
        )

        if self._should_send_startup_alert():
            self.telegram.send_sync(
                f"🐝 *THE HIVE IS AWAKE*\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Universe: {len(self.symbols)} actifs\n"
                f"🔎 Batch: {len(self.get_symbol_batch(advance=False))} actifs / cycle\n"
                f"🧭 Sample: {self._format_symbol_summary()}\n"
                f"⚙️ Risk: {self.risk.max_risk_per_trade}%\n"
                f"🕐 {datetime.now().strftime('%H:%M')}"
            )
        else:
            logger.info("Notification de demarrage ignoree par cooldown Telegram.")

    async def stop(self):
        """Arrete le pilote automatique."""
        if not self.is_active:
            return
        self.is_active = False
        for task in [self._loop_task, self._daily_report_task, self._news_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._loop_task = None
        self._daily_report_task = None
        self._news_task = None
        logger.info("AUTO-TRADING ENGINE STOPPED")

    @staticmethod
    def _env_flag(name: str, default: bool = True) -> bool:
        """Lit un bool depuis l'environnement."""
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        """Lit un entier depuis l'environnement."""
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("Variable %s invalide (%s). Repli sur %s.", name, raw, default)
            return default

    @staticmethod
    def _env_text(name: str, default: str) -> str:
        """Lit une chaine depuis l'environnement avec repli.

        Args:
            name (str): Nom de la variable d'environnement.
            default (str): Valeur de repli si la variable est absente.

        Returns:
            str: Valeur textuelle nettoyee.
        """
        raw = os.getenv(name)
        if raw is None:
            return default
        cleaned = raw.strip()
        return cleaned or default

    def _log_pause_state(
        self,
        key: str,
        message: str,
        cooldown_seconds: int = 180,
    ) -> None:
        """Journalise un etat de pause sans spammer la console.

        Args:
            key (str): Cle stable identifiant le type de pause.
            message (str): Message a afficher si le cooldown est expire.
            cooldown_seconds (int): Delai minimal entre deux logs identiques.
        """
        now = datetime.now()
        last_logged = self._pause_log_state.get(key)
        if last_logged is None or (now - last_logged).total_seconds() >= cooldown_seconds:
            logger.warning(message)
            self._pause_log_state[key] = now

    def _clear_pause_state(self, *keys: str) -> None:
        """Efface les etats de pause devenus obsoletes.

        Args:
            *keys (str): Cles de pause a oublier.
        """
        for key in keys:
            self._pause_log_state.pop(key, None)

    @staticmethod
    def _json_safe_value(value):
        """Convertit une valeur en type JSON natif.

        Args:
            value: Valeur issue du moteur local, de MT5, de ``numpy`` ou de ``pandas``.

        Returns:
            object: Valeur compatible avec ``json.dumps``.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): AutoTradingEngine._json_safe_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [AutoTradingEngine._json_safe_value(item) for item in value]
        if hasattr(value, "item") and callable(value.item):
            try:
                return AutoTradingEngine._json_safe_value(value.item())
            except Exception:
                pass
        if hasattr(value, "isoformat") and callable(value.isoformat):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value)

    def _resolve_inference_horizon(self, skill: SkilledBehavior | None) -> str:
        """Choisit l'horizon live le plus adapte a la skill courante.

        Args:
            skill (SkilledBehavior | None): Skill decidee par le manager.

        Returns:
            str: Horizon MuZero a transmettre au Lab.
        """
        if self._live_inference_horizon != "auto":
            return self._live_inference_horizon

        if skill == SkilledBehavior.SCALPING:
            return "scalp"
        if skill in {SkilledBehavior.HEDGING, SkilledBehavior.ACCUMULATION}:
            return "swing"
        return "intraday"

    async def refresh_symbol_universe(self, force: bool = False) -> list[str]:
        """Rafraichit l'univers de marche depuis MT5."""
        if not self._dynamic_universe_enabled:
            return self.symbols

        previous_symbols = list(self.symbols)
        now = datetime.now()
        if (
            not force
            and self._last_universe_refresh is not None
            and (now - self._last_universe_refresh) < timedelta(minutes=self._universe_refresh_minutes)
        ):
            return self.symbols

        discovered = await self.mt5.discover_symbols(
            include_forex=self._scan_forex,
            include_cfd=self._scan_cfd,
            include_crypto=self._scan_crypto,
            max_symbols=self._universe_max_symbols,
        )
        if not discovered:
            logger.warning("Univers dynamique vide. Conservation de la liste precedente.")
            return self.symbols

        if self._lab_universe_enabled:
            live_symbols = await self._refresh_lab_live_universe(force=force)
            if live_symbols:
                allowed_symbols = set(live_symbols)
                restricted = [symbol for symbol in discovered if symbol in allowed_symbols]
                if restricted:
                    logger.info(
                        "Univers live restreint par EVA Lab: %s/%s symboles (%s).",
                        len(restricted),
                        len(discovered),
                        self._lab_universe_source,
                    )
                    discovered = restricted
                else:
                    logger.warning(
                        "Aucun symbole MT5 ne correspond a l'univers EVA Lab (%s). Conservation de la liste precedente.",
                        self._lab_universe_source,
                    )
                    return previous_symbols
            elif previous_symbols:
                logger.warning(
                    "Univers EVA Lab indisponible. Conservation de la liste precedente (%s symboles).",
                    len(previous_symbols),
                )
                return previous_symbols

        self.symbols = discovered
        self._last_universe_refresh = now
        self._symbol_cursor = 0
        self.risk.register_symbol_universe(
            {
                symbol: self.mt5.classify_symbol(symbol) or "unknown"
                for symbol in self.symbols
            }
        )
        logger.info("Univers de marche mis a jour: %s symboles detectes.", len(self.symbols))
        return self.symbols

    async def _refresh_lab_live_universe(self, force: bool = False) -> list[str]:
        """Charge l'univers live recommande par EVA Lab.

        Args:
            force (bool): Force une requete immediate meme si le cache est frais.

        Returns:
            list[str]: Liste de symboles recommandes pour le live.
        """
        if not self._lab_universe_enabled:
            return self._lab_universe_symbols

        now = datetime.now()
        if (
            not force
            and self._lab_universe_last_refresh is not None
            and (now - self._lab_universe_last_refresh)
            < timedelta(minutes=self._lab_universe_refresh_minutes)
        ):
            return self._lab_universe_symbols

        lab_host = self._env_text("LAB_HOST", "localhost")
        lab_port = self._env_int("LAB_PORT", 8600)
        url = (
            f"http://{lab_host}:{lab_port}/live/universe"
            f"?horizon={self._lab_universe_horizon}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5.0) as response:
                    if response.status != 200:
                        logger.warning(
                            "Univers EVA Lab indisponible (%s). HTTP %s.",
                            self._lab_universe_horizon,
                            response.status,
                        )
                        return self._lab_universe_symbols

                    payload = await response.json()
                    live_universe = payload.get("live_universe", {}) or {}
                    symbols = [
                        str(symbol).strip()
                        for symbol in (live_universe.get("symbols", []) or [])
                        if str(symbol).strip()
                    ]
                    self._lab_universe_symbols = list(dict.fromkeys(symbols))
                    self._lab_universe_source = str(live_universe.get("source") or "unknown")
                    self._lab_universe_gate_allowed = bool(
                        (payload.get("promotion_gate", {}) or {}).get("allowed", False)
                    )
                    self._lab_universe_selection = str(payload.get("selection") or "none")
                    self._lab_universe_gate_reason = str(
                        (payload.get("promotion_gate", {}) or {}).get("reason") or "unknown"
                    )
                    self._lab_universe_last_refresh = now
                    if self._require_valid_champion and not self._lab_universe_gate_allowed:
                        logger.warning(
                            "Live gele: aucun champion valide pour %s (%s).",
                            self._lab_universe_horizon,
                            self._lab_universe_gate_reason,
                        )
                    return self._lab_universe_symbols
        except Exception as exc:
            logger.warning("Lecture de l'univers EVA Lab impossible: %s", exc)
            return self._lab_universe_symbols

    def get_live_universe_status(self) -> dict[str, object]:
        """Expose l'etat de restriction d'univers utilise par le banker.

        Returns:
            dict[str, object]: Etat du cache EVA Lab et univers courant.
        """
        return {
            "enabled": self._lab_universe_enabled,
            "horizon": self._lab_universe_horizon,
            "source": self._lab_universe_source,
            "symbols_total": len(self._lab_universe_symbols),
            "gate_allowed": self._lab_universe_gate_allowed,
            "selection": self._lab_universe_selection,
            "gate_reason": self._lab_universe_gate_reason,
            "require_valid_champion": self._require_valid_champion,
            "live_entries_allowed": (not self._require_valid_champion) or self._lab_universe_gate_allowed,
            "last_refresh": (
                self._lab_universe_last_refresh.isoformat()
                if self._lab_universe_last_refresh is not None
                else None
            ),
        }

    def _is_live_model_allowed(self, inference_result: dict[str, object] | None) -> tuple[bool, str]:
        """Valide qu'une prediction provient bien d'un champion live autorise.

        Args:
            inference_result (dict[str, object] | None): Reponse brute d'EVA Lab.

        Returns:
            tuple[bool, str]: ``(autorise, raison)`` pour l'execution live.
        """
        if not self._require_valid_champion:
            return True, "validation_desactivee"

        payload = inference_result or {}
        selection = str(payload.get("selection") or self._lab_universe_selection or "none").lower()
        if selection in {"champion", "legacy_champion"}:
            return True, "champion_valide"

        reason = str(payload.get("reason") or self._lab_universe_gate_reason or selection or "unknown")
        return False, reason

    def get_symbol_batch(self, advance: bool = True) -> list[str]:
        """Retourne le prochain lot de symboles a scanner."""
        if not self.symbols:
            return []

        batch_size = min(self._scan_batch_size, len(self.symbols))
        start_index = self._symbol_cursor % len(self.symbols)
        batch = [
            self.symbols[(start_index + offset) % len(self.symbols)]
            for offset in range(batch_size)
        ]
        if advance:
            self._symbol_cursor = (start_index + batch_size) % len(self.symbols)
        return batch

    def _format_symbol_summary(self, max_display: int = 12) -> str:
        """Formate un apercu court de l'univers pour Telegram."""
        if not self.symbols:
            return "aucun symbole"
        if len(self.symbols) <= max_display:
            return ", ".join(self.symbols)
        head = ", ".join(self.symbols[:max_display])
        return f"{head} ... (+{len(self.symbols) - max_display})"

    def _should_send_startup_alert(self) -> bool:
        """Evite de reemettre l'alerte de demarrage a chaque restart local."""
        now = datetime.now()
        state_path = os.path.abspath(self._startup_alert_state_file)
        try:
            if os.path.exists(state_path):
                previous_raw = open(state_path, "r", encoding="utf-8").read().strip()
                previous = datetime.fromisoformat(previous_raw)
                if now - previous < self._startup_alert_cooldown:
                    return False
        except Exception as exc:
            logger.warning("Etat de cooldown Telegram illisible: %s", exc)

        try:
            with open(state_path, "w", encoding="utf-8") as handle:
                handle.write(now.isoformat())
        except Exception as exc:
            logger.warning("Impossible d'ecrire le cooldown Telegram: %s", exc)
        return True

    def _should_send_veto_alert(self, veto_key: str) -> bool:
        """Applique un cooldown sur les alertes Telegram de veto.

        Args:
            veto_key (str): Cle logique de l'alerte a limiter.

        Returns:
            bool: ``True`` si l'alerte peut etre emise immediatement.
        """
        now = datetime.now()
        previous = self._last_veto_sent.get(veto_key)
        if previous and (now - previous) < self._veto_alert_cooldown:
            return False
        self._last_veto_sent[veto_key] = now
        return True

    @staticmethod
    def _is_unusable_reasoning(reasoning: str) -> bool:
        """Detecte un raisonnement LLM inutilisable pour Telegram.

        Args:
            reasoning (str): Texte de synthese recu.

        Returns:
            bool: ``True`` si le texte est vide ou correspond a un echec connu.
        """
        if not reasoning:
            return True
        lowered = reasoning.strip().lower()
        error_markers = (
            "llm connection failed",
            "error:",
            "llm unreachable",
            "lab error",
            "http ",
            "notfounderror",
        )
        return any(marker in lowered for marker in error_markers)

    @staticmethod
    def _format_bias_label(bias: str) -> str:
        """
        Traduit un biais de marche en libelle francais court.

        Args:
            bias (str): Biais brut (`BULLISH`, `BEARISH`, etc.).

        Returns:
            str: Libelle francais destine aux messages operateurs.
        """
        labels = {
            "BULLISH": "haussier",
            "BEARISH": "baissier",
            "RANGING": "range",
            "NEUTRAL": "neutre",
            "UNKNOWN": "indetermine",
        }
        return labels.get(str(bias or "").upper(), str(bias or "indetermine").lower())

    def _build_trade_reasoning(
        self,
        execution_comment: str,
        llm_reasoning: str,
        cortex_bias: str,
        gnn_bias: str,
    ) -> dict[str, str]:
        """Construit la logique d'execution et la synthese Telegram.

        Args:
            execution_comment (str): Commentaire produit par le moteur de trade.
            llm_reasoning (str): Synthese LLM optionnelle.
            cortex_bias (str): Biais cortex courant.
            gnn_bias (str): Biais GNN courant.

        Returns:
            dict[str, str]: Dictionnaire contenant :
                - ``logic``: Commentaire moteur compact.
                - ``summary``: Phrase de synthese en francais.
        """
        base_reason = (execution_comment or "").strip()
        fallback_summary = (
            "Le signal reste exploitable: "
            f"Cortex {self._format_bias_label(cortex_bias)}, "
            f"GNN {self._format_bias_label(gnn_bias)}, "
            "execution pilotee par la logique locale."
        )
        if self._is_unusable_reasoning(llm_reasoning):
            return {
                "logic": base_reason or f"Cortex={cortex_bias} | GNN={gnn_bias}",
                "summary": fallback_summary,
            }

        llm_clean = " ".join(str(llm_reasoning).split())
        return {
            "logic": base_reason or f"Cortex={cortex_bias} | GNN={gnn_bias}",
            "summary": llm_clean[:220] if llm_clean else fallback_summary,
        }

    @staticmethod
    def _format_horizon_code(horizon: str) -> str:
        """
        Retourne un code compact d'horizon pour le commentaire MT5.

        Args:
            horizon (str): Horizon live utilise par le Lab.

        Returns:
            str: Code court (`SCP`, `INT`, `SWG` ou `UNK`).
        """
        mapping = {
            "scalp": "SCP",
            "intraday": "INT",
            "swing": "SWG",
        }
        return mapping.get(str(horizon or "").lower(), "UNK")

    @staticmethod
    def _format_engine_code(engine_label: str) -> str:
        """
        Retourne un code compact de moteur pour le commentaire MT5.

        Args:
            engine_label (str): Nom complet du moteur d'inference.

        Returns:
            str: Code court (`MZ`, `DRV`, `RSI` ou `AI`).
        """
        lowered = str(engine_label or "").lower()
        if "muzero" in lowered:
            return "MZ"
        if "dreamer" in lowered:
            return "DRV"
        if "rsi" in lowered:
            return "RSI"
        return "AI"

    def _build_order_comment(
        self,
        engine_label: str,
        live_horizon: str,
        selection: str,
        model_value: float = 0.0,
    ) -> str:
        """
        Construit un commentaire MT5 compact et lisible.

        Args:
            engine_label (str): Nom du moteur de decision.
            live_horizon (str): Horizon applique a l'inference.
            selection (str): Type de selection live (`champion`, `fallback`, etc.).
            model_value (float): Score ou valeur renvoyee par le moteur.

        Returns:
            str: Commentaire court compatible MT5.
        """
        engine_code = self._format_engine_code(engine_label)
        horizon_code = self._format_horizon_code(live_horizon)
        selection_code = "CH" if str(selection or "").lower() in {"champion", "legacy_champion"} else "FB"
        value_code = f"{model_value:.1f}"
        return f"{engine_code}-{horizon_code}-{selection_code}-v{value_code}"[:31]

    def _is_symbol_entry_cooling_down(self, symbol: str) -> bool:
        """Indique si un symbole est encore en cooldown d'entree.

        Args:
            symbol (str): Symbole a verifier.

        Returns:
            bool: ``True`` si une entree recente interdit une nouvelle position.
        """
        last_entry_at = self._last_symbol_entry_at.get(symbol)
        if last_entry_at is None:
            return False
        return (datetime.now() - last_entry_at) < self._symbol_entry_cooldown

    @staticmethod
    def _get_symbol_pip_size(symbol: str) -> float:
        """
        Retourne la taille de pip de reference pour un symbole.

        Args:
            symbol (str): Symbole analyse.

        Returns:
            float: Taille de pip utilisee pour calibrer le Shepherd.
        """
        symbol_upper = symbol.upper()
        if "JPY" in symbol_upper and len("".join(char for char in symbol_upper if char.isalpha())) >= 6:
            return 0.01
        if "XAU" in symbol_upper or "XAG" in symbol_upper:
            return 0.1
        if any(token in symbol_upper for token in ["US30", "US100", "GER40", ".CASH"]):
            return 1.0
        if "BTC" in symbol_upper:
            return 10.0
        if "ETH" in symbol_upper:
            return 1.0
        return 0.0001

    def _get_shepherd_thresholds(
        self,
        symbol: str,
        open_price: float,
        current_sl: float,
        trade_skill: str,
    ) -> dict[str, float]:
        """
        Calibre les seuils de protection des positions ouvertes.

        Args:
            symbol (str): Symbole de la position.
            open_price (float): Prix d'ouverture.
            current_sl (float): Stop loss courant.
            trade_skill (str): Skill d'origine si connue.

        Returns:
            dict[str, float]: Seuils `be_threshold`, `trail_activation`,
                `trail_distance` et `stale_minutes`.
        """
        pip_size = self._get_symbol_pip_size(symbol)
        sl_distance = abs(open_price - current_sl) if current_sl > 0 else 0.0
        is_scalp = trade_skill.upper() == SkilledBehavior.SCALPING.name

        if is_scalp:
            be_threshold = max(pip_size * 3.0, sl_distance * 0.25)
            trail_activation = max(pip_size * 5.0, sl_distance * 0.40)
            trail_distance = max(pip_size * 2.0, sl_distance * 0.20)
            stale_minutes = float(self._scalp_stale_minutes)
        else:
            be_threshold = max(pip_size * 6.0, sl_distance * 0.35)
            trail_activation = max(pip_size * 10.0, sl_distance * 0.60)
            trail_distance = max(pip_size * 4.0, sl_distance * 0.25)
            stale_minutes = 0.0

        return {
            "be_threshold": be_threshold,
            "trail_activation": trail_activation,
            "trail_distance": trail_distance,
            "stale_minutes": stale_minutes,
        }

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TELEGRAM FORMATTERS (Sprint 9)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _fmt_open_msg(self, symbol: str, action: str, entry_price: float, sl_price: float,
                      rsi: float, atr: float, vwap: float, adx: float, cortex_bias: str, gnn_bias: str,
                      logic_comment: str, ai_summary: str, indicators: dict = None) -> str:
        """Formate un message d'ouverture Telegram lisible en ASCII."""
        sl_dist = abs(entry_price - sl_price)
        
        # Format Indicators (Safe Get)
        indicators = indicators or {}
        macd = indicators.get("MACD_Hist", 0.0)
        bb_pct = indicators.get("BB_Pct", 0.5)
        rvol = indicators.get("RVOL", 1.0)
        sup = indicators.get("sr_sup", 0.0)
        res = indicators.get("sr_res", 0.0)
        
        # Visual MACD
        macd_icon = "UP" if macd > 0 else "DOWN"
        
        return (
            f"*E.V.A | Nouvelle position (M1/M15)*\n"
            f"-------------------\n"
            f"Actif: {symbol}\n"
            f"Action: {action}\n"
            f"Entree: {entry_price:.5f}\n"
            f"SL: {sl_price:.5f} ({sl_dist:.2f} pts)\n\n"
            f"*Marches & signaux*\n"
            f"- RSI: {rsi:.1f} | ADX: {adx:.1f}\n"
            f"- VWAP: {vwap:.2f}\n"
            f"- MACD: {macd_icon} {macd:.4f}\n"
            f"- Vol relatif: {rvol:.1f}x\n"
            f"- Position BB: {bb_pct*100:.1f}%\n"
            f"- S/R: {sup:.2f} / {res:.2f}\n\n"
            f"*Analyse IA*\n"
            f"- Cortex: {cortex_bias}\n"
            f"- GNN (Proxmox): {gnn_bias}\n"
            f"- Synthese: {ai_summary}\n"
            f"- Logique: {logic_comment}\n\n"
            f"{datetime.now().strftime('%H:%M')} | The Hive"
        )

    def _fmt_close_msg(self, symbol: str, action: str, entry_price: float, exit_price: float,
                       profit: float, duration_min: int, reason: str = "SL/TP Hit") -> str:
        """Formate un message de fermeture Telegram lisible en ASCII."""
        pips = exit_price - entry_price
        if action == "SELL":
            pips = -pips
        # Normalize pips based on asset type
        pip_size = 0.1 if "XAU" in symbol else (1.0 if "US30" in symbol or "BTC" in symbol else 0.0001)
        pips_display = pips / pip_size
        
        emoji = "WIN" if profit >= 0 else "LOSS"
        pnl_sign = "+" if profit >= 0 else ""
        
        # Duration formatting
        if duration_min >= 60:
            dur_str = f"{duration_min // 60}h{duration_min % 60:02d}m"
        else:
            dur_str = f"{duration_min}min"
        
        return (
            f"*E.V.A | Trade ferme*\n"
            f"-------------------\n"
            f"Actif: {symbol}\n"
            f"Action: {action}\n"
            f"Resultat: {emoji} {pnl_sign}{pips_display:.1f} pips\n\n"
            f"*Financier*\n"
            f"- Entree: {entry_price:.5f}\n"
            f"- Sortie: {exit_price:.5f}\n"
            f"- P&L: {pnl_sign}${profit:.2f}\n"
            f"- Duree: {dur_str}\n\n"
            f"Raison: {reason}\n"
            f"{datetime.now().strftime('%H:%M')} | The Hive"
        )

    def _fmt_shepherd_msg(self, symbol: str, action: str, event: str, 
                          new_sl: float, profit_pips: float) -> str:
        """Formate un message Shepherd lisible en ASCII."""
        return (
            f"*E.V.A Shepherd | {event}*\n"
            f"-------------------\n"
            f"Actif: {symbol} {action}\n"
            f"Nouveau SL: {new_sl:.5f}\n"
            f"Profit protege: +{profit_pips:.1f} pips"
        )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # CLOSE DETECTION (Sprint 9)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    async def _detect_closed_positions(self, current_positions: list):
        """DÃ©tecte les positions fermÃ©es et envoie une notification."""
        if current_positions is None:
            # Glitch in MT5 retrieval, abort detection safely
            return
            
        current_tickets = {pos.ticket for pos in current_positions}
        
        # Find tickets that disappeared (= closed)
        closed_tickets = self._known_tickets - current_tickets
        
        for ticket in closed_tickets:
            info = self._trade_open_info.get(ticket, {})
            if not info:
                continue
            
            # Try to get the actual close info from MT5 deal history
            try:
                from_dt = info.get("open_time", datetime.now() - timedelta(days=1))
                to_dt = datetime.now() + timedelta(days=1) # Deal with server timezone ahead of local
                deals = await self.mt5.get_deal_history(from_dt, to_dt)
                
                # Find the closing deal for this position
                close_deal = None
                for deal in deals:
                    if deal.get("position_id") == ticket or deal.get("magic") == 12345:
                        if deal.get("symbol") == info.get("symbol"):
                            close_deal = deal
                            break
                
                if close_deal:
                    profit = close_deal["profit"] + close_deal.get("swap", 0) + close_deal.get("commission", 0)
                    exit_price = close_deal["price"]
                    duration = (close_deal["time"] - info["open_time"]).total_seconds() / 60
                    reason = close_deal.get("comment", "SL/TP Hit") or "SL/TP Hit"
                else:
                    # Fallback: no deal found, use stored info
                    profit = 0.0
                    exit_price = info.get("entry_price", 0.0)
                    duration = (datetime.now() - info["open_time"]).total_seconds() / 60
                    reason = "FermÃ© (dÃ©tails indisponibles)"
                
                msg = self._fmt_close_msg(
                    symbol=info["symbol"],
                    action=info["action"],
                    entry_price=info["entry_price"],
                    exit_price=exit_price,
                    profit=profit,
                    duration_min=int(duration),
                    reason=reason
                )
                self.telegram.send_sync(msg)
                logger.info(f"ðŸ“¤ Close notification sent for {info['symbol']} #{ticket} (P&L: ${profit:.2f})")
                
                # ðŸ§  FEEDBACK LOOP: Send real P&L to Lab for micro-training
                asyncio.create_task(self._send_pnl_feedback(
                    symbol=info["symbol"],
                    action=info["action"],
                    price=exit_price,
                    pnl=profit,
                    ticket=ticket,
                ))
                
                # ðŸ›¡ï¸ ANTI-TILT LOOP: Report losses to Nemesis for Self-Healing
                if profit < 0:
                    asyncio.create_task(get_nemesis_system().report_loss(
                        trade_id=str(ticket),
                        loss_amount=abs(profit),
                        market_context={"symbol": info["symbol"], "action": info["action"], "volatility": 0, "news_event": False, "trend_reversal": False}
                    ))
                
                # ðŸ’° ACCOUNTANT LOOP: Send financial event for Drawdown validation
                asyncio.create_task(self._send_pnl_to_accountant(
                    symbol=info["symbol"],
                    profit=profit
                ))

                # ðŸ“¸ VIRALIZATION LOOP: Send winning trade to Muse Media Factory (Port 8601)
                if profit >= 0.5:
                    asyncio.create_task(self._viralize_trade(
                        symbol=info["symbol"],
                        action=info["action"],
                        pnl=profit
                    ))
            except Exception as e:
                logger.error(f"Error processing closed ticket #{ticket}: {e}")
                # Cleanup if it failed mid-way
                self._trade_open_info.pop(ticket, None)
        
        # Update known tickets to current state
        self._known_tickets = current_tickets

    async def _sync_open_positions(self):
        """Peuple l'Ã©tat au dÃ©marrage avec les positions existantes sur MT5 (Sprint 12)."""
        logger.info("ðŸ”„ Syncing existing positions from MT5 state...")
        try:
            positions = await self.mt5.get_open_positions()
            if positions is not None:
                for pos in positions:
                    self._known_tickets.add(pos.ticket)
                    self._trade_open_info[pos.ticket] = {
                        "symbol": pos.symbol,
                        "action": pos.action.value if hasattr(pos.action, 'value') else str(pos.action),
                        "entry_price": float(pos.open_price),
                        "open_time": pos.open_time,
                    }
                logger.info(f"âœ… Synced {len(positions)} existing positions.")
        except Exception as e:
            logger.error(f"Failed to startup-sync positions: {e}")

    async def _viralize_trade(self, symbol: str, action: str, pnl: float):
        """Notifie l'agent The Muse pour gÃ©nÃ©rer une image virale d'un gain."""
        try:
            payload = {
                "symbol": symbol,
                "action": action,
                "pnl": pnl
            }
            # Muse run par dÃ©faut sur le port 9100 selon le docker-compose
            muse_url = f"http://{self.settings.api_host}:9100/viralize/trade"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(muse_url, json=payload, timeout=60) as resp:
                    if resp.status == 200:
                        logger.info(f"âœ¨ Trade Viralization Success for {symbol}")
                    else:
                        logger.warning(f"âš ï¸ Muse Viralization Failed: {resp.status} - {await resp.text()}")
        except Exception as e:
            logger.error(f"Error calling Muse for viralization: {e}")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # DAILY REPORT (Sprint 9)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    async def _half_day_report_loop(self):
        """Envoie un rapport rÃ©capitulatif toutes les demi-journÃ©es (midi et minuit)."""
        while self.is_active:
            try:
                now = datetime.now()
                # Determine next target: 11:55 or 23:55 (just before half-day ends)
                target1 = now.replace(hour=11, minute=55, second=0, microsecond=0)
                target2 = now.replace(hour=23, minute=55, second=0, microsecond=0)
                
                if now < target1:
                    next_report = target1
                elif now < target2:
                    next_report = target2
                else:
                    next_report = target1 + timedelta(days=1)
                
                wait_seconds = (next_report - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                
                if not self.is_active:
                    break
                
                await self._send_half_day_report()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Half-day report error: {e}")
                await asyncio.sleep(3600)

    async def _send_half_day_report(self):
        """GÃ©nÃ¨re et envoie le rapport de la demi-journÃ©e."""
        try:
            now = datetime.now()
            # Define period:
            if now.hour < 15:
                period_name = "MatinÃ©e"
                period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                period_name = "AprÃ¨s-Midi"
                period_start = now.replace(hour=12, minute=0, second=0, microsecond=0)
            
            period_end = now
            
            # Get deals from the period
            deals = await self.mt5.get_deal_history(period_start, period_end)
            
            # Get account info
            summary = await self.mt5.get_account_summary()
            
            total_trades = len(deals)
            wins = sum(1 for d in deals if d["profit"] > 0)
            losses = sum(1 for d in deals if d["profit"] < 0)
            total_pnl = sum(d["profit"] + d.get("swap", 0) + d.get("commission", 0) for d in deals)
            
            best_trade = max(deals, key=lambda d: d["profit"]) if deals and any(d["profit"] > 0 for d in deals) else None
            worst_trade = min(deals, key=lambda d: d["profit"]) if deals and any(d["profit"] < 0 for d in deals) else None
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            balance = summary.get("balance", 0)
            pnl_pct = (total_pnl / balance * 100) if balance > 0 else 0
            pnl_sign = "+" if total_pnl >= 0 else ""
            
            best_str = f"{best_trade['symbol']} +${best_trade['profit']:.2f}" if best_trade and best_trade["profit"] > 0 else "N/A"
            worst_str = f"{worst_trade['symbol']} ${worst_trade['profit']:.2f}" if worst_trade and worst_trade["profit"] < 0 else "N/A"
            
            # Additional Context for Report
            nemesis_str = "Actif" if self.risk._is_anti_tilt_active() else "Inactif"
            dd_pct = getattr(self.risk, "_get_daily_drawdown_percent", lambda: 0.0)()
            
            msg = (
                f"ðŸ“ˆ *E.V.A | Bilan {period_name}*\n"
                f"â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”\n"
                f"ðŸ“† Date: {now.strftime('%d/%m/%Y %H:%M')}\n\n"
                f"ðŸ“Š *Performances*\n"
                f"  â€¢ P&L: {pnl_sign}${total_pnl:.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                f"  â€¢ Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)\n"
                f"  â€¢ Balance: ${balance:,.2f}\n"
                f"  â€¢ Drawdown JournÃ©e: {dd_pct}%\n\n"
                f"ðŸ† *Top / Flop*\n"
                f"  â€¢ Best: {best_str}\n"
                f"  â€¢ Worst: {worst_str}\n\n"
                f"ðŸ›¡ï¸ *SÃ©curitÃ©*\n"
                f"  â€¢ Marge Libre: ${summary.get('margin_free', 0):,.2f}\n"
                f"  â€¢ Nemesis (Anti-Tilt): {nemesis_str}\n\n"
                f"ðŸ§  _The Hive continuously learning._"
            )
            
            self.telegram.send_sync(msg)
        except Exception as e:
            logger.error(f"Error generating half-day report: {e}")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # ACCOUNTANT & LAB INTEGRATION (REST API)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    async def _send_pnl_to_accountant(self, symbol: str, profit: float):
        """Envoie le rÃ©sultat financier Ã  l'Accountant (Port 8500) pour le suivi de la Drawdown"""
        try:
            import aiohttp
            import os
            # Use LAB_HOST as a fallback since Accountant runs on the same Proxmox server
            accountant_host = os.getenv("ACCOUNTANT_HOST", os.getenv("LAB_HOST", "localhost"))
            url = f"http://{accountant_host}:8500/pnl"
            
            # Re-fetch latest balance for true equity tracking
            summary = await self.mt5.get_account_summary()
            balance = float(summary.get("balance", 100000.0))
            equity = float(summary.get("equity", balance))
            
            payload = {
                "symbol": symbol,
                "profit_loss": profit,
                "balance": balance,
                "equity": equity
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5.0) as resp:
                     if resp.status == 200:
                         logger.debug(f"P&L of ${profit:.2f} properly accounted.")
                     else:
                         logger.warning(f"Accountant returned HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"Failed to reach Accountant: {e}")

    async def _send_pnl_feedback(self, symbol: str, action: str, price: float, pnl: float):
        """Envoie le P&L rÃ©el d'une transaction fermÃ©e au Lab pour Shadow Learning"""
        try:
            import aiohttp
            import os
            lab_host = os.getenv("LAB_HOST", "localhost")
            url = f"http://{lab_host}:8600/shadow/record"
            
            payload = {
                "symbol": symbol,
                "action": action,
                "price": price,
                "volume": 0.01,
                "pnl": pnl,
                "done": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5.0) as resp:
                     if resp.status == 200:
                         logger.debug(f"P&L feedback for {symbol} sent to Lab.")
                     else:
                         logger.warning(f"Lab returned HTTP {resp.status} for P&L feedback.")
        except Exception as e:
            logger.warning(f"Failed to send P&L feedback to Lab: {e}")
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # MAIN DRIFT LOOP
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    async def _drift_loop(self):
        """Boucle principale de drift (Multi-Asset)"""
        logger.info("ðŸŒŠ Entering Drift Loop (The Hive Mind)...")
        while self.is_active:
            try:
                nemesis = get_nemesis_system()

                # Toujours maintenir la gestion des positions avant de decider si
                # les nouvelles entrees doivent etre bloquees. Sinon un risque
                # ou Nemesis actif empeche aussi le Shepherd de securiser/cloturer.
                summary = await self.mt5.get_account_summary()
                if not summary:
                    self._log_pause_state(
                        "mt5_offline",
                        "Auto-Trading pause: MT5 hors ligne ou compte indisponible.",
                        cooldown_seconds=60,
                    )
                    await asyncio.sleep(30)
                    continue
                self._clear_pause_state("mt5_offline")
                balance = Decimal(str(summary.get("balance", 100000)))
                self.risk.update_account_balance(balance)

                await self.refresh_symbol_universe()

                positions = await self.mt5.get_open_positions()
                if positions is None:
                    self._log_pause_state(
                        "positions_unavailable",
                        "MT5 indisponible pour la lecture des positions. Nouvelle tentative dans 30s.",
                        cooldown_seconds=60,
                    )
                    await asyncio.sleep(30)
                    continue
                self._clear_pause_state("positions_unavailable")

                self.risk.update_positions_count(len(positions))
                open_symbols = {
                    position.symbol
                    for position in positions
                    if getattr(position, "symbol", None)
                }

                # CLOSE DETECTION (Sprint 9)
                await self._detect_closed_positions(positions)

                # THE SHEPHERD (MANAGEMENT)
                positions_changed = False
                for pos in positions:
                    try:
                        age_seconds = (datetime.now() - pos.open_time).total_seconds()
                        if age_seconds < self._shepherd_min_age_seconds:
                            continue

                        current_price = float(pos.current_price)
                        open_price = float(pos.open_price)
                        sl = float(pos.stop_loss) if pos.stop_loss else 0.0
                        trade_info = self._trade_open_info.get(pos.ticket, {})
                        trade_skill = str(trade_info.get("skill", "") or "")
                        thresholds = self._get_shepherd_thresholds(
                            symbol=pos.symbol,
                            open_price=open_price,
                            current_sl=sl,
                            trade_skill=trade_skill,
                        )
                        be_threshold = thresholds["be_threshold"]
                        trail_activation = thresholds["trail_activation"]
                        trail_distance = thresholds["trail_distance"]
                        stale_minutes = thresholds["stale_minutes"]
                        age_minutes = age_seconds / 60.0

                        if pos.action == TradeAction.BUY:
                            profit = current_price - open_price
                            if stale_minutes > 0 and age_minutes >= stale_minutes and profit <= be_threshold:
                                close_result = await self.mt5.close_position(pos.ticket)
                                if close_result.get("success"):
                                    positions_changed = True
                                    logger.info(
                                        "Shepherd: cloture stale BUY sur %s #%s apres %.1f min (profit=%.5f).",
                                        pos.symbol,
                                        pos.ticket,
                                        age_minutes,
                                        profit,
                                    )
                                else:
                                    logger.warning(
                                        "Shepherd: echec cloture stale BUY sur %s #%s: %s",
                                        pos.symbol,
                                        pos.ticket,
                                        close_result.get("message", "erreur inconnue"),
                                    )
                                continue
                            if profit > be_threshold and (sl == 0.0 or sl < open_price):
                                new_sl = open_price + max(self._get_symbol_pip_size(pos.symbol), trail_distance * 0.25)
                                modify_result = await self.mt5.modify_position(pos.ticket, sl=new_sl, tp=0.0)
                                if modify_result.get("success"):
                                    msg = self._fmt_shepherd_msg(pos.symbol, "BUY", "SECURED", new_sl, profit)
                                    logger.info(msg)
                                    self.telegram.send_sync(msg)
                                else:
                                    logger.warning(
                                        "Shepherd: echec de securisation BUY sur %s #%s: %s",
                                        pos.symbol,
                                        pos.ticket,
                                        modify_result.get("message", "erreur inconnue"),
                                    )
                            elif profit > trail_activation:
                                trailing_sl = current_price - trail_distance
                                if trailing_sl > sl:
                                    modify_result = await self.mt5.modify_position(pos.ticket, sl=trailing_sl, tp=0.0)
                                    if modify_result.get("success"):
                                        msg = self._fmt_shepherd_msg(pos.symbol, "BUY", "TRAILING", trailing_sl, profit)
                                        logger.info(msg)
                                    else:
                                        logger.warning(
                                            "Shepherd: echec de trailing BUY sur %s #%s: %s",
                                            pos.symbol,
                                            pos.ticket,
                                            modify_result.get("message", "erreur inconnue"),
                                        )

                        elif pos.action == TradeAction.SELL:
                            profit = open_price - current_price
                            if stale_minutes > 0 and age_minutes >= stale_minutes and profit <= be_threshold:
                                close_result = await self.mt5.close_position(pos.ticket)
                                if close_result.get("success"):
                                    positions_changed = True
                                    logger.info(
                                        "Shepherd: cloture stale SELL sur %s #%s apres %.1f min (profit=%.5f).",
                                        pos.symbol,
                                        pos.ticket,
                                        age_minutes,
                                        profit,
                                    )
                                else:
                                    logger.warning(
                                        "Shepherd: echec cloture stale SELL sur %s #%s: %s",
                                        pos.symbol,
                                        pos.ticket,
                                        close_result.get("message", "erreur inconnue"),
                                    )
                                continue
                            if profit > be_threshold and (sl == 0.0 or sl > open_price):
                                new_sl = open_price - max(self._get_symbol_pip_size(pos.symbol), trail_distance * 0.25)
                                modify_result = await self.mt5.modify_position(pos.ticket, sl=new_sl, tp=0.0)
                                if modify_result.get("success"):
                                    msg = self._fmt_shepherd_msg(pos.symbol, "SELL", "SECURED", new_sl, profit)
                                    logger.info(msg)
                                    self.telegram.send_sync(msg)
                                else:
                                    logger.warning(
                                        "Shepherd: echec de securisation SELL sur %s #%s: %s",
                                        pos.symbol,
                                        pos.ticket,
                                        modify_result.get("message", "erreur inconnue"),
                                    )
                            elif profit > trail_activation:
                                trailing_sl = current_price + trail_distance
                                if sl == 0.0 or trailing_sl < sl:
                                    modify_result = await self.mt5.modify_position(pos.ticket, sl=trailing_sl, tp=0.0)
                                    if modify_result.get("success"):
                                        msg = self._fmt_shepherd_msg(pos.symbol, "SELL", "TRAILING", trailing_sl, profit)
                                        logger.info(msg)
                                    else:
                                        logger.warning(
                                            "Shepherd: echec de trailing SELL sur %s #%s: %s",
                                            pos.symbol,
                                            pos.ticket,
                                            modify_result.get("message", "erreur inconnue"),
                                        )

                    except Exception as e_shepherd:
                        logger.error(f"Shepherd Error on {pos.ticket}: {e_shepherd}")

                if positions_changed:
                    refreshed_positions = await self.mt5.get_open_positions()
                    if refreshed_positions is not None:
                        positions = refreshed_positions
                        self.risk.update_positions_count(len(positions))
                        open_symbols = {
                            position.symbol
                            for position in positions
                            if getattr(position, "symbol", None)
                        }

                status = await self.risk.get_current_status()
                if not status.trading_allowed:
                    self._log_pause_state(
                        "risk_limits",
                        "Auto-Trading pause: limites de risque atteintes. Shepherd maintenu actif.",
                        cooldown_seconds=180,
                    )
                    await asyncio.sleep(60)
                    continue
                self._clear_pause_state("risk_limits")

                if nemesis.should_block_trading():
                    self._log_pause_state(
                        "nemesis_active",
                        "Auto-Trading pause: phase Nemesis active. Shepherd maintenu actif.",
                        cooldown_seconds=180,
                    )
                    await asyncio.sleep(60)
                    continue
                self._clear_pause_state("nemesis_active")

                if len(positions) >= self.risk.max_open_positions:
                    logger.info(
                        "Nombre maximal de positions atteint (%s). Attente du prochain cycle.",
                        self.risk.max_open_positions,
                    )
                    await asyncio.sleep(60)
                    continue

                symbols_to_scan = self.get_symbol_batch()
                if not symbols_to_scan:
                    logger.warning("Aucun symbole disponible pour le prochain cycle de scan.")
                    await asyncio.sleep(60)
                    continue

                await self.mt5.initialize_symbols(symbols_to_scan)

                # 3. Iterate over symbols (The Hive Mind)
                for symbol in symbols_to_scan:
                    if not self.is_active: break
                    
                    try:
                        if symbol in open_symbols:
                            logger.info(
                                "Signal ignore sur %s: position deja ouverte sur ce symbole.",
                                symbol,
                            )
                            continue

                        if self._is_symbol_entry_cooling_down(symbol):
                            logger.info(
                                "Signal ignore sur %s: cooldown d'entree encore actif.",
                                symbol,
                            )
                            continue

                        # NEW (Sprint 10) : Night Session Filter (Rollover Trap)
                        if not self.risk.is_within_trading_session(symbol):
                            continue
                            
                        # NEW: Localized News Tracking (Sprint 11 P3)
                        if getattr(self, "news", None) and self.news.should_block_trading(symbol):
                            continue
                            
                        # 3. Get Context from Cortex (Sprint 10)
                        last_strat = self.latest_decisions.get(symbol, {})
                        last_time = last_strat.get("timestamp")
                        
                        bias = "NEUTRAL"
                        gnn_bias = "N/A" # Default if not refreshed
                        should_refresh = not last_time or (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() > 900
                        
                        if should_refresh:
                            try:
                                strat_result = await self.cortex.analyze_market_context(symbol)
                                self.latest_decisions[symbol] = strat_result
                                last_strat = strat_result  # Keep memory pointer fresh for Telegram output
                                bias = strat_result.get("bias", "NEUTRAL")
                                
                                # --- NEW (Sprint 12): Signal Reversal Logic ---
                                # Close positions of opposite direction if bias is strong
                                asyncio.create_task(self._handle_reversal(symbol, bias))
                                
                                # 4. Neural Analysis (GNN / Dreamer)
                                gnn_bias = strat_result.get("gnn_bias", "UNKNOWN")
                            except Exception as e_cortex:
                                logger.error(f"ðŸ§  Cortex Error: {e_cortex}")
                        else:
                            bias = last_strat.get("bias", "NEUTRAL")
                            gnn_bias = last_strat.get("gnn_bias", "N/A")

                        # A. Market Data
                        tick = await self.mt5.get_symbol_tick(symbol)
                        if not tick or (isinstance(tick, dict) and "bid" not in tick):
                            continue
                            
                        current_price = tick['bid'] if isinstance(tick, dict) else tick.bid
                        
                        # B. AI Analysis (Dreamer V3)
                        candles = await self.mt5.get_recent_candles(symbol, count=100)
                        
                        # Indicators
                        from shared.indicators import IndicatorFactory
                        features = {}
                        rsi_val = 50.0
                        
                        if candles and len(candles) >= 50:
                            closes = [c["close"] for c in candles]
                            highs = [c["high"] for c in candles]
                            lows = [c["low"] for c in candles]
                            volumes = [c["tick_volume"] for c in candles]
                            
                            
                            rsi_val = IndicatorFactory.rsi(closes, 14).iloc[-1]
                            ema_200_val = IndicatorFactory.ema(closes, 200).iloc[-1]
                            macd_data = IndicatorFactory.macd(closes)
                            bb_data = IndicatorFactory.bollinger_bands(closes)
                            atr_val = IndicatorFactory.atr(highs, lows, closes, 14).iloc[-1]
                            fib_levels = IndicatorFactory.get_fibonacci_levels(highs, lows, 100)
                            rvol = IndicatorFactory.relative_volume(volumes, 20).iloc[-1]
                            cycles = IndicatorFactory.detect_cycles(closes)
                            
                            features = {
                                "RSI": rsi_val,
                                "MACD_Hist": macd_data["histogram"].iloc[-1],
                                "BB_Pct": bb_data["pct_b"].iloc[-1],
                                "ATR": atr_val,
                                "RVOL": rvol,
                                "Cycle_High": cycles["bars_since_high"],
                                "Cycle_Low": cycles["bars_since_low"],
                                "Fib_0": fib_levels.get("fib_0", 0.0),
                                "Fib_236": fib_levels.get("fib_236", 0.0),
                                "Fib_382": fib_levels.get("fib_382", 0.0),
                                "Fib_500": fib_levels.get("fib_500", 0.0),
                                "Fib_618": fib_levels.get("fib_618", 0.0),
                                "Fib_100": fib_levels.get("fib_100", 0.0)
                            }
                            
                            # ----- EXTENDED FEATURES (For future AI V4 training & Shadow Learning) -----
                            vwap_val = IndicatorFactory.vwap(highs, lows, closes, volumes).iloc[-1]
                            obv_val = IndicatorFactory.obv(closes, volumes).iloc[-1]
                            momentum_val = IndicatorFactory.momentum(closes, 10).iloc[-1]
                            trix_val = IndicatorFactory.trix(closes, 15).iloc[-1]
                            stoch_data = IndicatorFactory.stochastic(highs, lows, closes)
                            cci_val = IndicatorFactory.cci(highs, lows, closes).iloc[-1]
                            adx_data = IndicatorFactory.adx(highs, lows, closes)
                            ichimoku_data = IndicatorFactory.ichimoku(highs, lows, closes)
                            trendlines_val = IndicatorFactory.trendlines(closes).iloc[-1]
                            sr_data = IndicatorFactory.support_resistance(highs, lows, closes)
                            gann_data = IndicatorFactory.gann_angles(highs, lows, 100)

                            extended_features = {
                                "EMA_200": ema_200_val,
                                "vwap": vwap_val,
                                "obv": obv_val,
                                "momentum": momentum_val,
                                "trix": trix_val,
                                "stoch_k": stoch_data["percent_k"].iloc[-1],
                                "stoch_d": stoch_data["percent_d"].iloc[-1],
                                "cci": cci_val,
                                "adx": adx_data["adx"].iloc[-1],
                                "adx_plus_di": adx_data["plus_di"].iloc[-1],
                                "adx_minus_di": adx_data["minus_di"].iloc[-1],
                                "ichi_tenkan": ichimoku_data["tenkan_sen"].iloc[-1],
                                "ichi_kijun": ichimoku_data["kijun_sen"].iloc[-1],
                                "ichi_senkou_a": ichimoku_data["senkou_span_a"].iloc[-1],
                                "ichi_senkou_b": ichimoku_data["senkou_span_b"].iloc[-1],
                                "trendline_slope": trendlines_val,
                                "sr_res": sr_data["nearest_resistance"],
                                "sr_sup": sr_data["nearest_support"],
                                "fib_786": fib_levels.get("fib_786", 0.0),
                                "fib_ext_1618": fib_levels.get("fib_ext_1618", 0.0),
                                "fib_ext_2618": fib_levels.get("fib_ext_2618", 0.0),
                                "gann_1x1": gann_data["gann_1x1"],
                                "gann_1x2": gann_data["gann_1x2"],
                                "gann_2x1": gann_data["gann_2x1"],
                                "Return_1": ((closes[-1] - closes[-2]) / closes[-2]) if len(closes) > 1 and closes[-2] else 0.0,
                                "Spread_Norm": ((tick.get("ask", current_price) - tick.get("bid", current_price)) / current_price) if isinstance(tick, dict) and current_price else 0.0,
                            }
                                
                            # --- FORMATTED LOGGING ---
                            from colorama import Fore, Style
                            sym_color = Fore.CYAN if "XAU" in symbol else (Fore.YELLOW if "BTC" in symbol else Fore.WHITE)
                            bias_color = Fore.GREEN if bias == "BULLISH" else (Fore.RED if bias == "BEARISH" else Fore.LIGHTBLACK_EX)
                            
                            logger.info(
                                f"ðŸ§  {sym_color}{symbol:<8}{Style.RESET_ALL} | "
                                f"Price: {current_price:<9.2f} | "
                                f"RSI: {rsi_val:<4.1f} | "
                                f"ADX: {adx_data['adx'].iloc[-1]:<4.1f} | VWAP: {vwap_val:<9.2f} | "
                                f"Cortex: {bias_color}[{bias}]{Style.RESET_ALL}"
                            )
                        else:
                            features = {"RSI": 50.0}
                            extended_features = {}

                        # C. Dreamer Inference
                        merged_indicators = dict(features)
                        merged_indicators.update({
                            "VWAP": extended_features.get("vwap", 0.0),
                            "OBV": extended_features.get("obv", 0.0),
                            "Momentum": extended_features.get("momentum", 0.0),
                            "TRIX": extended_features.get("trix", 0.0),
                            "Stoch_K": extended_features.get("stoch_k", 50.0),
                            "Stoch_D": extended_features.get("stoch_d", 50.0),
                            "CCI": extended_features.get("cci", 0.0),
                            "ADX": extended_features.get("adx", 0.0),
                            "ADX_Plus_DI": extended_features.get("adx_plus_di", 0.0),
                            "ADX_Minus_DI": extended_features.get("adx_minus_di", 0.0),
                            "Ichi_Tenkan": extended_features.get("ichi_tenkan", current_price),
                            "Ichi_Kijun": extended_features.get("ichi_kijun", current_price),
                            "Ichi_Senkou_A": extended_features.get("ichi_senkou_a", current_price),
                            "Ichi_Senkou_B": extended_features.get("ichi_senkou_b", current_price),
                            "EMA_200": extended_features.get("EMA_200", current_price),
                            "Return_1": extended_features.get("Return_1", 0.0),
                            "Spread_Norm": extended_features.get("Spread_Norm", 0.0),
                        })
                        latest_candle = candles[-1] if candles else {}
                        skill = self.manager.plan_strategy(
                            {
                                "price": float(current_price),
                                "indicators": {"RSI": float(features.get("RSI", 50.0) or 50.0)},
                            }
                        )
                        live_horizon = self._resolve_inference_horizon(skill)
                        observation = {
                            "symbol": symbol,
                            "horizon": live_horizon,
                            "price": float(current_price),
                            "timestamp": latest_candle.get("time"),
                            "latest_candle": {
                                "open": float(latest_candle.get("open", current_price) or current_price),
                                "high": float(latest_candle.get("high", current_price) or current_price),
                                "low": float(latest_candle.get("low", current_price) or current_price),
                                "close": float(latest_candle.get("close", current_price) or current_price),
                                "tick_volume": float(latest_candle.get("tick_volume", 0.0) or 0.0),
                                "spread": float((tick.get("ask", current_price) - tick.get("bid", current_price)) if isinstance(tick, dict) else 0.0),
                            },
                            "indicators": merged_indicators
                        }
                        observation = self._json_safe_value(observation)
                        
                        action = None
                        comment = "Hold"
                        lab_result: dict[str, object] = {}
                        lab_selection = "none"
                        lab_selection_policy = "unknown"
                        live_model_allowed = not self._require_valid_champion
                        live_block_reason = "aucun"
                        mt5_order_comment = "EVA"
                        
                        try:
                            from shared.internal_auth import InternalAuth
                            lab_host = os.getenv("LAB_HOST", "localhost")
                            lab_url = f"http://{lab_host}:8600/dreamer/predict"
                            token = InternalAuth.generate_token("banker")
                            
                            async with aiohttp.ClientSession() as session:
                                async with session.post(lab_url, json=observation, headers={"X-Hive-Internal-Token": token}, timeout=5.0) as resp:
                                    if resp.status == 200:
                                        lab_result = await resp.json()
                                        mz_action = lab_result.get("action", 0)
                                        mz_value = lab_result.get("value", 0.0)
                                        model_engine = str(lab_result.get("engine", "Modele")).strip() or "Modele"
                                        lab_selection = str(lab_result.get("selection") or "none")
                                        lab_selection_policy = str(
                                            lab_result.get("selection_policy") or "unknown"
                                        )
                                        live_model_allowed, live_block_reason = self._is_live_model_allowed(lab_result)
                                        dreamer_comment = f"{model_engine} (v={mz_value:.2f})"
                                        mt5_order_comment = self._build_order_comment(
                                            engine_label=model_engine,
                                            live_horizon=live_horizon,
                                            selection=lab_selection,
                                            model_value=float(mz_value or 0.0),
                                        )

                                        if mz_action == 1:
                                            action = TradeAction.BUY
                                            comment = f"{dreamer_comment} -> BUY"
                                        elif mz_action == 2:
                                            action = TradeAction.SELL
                                            comment = f"{dreamer_comment} -> SELL"

                                        if not live_model_allowed:
                                            comment = f"Champion requis ({lab_selection})"
                                        if action is not None and not live_model_allowed:
                                            logger.info(
                                                "Entree live refusee sur %s: champion requis (%s / %s).",
                                                symbol,
                                                lab_selection,
                                                live_block_reason,
                                            )
                                            action = None
                                    else:
                                        logger.error(f"Dreamer Inference failed: HTTP {resp.status}")
                                        action = None
                                        comment = f"Lab error (HTTP {resp.status})"
                                        
                        except Exception as e_lab:
                            # En cas d'echec reseau ou de serialisation, on force HOLD pour proteger le compte.
                            logger.error(
                                "Inference Dreamer impossible pour %s: %s - %s. Passage en attente.",
                                symbol,
                                e_lab.__class__.__name__,
                                e_lab,
                            )
                            action = None
                            comment = "Erreur liaison Lab"

                        if action == TradeAction.BUY and bias == "BEARISH":
                            logger.info(f"ðŸ™… Cortex VETO: Blocking BUY on {symbol} (Trend is BEARISH on M15)")
                            action = None
                            comment = "Blocked by Cortex (Bearish Trend on M15)"
                        elif action == TradeAction.SELL and bias == "BULLISH":
                            logger.info(f"ðŸ™… Cortex VETO: Blocking SELL on {symbol} (Trend is BULLISH on M15)")
                            action = None
                            comment = "Blocked by Cortex (Bullish Trend on M15)"

                        # FORCE LOGGING for user visibility (via rich)
                        try:
                            from rich.console import Console
                            console = Console()
                            
                            act_color = "bold green" if action == TradeAction.BUY else ("bold red" if action == TradeAction.SELL else "bold yellow")
                            act_str = action.name if action else "HOLD"
                            bias_color = "bold green" if bias == "BULLISH" else ("bold red" if bias == "BEARISH" else "bold magenta")
                            sym_color = "bold cyan" if "XAU" in symbol else ("bold yellow" if "BTC" in symbol else "bold white")
                            
                            console.print(
                                f"[white]M1/M15[/white] [{sym_color}]{symbol}[/{sym_color}] | "
                                f"Price: [italic]{current_price:.2f}[/italic] | "
                                f"RSI: [magenta]{rsi_val:.1f}[/magenta] âž” "
                                f"Action: [{act_color}]{act_str}[/{act_color}] ({comment}) | "
                                f"Context: [{bias_color}]{bias}[/{bias_color}]"
                            )
                        except ImportError:
                            log_msg = f"[M1/M15] {symbol}: Price={current_price:.2f} RSI={rsi_val:.1f} -> Action={action} ({comment}) [Context: {bias}]"
                            logger.info(f"ðŸ§  {log_msg}")

                        # PUBLISH TO AGENT FEED (UI)
                        redis = get_redis_client()
                        await redis.publish("eva.banker.feed", {
                            "id": str(uuid.uuid4()),
                            "source_agent": "Banker",
                            "action": f"Analyse {symbol}: Price={current_price:.2f} | RSI={rsi_val:.1f} -> {action or 'Hold'} ({comment})",
                            "timestamp": datetime.now().isoformat(),
                            "type": "request" if action is None else "event"
                        })

                        # Store Decision State
                        decision_state = self.latest_decisions.get(symbol, {})
                        decision_state.update({
                            "price": float(current_price),
                            "rsi": rsi_val,
                            "macd": features.get("MACD_Hist", 0.0),
                            "vwap": float(vwap_val),
                            "adx": float(adx_data["adx"].iloc[-1]),
                            "live_horizon": live_horizon,
                            "action": action.value if action else "WAIT",
                            "comment": comment,
                            "timestamp": datetime.now().isoformat(),
                            "lab_selection": lab_selection,
                            "lab_selection_policy": lab_selection_policy,
                            "live_model_allowed": live_model_allowed,
                            "live_block_reason": live_block_reason,
                            "mt5_comment": mt5_order_comment,
                        })
                        if self._is_unusable_reasoning(str(decision_state.get("raw_thought", ""))):
                            decision_state["raw_thought"] = comment
                        self.latest_decisions[symbol] = decision_state

                        if action is None:
                            continue

                        # D. Execution
                        atr = features.get("ATR", 0.0)
                        is_high_vol = "XAU" in symbol or "BTC" in symbol or "US30" in symbol
                        
                        if atr > 0:
                            # Multiply ATR to give the algorithm breathing room.
                            # Usually SL = ATR * 2.5 is minimum for trend following.
                            sl_dist = Decimal(str(atr * 3.0)) 
                            tp_dist = Decimal("0.0") # Let profits run (Shepherd Mode)
                        else:
                            # Realistic baseline stop loss distances if ATR fails
                            # e.g Gold ($10), Indices ($30), Forex (20 pips)
                            if "XAU" in symbol: sl_dist = Decimal("8.0")
                            elif "US30" in symbol or "BTC" in symbol: sl_dist = Decimal("30.0")
                            else: sl_dist = Decimal("0.0020")
                            tp_dist = Decimal("0.0") # Let profits run
                            
                        entry_price = Decimal(str(current_price))
                        sl_price = entry_price - sl_dist if action == TradeAction.BUY else entry_price + sl_dist
                        tp_price = Decimal("0.0") # No TP
                        
                        # Dynamic Volume Calculation (Sprint 10)
                        balance = self.risk._account_balance
                        risk_pct = self.risk.max_risk_per_trade
                        
                        dynamic_vol = self.risk.calculate_lot_size(
                            balance=balance,
                            risk_percent=risk_pct,
                            sl_distance=sl_dist,
                            symbol=symbol
                        )
                        volume_constraints = await self.mt5.get_symbol_volume_constraints(symbol)
                        broker_min_volume = float(volume_constraints.get("min", Decimal("0.01")))

                        if dynamic_vol <= 0:
                            logger.info(
                                "Execution ignoree sur %s: volume calcule nul pour le budget risque courant.",
                                symbol,
                            )
                            if self._should_send_veto_alert("volume_zero"):
                                self.telegram.send_sync(
                                    f"*RISK VETO* | {symbol} {action.value if hasattr(action, 'value') else action} bloque\n"
                                    "Raison: risque autorise insuffisant pour calculer un volume exploitable."
                                )
                            action = None
                            comment = "Risque insuffisant"

                        if action and dynamic_vol < broker_min_volume:
                            logger.warning(
                                "Execution ignoree sur %s: volume calcule %.4f inferieur au minimum broker %.4f.",
                                symbol,
                                dynamic_vol,
                                broker_min_volume,
                            )
                            if self._should_send_veto_alert("volume_min"):
                                self.telegram.send_sync(
                                    f"*BROKER MIN VETO* | {symbol} {action.value if hasattr(action, 'value') else action} bloque\n"
                                    f"Volume risque: {dynamic_vol:.4f} | Min broker: {broker_min_volume:.4f}"
                                )
                            action = None
                            comment = "Volume minimum broker > risque autorise"

                        if action is None:
                            continue

                        # Safety Caps
                        final_vol = min(0.10, dynamic_vol)

                        order = TradeOrder(
                            symbol=symbol,
                            action=action,
                            volume=Decimal(str(final_vol)),
                            entry_price=entry_price,
                            stop_loss_price=sl_price,
                            take_profit_price=tp_price,
                            comment=mt5_order_comment
                        )
                        if action:
                            current_positions = await self.mt5.get_open_positions()
                            if current_positions and any(
                                getattr(position, "symbol", None) == symbol
                                for position in current_positions
                            ):
                                logger.info(
                                    "Execution ignoree sur %s: une position existe deja juste avant l'envoi.",
                                    symbol,
                                )
                                open_symbols.add(symbol)
                                action = None
                                comment = "Position deja ouverte"

                        if action:
                            # --- NEW (Sprint 13): Margin Pre-check ---
                            margin_required = await self.mt5.get_margin_required(symbol, action, final_vol) # Use final_vol here
                            account = await self.mt5.get_account_summary()
                            
                            if margin_required is not None and account:
                                free_margin = float(account.get("free_margin", 0.0))
                                if free_margin < margin_required:
                                    logger.warning(
                                        "Veto marge pour %s %s: requis=%.2f libre=%.2f",
                                        symbol,
                                        action.value if hasattr(action, "value") else action,
                                        margin_required,
                                        free_margin,
                                    )
                                    if self._should_send_veto_alert("margin"):
                                        self.telegram.send_sync(
                                            f"*MARGIN VETO* | {symbol} {action.value if hasattr(action, 'value') else action} bloque\n"
                                            f"Requis: ${margin_required:.2f} | Libre: ${free_margin:.2f}"
                                        )
                                    action = None
                                    comment = "Marge insuffisante"
                            
                        # If action was vetoed by margin check, skip execution
                        if action is None:
                            continue

                        validation = await self.risk.validate_order(order)
                        if validation["allowed"]:
                            logger.info(f"ðŸ¤– EXEC {symbol}: {action} | {comment}")
                            result = await self.worker.execute_skill(skill, order)
                            if result.get("success"):
                                # --- NEW (Sprint 11): LLM Micro-Reasoning ---
                                combined_indicators = {**features, **extended_features}
                                try:
                                    reasoning = await self.cortex.get_micro_reasoning(
                                        symbol=symbol, 
                                        action=action.value, 
                                        indicators=combined_indicators
                                    )
                                except Exception as exc_reasoning:
                                    logger.warning(
                                        "Synthese micro-raisonnement indisponible pour %s: %s",
                                        symbol,
                                        exc_reasoning,
                                    )
                                    reasoning = ""
                                telegram_reasoning = self._build_trade_reasoning(
                                    execution_comment=comment,
                                    llm_reasoning=reasoning,
                                    cortex_bias=last_strat.get("cortex_bias", last_strat.get("bias", "UNKNOWN")),
                                    gnn_bias=last_strat.get("gnn_bias", "UNKNOWN"),
                                )
                                
                                # Rich Telegram OPEN notification (Sprint 9/11)
                                open_msg = self._fmt_open_msg(
                                    symbol=symbol,
                                    action=action.value,
                                    entry_price=float(entry_price),
                                    sl_price=float(sl_price),
                                    rsi=rsi_val,
                                    atr=atr,
                                    vwap=extended_features.get("vwap", float(current_price)),
                                    adx=extended_features.get("adx", 0.0),
                                    cortex_bias=last_strat.get("cortex_bias", last_strat.get("bias", "UNKNOWN")),
                                    gnn_bias=last_strat.get("gnn_bias", "UNKNOWN"),
                                    logic_comment=telegram_reasoning["logic"],
                                    ai_summary=telegram_reasoning["summary"],
                                    indicators=combined_indicators
                                )
                                self.telegram.send_sync(open_msg)
                                
                                # Track this position for close detection
                                ticket = result.get("ticket", 0)
                                if ticket:
                                    now_open = datetime.now()
                                    self._trade_open_info[ticket] = {
                                        "symbol": symbol,
                                        "action": action.value,
                                        "entry_price": float(entry_price),
                                        "open_time": now_open,
                                        "skill": skill.value if hasattr(skill, "value") else str(skill),
                                    }
                                    self._known_tickets.add(ticket)
                                    self._last_symbol_entry_at[symbol] = now_open
                                    open_symbols.add(symbol)
                                
                                asyncio.create_task(self._record_learning_experience(order, result, observation))
                            else:
                                fail_msg = result.get("message", "Unknown error")
                                fail_code = result.get("retcode", "?")
                                fail_volume = result.get("normalized_volume", float(order.volume))
                                logger.error(
                                    "Ordre refuse pour %s %s: %s (retcode=%s)",
                                    symbol,
                                    action.value,
                                    fail_msg,
                                    fail_code,
                                )
                                self.telegram.send_sync(
                                    f"*ORDER FAILED* | {symbol} {action.value}\n"
                                    f"Raison: {fail_msg}\n"
                                    f"SL: {float(sl_price):.2f} | Vol envoye: {float(fail_volume):.2f}"
                                )
                        else:
                            logger.warning(f"Rejected {symbol}: {validation['reason']}")
                            
                        # Small delay between symbols
                        await asyncio.sleep(1.0)
                        
                    except Exception as e_sym:
                        logger.error(f"Error processing {symbol}: {e_sym}")
                        continue

                # 4. Wait (Drift Interval)
                await asyncio.sleep(self._drift_interval_seconds) 

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-Trading Loop Error: {e}")
                await asyncio.sleep(60)

    async def _record_learning_experience(self, order: TradeOrder, result: dict, observation: dict):
        """Envoie une observation d'ouverture au Lab pour le Shadow Learning."""
        try:
            import aiohttp
            from shared.internal_auth import InternalAuth
            
            lab_host = os.getenv("LAB_HOST", "localhost")
            lab_url = f"http://{lab_host}:8600/shadow/record"
            
            safe_observation = self._json_safe_value(observation)
            entry_price = float(safe_observation.get("price", 0.0) or 0.0)
            ticket = result.get("ticket")
            episode_id = f"live:{ticket}" if ticket else f"live:{uuid.uuid4()}"
            
            payload = {
                "symbol": order.symbol,
                "action": order.action.name,
                "price": entry_price,
                "volume": float(order.volume),
                "pnl": 0.0,
                "indicators": safe_observation.get("indicators", {}),
                "observation": safe_observation,
                "next_observation": safe_observation,
                "metadata": {
                    "source": "banker_live",
                    "episode_id": episode_id,
                    "ticket": ticket,
                    "magic": order.magic_number,
                    "comment": order.comment or "",
                },
                "timestamp": datetime.now().isoformat(),
                "done": False,
            }
            
            token = InternalAuth.generate_token("banker")
            headers = {
                "X-Hive-Internal-Token": token
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(lab_url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        logger.info("Shadow Learning: ouverture enregistree dans Lab (ticket=%s)", ticket)
                    else:
                        logger.warning("Shadow Learning: echec enregistrement ouverture (%s)", resp.status)
                        
        except Exception as e:
            logger.error("Envoi Shadow Learning impossible: %s", e)

    async def _send_pnl_feedback(self, symbol: str, action: str, price: float, pnl: float, ticket: int | None = None):
        """Envoie le P&L reel d'une cloture au Lab pour alimenter le dataset."""
        try:
            from shared.internal_auth import InternalAuth
            
            lab_host = os.getenv("LAB_HOST", "localhost")
            lab_url = f"http://{lab_host}:8600/shadow/feedback"
            safe_price = float(price or 0.0)
            observation = {
                "price": safe_price,
                "indicators": {"price_norm": safe_price / 3000.0 if safe_price else 0.0},
            }
            
            payload = {
                "symbol": symbol,
                "action": action,
                "price": safe_price,
                "volume": 0.0,
                "pnl": pnl,
                "indicators": observation["indicators"],
                "observation": observation,
                "next_observation": observation,
                "metadata": {
                    "source": "banker_live_close",
                    "episode_id": f"live:{ticket}" if ticket else f"close:{symbol}:{int(datetime.now().timestamp())}",
                    "ticket": ticket,
                },
                "timestamp": datetime.now().isoformat(),
                "done": True,
            }
            
            token = InternalAuth.generate_token("banker")
            headers = {"X-Hive-Internal-Token": token}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(lab_url, json=payload, headers=headers, timeout=5.0) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(
                            "Shadow Feedback: cloture %s P&L=%.2f enregistree (wm_loss=%s)",
                            symbol,
                            pnl,
                            result.get("wm_loss", "?"),
                        )
                    else:
                        logger.warning("Shadow Feedback: echec HTTP %s", resp.status)
                        
        except Exception as e:
            logger.error("Envoi du feedback P&L impossible: %s", e)

    async def _handle_reversal(self, symbol: str, bias: str):
        """Ferme les positions opposÃ©es au nouveau biais (Sprint 12)."""
        if bias not in ["BULLISH", "BEARISH"]:
            return
            
        for ticket, info in list(self._trade_open_info.items()):
            if info["symbol"] == symbol:
                should_close = False
                if bias == "BEARISH" and info["action"] == "BUY":
                    should_close = True
                elif bias == "BULLISH" and info["action"] == "SELL":
                    should_close = True
                
                if should_close:
                    logger.warning(f"ðŸ”„ Reversal {symbol}: Closing opposite {info['action']} #{ticket}")
                    await self.mt5.close_position(ticket)
                    # Note: notification de fermeture sera envoyÃ©e par le loop principal au prochain cycle

