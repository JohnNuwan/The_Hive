"""
Cerveau de l'Expert Banker (The Brain).
Contient la logique dÃ©cisionnelle (Manager), l'exÃ©cution (Worker) et la boucle d'autonomie.
"""

import asyncio
import json
import logging
import math
from collections import Counter, deque
from decimal import Decimal
from uuid import UUID
from datetime import datetime, timedelta
import os
from pathlib import Path
import aiohttp
import random
import uuid

from shared.redis_client import get_redis_client

from shared import (
    ConnectorMode,
    ExecutionEventEnvelope,
    RuntimeMode,
    TradeAction,
    TradeOrder,
    TradingContextEnvelope,
    TradingDecisionEnvelope,
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
        self._review_task = None
        self._news_task = None
        self.symbols = list(dict.fromkeys(self.settings.banker_symbols))
        self.risk.register_symbol_universe({symbol: self.mt5.classify_symbol(symbol) or "unknown" for symbol in self.symbols})
        self.latest_decisions = {} # Stores latest analysis per symbol
        self._decision_audit_limit = max(50, self._env_int("BANKER_DECISION_AUDIT_LIMIT", 200))
        self._decision_audit = deque(maxlen=self._decision_audit_limit)
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
        self._live_inference_url = self._env_text("BANKER_LIVE_INFERENCE_URL", "")
        self._live_inference_timeout_seconds = max(
            1,
            self._env_int("BANKER_LIVE_INFERENCE_TIMEOUT_SECONDS", 5),
        )
        self._require_valid_champion = self._env_flag("BANKER_REQUIRE_VALID_CHAMPION", True)
        self._training_compat_mode = self._normalize_training_compat_mode(
            self._env_text("BANKER_TRAINING_COMPAT_MODE", "disabled")
        )
        self._cpu_live_mode = self._training_compat_mode == "cpu_live"
        self._runtime_profile = self._resolve_runtime_profile(
            self._env_text("BANKER_RUNTIME_PROFILE", "auto")
        )
        self._shadow_learning_mode = "shadow_only"
        self._intraday_retrain_allowed = False
        self._intraday_promotion_allowed = False
        self._cpu_live_symbols = self._parse_symbol_allowlist(
            self._env_text(
                "BANKER_CPU_LIVE_SYMBOLS",
                "EURUSD,GBPUSD,USDJPY,XAUUSD",
            )
        )
        self._cpu_live_max_volume = max(
            0.01,
            self._env_float("BANKER_CPU_LIVE_MAX_VOLUME", 0.10),
        )
        self._cpu_live_symbol_max_volumes = self._parse_symbol_float_mapping(
            self._env_text("BANKER_CPU_LIVE_SYMBOL_MAX_VOLUMES", "")
        )
        self._ensemble_enabled = self._env_flag("BANKER_ENSEMBLE_ENABLED", False)
        self._ensemble_min_edge = max(0.0, self._env_float("BANKER_ENSEMBLE_MIN_EDGE", 0.15))
        if self._cpu_live_mode:
            # Le mode de compatibilite garde un chemin live minimal et stable
            # pendant qu'un gros run GPU monopolise `vLLM` et les ressources Lab.
            self._live_inference_horizon = "scalp"
            self._lab_universe_horizon = "scalp"
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
        self._trading_review_dir = Path(
            os.getenv(
                "BANKER_TRADING_REVIEW_DIR",
                os.path.join(os.getcwd(), "data", "checkpoints", "trading_reviews"),
            )
        )
        self._latest_trading_review_path = self._trading_review_dir / "latest.json"
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
        self._lab_universe_family = "mixed"
        self._lab_universe_dataset_id = None
        self._lab_universe_feature_profile = None
        self._lab_universe_live_champion_id = None
        self._lab_universe_live_champion_id_muzero = None
        self._lab_universe_live_champion_id_dreamer = None
        self._lab_universe_top_symbols: list[str] = []
        self._lab_universe_top_symbols_by_engine: dict[str, list[str]] = {
            "muzero": [],
            "dreamer": [],
        }

        # Sprint 10: News Filter
        self.news = NewsFilterService(filter_minutes=30)

        if self._cpu_live_mode:
            logger.info(
                "Mode de compatibilite training actif: cpu_live (scalp uniquement, champion_only exige, Cortex non obligatoire)."
            )

    async def start(self):
        """Demarre le pilote automatique."""
        if self.is_active:
            return
        self.is_active = True
        await self.refresh_symbol_universe()
        await self._sync_open_positions()

        self._loop_task = asyncio.create_task(self._drift_loop())
        self._daily_report_task = asyncio.create_task(self._half_day_report_loop())
        self._review_task = asyncio.create_task(self._daily_review_loop())
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
        for task in [self._loop_task, self._daily_report_task, self._review_task, self._news_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._loop_task = None
        self._daily_report_task = None
        self._review_task = None
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
    def _env_float(name: str, default: float) -> float:
        """Lit un flottant depuis l'environnement.

        Args:
            name (str): Nom de la variable d'environnement.
            default (float): Valeur de repli si la variable est absente.

        Returns:
            float: Valeur numerique retenue.
        """
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
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

    @staticmethod
    def _normalize_training_compat_mode(raw_mode: str) -> str:
        """Normalise le mode de compatibilite training/live.

        Args:
            raw_mode (str): Valeur brute issue de l'environnement.

        Returns:
            str: Mode retenu (`disabled` ou `cpu_live`).
        """
        normalized = str(raw_mode or "").strip().lower()
        if normalized in {"", "off", "none", "false", "0"}:
            return "disabled"
        if normalized == "cpu_live":
            return normalized
        logger.warning(
            "Mode de compatibilite training inconnu (%s). Repli sur disabled.",
            raw_mode,
        )
        return "disabled"

    def _resolve_runtime_profile(self, raw_profile: str) -> str:
        """Determine le profil d'exploitation expose par le banker.

        Args:
            raw_profile (str): Valeur brute lue dans l'environnement.

        Returns:
            str: Profil courant (`day_live_full_stack` ou
            `night_research_training`).
        """
        normalized = str(raw_profile or "").strip().lower()
        if normalized == "day_live_full_stack":
            return "day_live_full_stack"
        if normalized == "night_research_training":
            return "night_research_training"
        if normalized not in {"", "auto"}:
            logger.warning(
                "Profil runtime banker inconnu (%s). Repli sur le mode derive.",
                raw_profile,
            )
        return "night_research_training" if self._cpu_live_mode else "day_live_full_stack"

    @staticmethod
    def _parse_symbol_allowlist(raw_value: str) -> list[str]:
        """Normalise une liste de symboles separes par des virgules.

        Args:
            raw_value (str): Valeur brute issue de l'environnement.

        Returns:
            list[str]: Liste dedoublonnee de symboles.
        """
        symbols: list[str] = []
        seen: set[str] = set()
        for chunk in str(raw_value or "").split(","):
            symbol = chunk.strip().upper()
            if not symbol or symbol in seen:
                continue
            symbols.append(symbol)
            seen.add(symbol)
        return symbols

    @staticmethod
    def _parse_symbol_float_mapping(raw_value: str) -> dict[str, float]:
        """Normalise un mapping `SYMBOLE=volume` issu de l'environnement.

        Args:
            raw_value (str): Valeur brute issue de l'environnement.

        Returns:
            dict[str, float]: Volumes max par symbole.
        """

        mapping: dict[str, float] = {}
        for chunk in str(raw_value or "").split(","):
            entry = chunk.strip()
            if not entry or "=" not in entry:
                continue
            symbol, raw_limit = entry.split("=", 1)
            normalized_symbol = symbol.strip().upper()
            if not normalized_symbol:
                continue
            try:
                limit = float(raw_limit.strip())
            except ValueError:
                logger.warning(
                    "Volume cpu_live ignore pour %s: valeur illisible (%s).",
                    normalized_symbol,
                    raw_limit,
                )
                continue
            if limit <= 0:
                logger.warning(
                    "Volume cpu_live ignore pour %s: valeur non positive (%s).",
                    normalized_symbol,
                    raw_limit,
                )
                continue
            mapping[normalized_symbol] = limit
        return mapping

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
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else 0.0
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Decimal):
            decimal_value = float(value)
            return decimal_value if math.isfinite(decimal_value) else 0.0
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

    @staticmethod
    def _normalize_live_timestamp(value) -> str | None:
        """Normalise l'horodatage envoye au service d'inference live.

        Le service FastAPI attend une chaine ISO. MT5 peut cependant fournir
        un entier Unix ou un objet datetime selon la source des bougies.

        Args:
            value: Horodatage brut issu de la derniere bougie.

        Returns:
            str | None: Horodatage ISO serialisable, ou ``None`` si absent.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                return None
            try:
                return datetime.utcfromtimestamp(float(value)).isoformat() + "Z"
            except Exception:
                return str(value)
        if hasattr(value, "isoformat") and callable(value.isoformat):
            try:
                return value.isoformat()
            except Exception:
                return str(value)
        return str(value)

    def _resolve_inference_horizon(self, skill: SkilledBehavior | None) -> str:
        """Choisit l'horizon live le plus adapte a la skill courante.

        Args:
            skill (SkilledBehavior | None): Skill decidee par le manager.

        Returns:
            str: Horizon MuZero a transmettre au Lab.
        """
        if self._cpu_live_mode:
            return "scalp"

        if self._live_inference_horizon != "auto":
            return self._live_inference_horizon

        if skill == SkilledBehavior.SCALPING:
            return "scalp"
        if skill in {SkilledBehavior.HEDGING, SkilledBehavior.ACCUMULATION}:
            return "swing"
        return "intraday"

    def _resolve_live_inference_url(self) -> str:
        """Construit l'URL d'inference live a appeler.

        Returns:
            str: URL HTTP complete de prediction live.
        """
        explicit_url = self._live_inference_url.strip()
        if explicit_url:
            normalized = explicit_url.rstrip("/")
            if (
                normalized.endswith("/predict/live")
                or normalized.endswith("/dreamer/predict")
                or normalized.endswith("/predict/ensemble")
            ):
                return normalized
            if self._ensemble_enabled:
                return f"{normalized}/predict/ensemble"
            if self._cpu_live_mode:
                return f"{normalized}/predict/live"
            return f"{normalized}/dreamer/predict"

        lab_host = self._env_text("LAB_HOST", "localhost")
        lab_port = self._env_int("LAB_PORT", 8600)
        if self._ensemble_enabled:
            return f"http://{lab_host}:{lab_port}/predict/ensemble"
        if self._cpu_live_mode:
            return f"http://{lab_host}:{lab_port}/predict/live"
        return f"http://{lab_host}:{lab_port}/dreamer/predict"

    def _resolve_legacy_inference_url(self) -> str:
        """Construit l'URL legacy d'inference MuZero/Dreamer.

        Returns:
            str: URL HTTP complete du chemin legacy.
        """
        lab_host = self._env_text("LAB_HOST", "localhost")
        lab_port = self._env_int("LAB_PORT", 8600)
        return f"http://{lab_host}:{lab_port}/dreamer/predict"

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

        if self._cpu_live_mode and self._cpu_live_symbols:
            allowed_cpu_symbols = set(self._cpu_live_symbols)
            restricted_cpu = [symbol for symbol in discovered if symbol.upper() in allowed_cpu_symbols]
            if restricted_cpu:
                logger.info(
                    "Mode cpu_live: univers restreint aux majeurs valides (%s/%s symboles).",
                    len(restricted_cpu),
                    len(discovered),
                )
                discovered = restricted_cpu
            else:
                logger.warning(
                    "Mode cpu_live: aucun symbole courant ne correspond a l'allowlist %s.",
                    self._cpu_live_symbols,
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
        horizon = "scalp" if self._cpu_live_mode else self._lab_universe_horizon
        url = (
            f"http://{lab_host}:{lab_port}/live/universe"
            f"?horizon={horizon}&engine=muzero"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5.0) as response:
                    if response.status != 200:
                        logger.warning(
                            "Univers EVA Lab indisponible (%s). HTTP %s.",
                            horizon,
                            response.status,
                        )
                        return self._lab_universe_symbols

                    payload = await response.json()
                    live_universe = payload.get("live_universe", {}) or {}
                    remote_runtime_profile = str(payload.get("runtime_profile") or "").strip().lower()
                    if remote_runtime_profile in {"day_live_full_stack", "night_research_training"}:
                        # Le Lab est la source de verite pour l'alternance jour/nuit.
                        self._runtime_profile = remote_runtime_profile
                    symbols = [
                        str(symbol).strip()
                        for symbol in (live_universe.get("symbols", []) or [])
                        if str(symbol).strip()
                    ]
                    self._lab_universe_symbols = list(dict.fromkeys(symbols))
                    self._lab_universe_source = str(live_universe.get("source") or "unknown")
                    self._lab_universe_family = str(payload.get("family") or "mixed")
                    self._lab_universe_dataset_id = str(payload.get("dataset_id") or "").strip() or None
                    self._lab_universe_feature_profile = (
                        str(payload.get("feature_profile") or "").strip() or None
                    )
                    self._lab_universe_live_champion_id = (
                        str(payload.get("live_champion_id") or "").strip() or None
                    )
                    self._lab_universe_live_champion_id_muzero = (
                        str(payload.get("live_champion_id_muzero") or "").strip() or None
                    )
                    self._lab_universe_live_champion_id_dreamer = (
                        str(payload.get("live_champion_id_dreamer") or "").strip() or None
                    )
                    top_symbols_by_engine = dict(payload.get("top_live_symbols_by_engine") or {})
                    self._lab_universe_top_symbols_by_engine = {
                        "muzero": list(dict.fromkeys(top_symbols_by_engine.get("muzero") or [])),
                        "dreamer": list(dict.fromkeys(top_symbols_by_engine.get("dreamer") or [])),
                    }
                    self._lab_universe_top_symbols = list(
                        dict.fromkeys(
                            (payload.get("top_live_symbols") or [])
                            or self._lab_universe_top_symbols_by_engine.get("muzero", [])
                            or self._lab_universe_symbols[:5]
                        )
                    )
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
                            horizon,
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
        cortex_status = self.cortex.get_runtime_status()
        return {
            "enabled": self._lab_universe_enabled,
            "horizon": "scalp" if self._cpu_live_mode else self._lab_universe_horizon,
            "source": self._lab_universe_source,
            "live_family": self._lab_universe_family,
            "live_champion_id": self._lab_universe_live_champion_id,
            "live_champion_id_muzero": self._lab_universe_live_champion_id_muzero,
            "live_champion_id_dreamer": self._lab_universe_live_champion_id_dreamer,
            "feature_profile": self._lab_universe_feature_profile,
            "dataset_id": self._lab_universe_dataset_id,
            "symbols_total": len(self._lab_universe_symbols),
            "live_top_symbols": list(self._lab_universe_top_symbols),
            "live_top_symbols_by_engine": {
                engine: list(symbols)
                for engine, symbols in self._lab_universe_top_symbols_by_engine.items()
            },
            "cpu_live_symbols": list(self._cpu_live_symbols),
            "gate_allowed": self._lab_universe_gate_allowed,
            "selection": self._lab_universe_selection,
            "gate_reason": self._lab_universe_gate_reason,
            "require_valid_champion": self._require_valid_champion,
            "live_entries_allowed": (not self._require_valid_champion) or self._lab_universe_gate_allowed,
            "training_compat_mode": self._training_compat_mode,
            "cpu_live_mode": self._cpu_live_mode,
            "ensemble_enabled": self._ensemble_enabled,
            "cortex_required": bool(cortex_status.get("required", not self._cpu_live_mode)),
            "cortex": cortex_status,
            "last_refresh": (
                self._lab_universe_last_refresh.isoformat()
                if self._lab_universe_last_refresh is not None
                else None
            ),
        }

    def get_runtime_mode_status(self) -> dict[str, object]:
        """Retourne le mode runtime actif du banker.

        Returns:
            dict[str, object]: Etat du chemin live local.
        """
        cortex_status = self.cortex.get_runtime_status()
        return {
            "runtime_mode": self._resolve_runtime_mode().value,
            "runtime_profile": self._runtime_profile,
            "shadow_learning_mode": self._shadow_learning_mode,
            "intraday_retrain_allowed": self._intraday_retrain_allowed,
            "intraday_promotion_allowed": self._intraday_promotion_allowed,
            "training_compat_mode": self._training_compat_mode,
            "cpu_live_mode": self._cpu_live_mode,
            "allowed_horizons": ["scalp"] if self._cpu_live_mode else ["auto"],
            "cortex_required": bool(cortex_status.get("required", not self._cpu_live_mode)),
            "cortex_mode": cortex_status.get("mode"),
            "cortex_backend": cortex_status.get("backend"),
            "cortex": cortex_status,
            "gnn_mode": "consultatif" if self._cpu_live_mode else "fusionne",
            "selection_policy_required": "champion_only" if self._cpu_live_mode else "default",
            "live_inference_url": self._resolve_live_inference_url(),
            "live_inference_timeout_seconds": self._live_inference_timeout_seconds,
            "cpu_live_symbols": list(self._cpu_live_symbols),
            "cpu_live_max_volume": self._cpu_live_max_volume,
            "cpu_live_symbol_max_volumes": dict(self._cpu_live_symbol_max_volumes),
            "ensemble_enabled": self._ensemble_enabled,
            "ensemble_mode": "vote_50_50" if self._ensemble_enabled else "muzero_only",
            "ensemble_min_edge": self._ensemble_min_edge,
        }

    def get_execution_mechanics_status(self) -> dict[str, object]:
        """Expose les regles d'execution qui encadrent le live local.

        Returns:
            dict[str, object]: Parametres de volume, stale close et cooldown.
        """
        max_open_positions = int(getattr(self.risk, "max_open_positions", 0) or 0)
        cortex_status = self.cortex.get_runtime_status()
        return {
            "max_open_positions": max_open_positions,
            "symbol_entry_cooldown_minutes": int(self._symbol_entry_cooldown.total_seconds() / 60),
            "scalp_stale_minutes": self._scalp_stale_minutes,
            "cpu_live_max_volume": self._cpu_live_max_volume,
            "cpu_live_symbols": list(self._cpu_live_symbols),
            "cpu_live_symbol_max_volumes": dict(self._cpu_live_symbol_max_volumes),
            "live_family": self._lab_universe_family,
            "live_champion_id": self._lab_universe_live_champion_id,
            "live_champion_id_muzero": self._lab_universe_live_champion_id_muzero,
            "live_champion_id_dreamer": self._lab_universe_live_champion_id_dreamer,
            "live_top_symbols": list(self._lab_universe_top_symbols),
            "live_top_symbols_by_engine": {
                engine: list(symbols)
                for engine, symbols in self._lab_universe_top_symbols_by_engine.items()
            },
            "live_inference_horizon": self._live_inference_horizon,
            "selection_policy_required": "champion_only" if self._cpu_live_mode else "default",
            "ensemble_enabled": self._ensemble_enabled,
            "ensemble_mode": "vote_50_50" if self._ensemble_enabled else "muzero_only",
            "ensemble_min_edge": self._ensemble_min_edge,
            "cortex": cortex_status,
        }

    def _resolve_runtime_mode(self) -> RuntimeMode:
        """Retourne le mode runtime canonique du banker.

        Returns:
            RuntimeMode: Mode de fonctionnement actuellement impose.
        """
        if not self.is_active:
            return RuntimeMode.MAINTENANCE
        if self._cpu_live_mode:
            return RuntimeMode.TRAINING_CPU_LIVE
        return RuntimeMode.DEMO_LIVE

    def _build_event_connectors(self, decision_state: dict[str, object] | None = None) -> dict[str, object]:
        """Construit un instantane minimal des connecteurs utiles au live.

        Args:
            decision_state (dict[str, object] | None): Etat de decision courant.

        Returns:
            dict[str, object]: Modes de connecteurs exposes pour EVA Core.
        """
        state = decision_state or {}
        live_model_allowed = bool(state.get("live_model_allowed", False))
        model_status = str(state.get("model_status") or "").lower()
        gnn_confidence = float(state.get("gnn_confidence") or 0.0)

        live_inference_mode = ConnectorMode.LIVE
        if model_status in {"blocked", "error", "unavailable"}:
            live_inference_mode = ConnectorMode.PAPER

        gnn_mode = ConnectorMode.DISABLED if gnn_confidence <= 0.0 else ConnectorMode.LIVE
        mt5_mode = ConnectorMode.LIVE if getattr(self.mt5, "is_connected", False) and not getattr(self.mt5, "mock_mode", False) else ConnectorMode.PAPER
        cortex_status = self.cortex.get_runtime_status()

        return {
            "mt5": {
                "mode": mt5_mode.value,
                "connected": bool(getattr(self.mt5, "is_connected", False)),
            },
            "live_inference": {
                "mode": live_inference_mode.value,
                "allowed": live_model_allowed,
                "url": self._resolve_live_inference_url(),
                "ensemble_mode": state.get("ensemble_mode"),
                "degraded_fallback_reason": state.get("degraded_fallback_reason"),
            },
            "gnn": {
                "mode": gnn_mode.value,
                "role": "consultatif" if self._cpu_live_mode else "fusionne",
            },
            "cortex": {
                "mode": str(cortex_status.get("mode") or "disabled"),
                "backend": str(cortex_status.get("backend") or "none"),
                "required": bool(cortex_status.get("required", False)),
                "consultative": bool(cortex_status.get("consultative", self._cpu_live_mode)),
            },
            "vllm": {
                "mode": ConnectorMode.DISABLED.value if self._cpu_live_mode else ConnectorMode.LIVE.value,
            },
        }

    async def _publish_envelope_snapshot(
        self,
        channel: str,
        cache_key: str,
        envelope,
        ttl_seconds: int = 900,
    ) -> None:
        """Publie et met en cache une enveloppe de facon defensive.

        Args:
            channel (str): Canal Pub/Sub Redis cible.
            cache_key (str): Cle de cache associee.
            envelope: Modele Pydantic a serialiser.
            ttl_seconds (int): Duree de vie du cache.
        """
        try:
            redis = get_redis_client()
            payload = self._json_safe_value(envelope.model_dump())
            await redis.cache_set(cache_key, payload, ttl_seconds=ttl_seconds)
            await redis.publish(channel, payload)
        except Exception as exc:
            logger.debug("Publication Redis ignoree sur %s: %s", channel, exc)

    async def _publish_trading_context_event(
        self,
        symbol: str,
        horizon: str,
        decision_state: dict[str, object],
    ) -> None:
        """Publie le contexte de marche interprete pour un symbole.

        Args:
            symbol (str): Symbole analyse.
            horizon (str): Horizon MuZero retenu.
            decision_state (dict[str, object]): Etat de decision courant.
        """
        envelope = TradingContextEnvelope(
            runtime_mode=self._resolve_runtime_mode(),
            symbol=symbol,
            horizon=horizon,
            market_state={
                "price": decision_state.get("price"),
                "rsi": decision_state.get("rsi"),
                "adx": decision_state.get("adx"),
                "vwap": decision_state.get("vwap"),
                "cortex_bias": decision_state.get("cortex_bias"),
                "gnn_bias": decision_state.get("gnn_bias"),
                "final_bias": decision_state.get("final_bias"),
                "bias_strength": decision_state.get("bias_strength"),
                "comment": decision_state.get("comment"),
            },
            connectors=self._build_event_connectors(decision_state),
            metadata={"selection": decision_state.get("selection"), "checkpoint": decision_state.get("checkpoint")},
        )
        await self._publish_envelope_snapshot(
            channel="eva.trading.context",
            cache_key=f"eva:state:trading:context:{symbol.upper()}",
            envelope=envelope,
        )

    async def _publish_trading_decision_event(
        self,
        symbol: str,
        horizon: str,
        decision_state: dict[str, object],
    ) -> None:
        """Publie la decision brute puis filtree du banker.

        Args:
            symbol (str): Symbole analyse.
            horizon (str): Horizon MuZero retenu.
            decision_state (dict[str, object]): Etat de decision courant.
        """
        envelope = TradingDecisionEnvelope(
            runtime_mode=self._resolve_runtime_mode(),
            symbol=symbol,
            horizon=horizon,
            raw_model_action=str(decision_state.get("raw_model_action") or "HOLD"),
            post_veto_action=str(decision_state.get("post_veto_action") or "HOLD"),
            selection=str(decision_state.get("selection") or "none"),
            checkpoint=str(decision_state.get("checkpoint") or "") or None,
            final_bias=str(decision_state.get("final_bias") or "NEUTRAL"),
            veto_reason=str(decision_state.get("veto_reason") or "") or None,
            engine=str(decision_state.get("engine_name") or "") or None,
            ensemble_mode=str(decision_state.get("ensemble_mode") or "") or None,
            degraded_fallback_reason=str(decision_state.get("degraded_fallback_reason") or "") or None,
            connectors=self._build_event_connectors(decision_state),
            payload=self._json_safe_value(decision_state),
            metadata={"symbol_family": self.mt5.classify_symbol(symbol) or "unknown"},
        )
        await self._publish_envelope_snapshot(
            channel="eva.trading.decision",
            cache_key=f"eva:state:trading:decision:{symbol.upper()}",
            envelope=envelope,
        )

    async def _publish_execution_event(
        self,
        symbol: str,
        action: str,
        stage: str,
        allowed: bool,
        reason: str | None = None,
        volume: float | None = None,
        spread_points: float | None = None,
        ticket: int | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Publie un evenement d'execution ou de refus.

        Args:
            symbol (str): Symbole concerne.
            action (str): Action demandee.
            stage (str): Etape du pipeline d'execution.
            allowed (bool): Indique si l'etape passe.
            reason (str | None): Motif principal.
            volume (float | None): Volume concerne.
            spread_points (float | None): Spread releve.
            ticket (int | None): Ticket retourne par MT5.
            payload (dict[str, object] | None): Metadonnees additionnelles.
        """
        envelope = ExecutionEventEnvelope(
            runtime_mode=self._resolve_runtime_mode(),
            symbol=symbol,
            action=action,
            stage=stage,
            allowed=allowed,
            reason=reason,
            volume=volume,
            spread_points=spread_points,
            ticket=ticket,
            payload=self._json_safe_value(payload or {}),
        )
        await self._publish_envelope_snapshot(
            channel="eva.trading.execution",
            cache_key=f"eva:state:trading:execution:{symbol.upper()}",
            envelope=envelope,
        )

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
        selection_policy = str(payload.get("selection_policy") or "").lower()
        if self._cpu_live_mode and selection_policy != "champion_only":
            return False, f"selection_policy_invalide:{selection_policy or 'unknown'}"

        model_status = str(payload.get("model_status") or "").lower()
        if model_status == "blocked":
            reason = str(payload.get("reason") or "modele_bloque")
            return False, reason

        selection = str(payload.get("selection") or self._lab_universe_selection or "none").lower()
        if selection in {"champion", "legacy_champion"}:
            return True, "champion_valide"

        if selection in {"ensemble_50_50", "degraded_muzero_only"}:
            governance = dict(payload.get("governance") or {})
            muzero_payload = dict(governance.get("muzero") or {})
            muzero_selection = str(muzero_payload.get("selection") or "").lower()
            muzero_status = str(muzero_payload.get("model_status") or "").lower()
            if muzero_selection in {"champion", "legacy_champion"} and muzero_status == "live":
                if selection == "ensemble_50_50":
                    return True, "ensemble_valide"
                return True, "degraded_muzero_valide"

        reason = str(payload.get("reason") or self._lab_universe_gate_reason or selection or "unknown")
        return False, reason

    @staticmethod
    def _trade_action_label(action: TradeAction | None) -> str:
        """Retourne un libelle stable pour une action de trading.

        Args:
            action (TradeAction | None): Action interne eventuelle.

        Returns:
            str: Libelle ``BUY``, ``SELL`` ou ``HOLD``.
        """
        if action is None:
            return "HOLD"
        return str(getattr(action, "value", action))

    @staticmethod
    def _model_action_label(action_id: object, fallback: object = None) -> str:
        """Normalise un identifiant d'action MuZero en libelle lisible.

        Args:
            action_id (object): Identifiant brut retourne par le modele.
            fallback (object): Libelle secondaire fourni par EVA Lab.

        Returns:
            str: Libelle d'action stable.
        """
        action_map = {
            0: "HOLD",
            1: "BUY",
            2: "SELL",
            3: "SPLIT",
            4: "CLOSE",
        }
        try:
            normalized_id = int(action_id)
        except (TypeError, ValueError):
            normalized_id = None

        if normalized_id in action_map:
            return action_map[normalized_id]

        candidate = str(fallback or "").strip().upper()
        return candidate or "UNKNOWN"

    def _apply_context_veto(
        self,
        action: TradeAction | None,
        decision_context: dict[str, object] | None,
    ) -> tuple[TradeAction | None, str | None]:
        """Applique un veto contextuel symetrique sur une action live.

        Le veto est reserve aux cas ou le biais oppose est directionnel,
        explicite et suffisamment fort. Un contexte ``NEUTRAL`` ou ``RANGING``
        ne bloque jamais une entree a lui seul.

        Args:
            action (TradeAction | None): Action brute issue du modele.
            decision_context (dict[str, object] | None): Contexte fusionne
                ``Cortex + GNN`` calcule sur le symbole.

        Returns:
            tuple[TradeAction | None, str | None]: Action retenue et motif de veto.
        """
        if action is None:
            return None, None

        final_bias = str((decision_context or {}).get("bias") or "NEUTRAL").upper()
        bias_strength = str((decision_context or {}).get("bias_strength") or "weak").lower()

        if final_bias not in {"BULLISH", "BEARISH"}:
            return action, None

        if bias_strength == "weak":
            return action, None

        if action == TradeAction.BUY and final_bias == "BEARISH":
            return None, "veto_biais_baissier_confirme"

        if action == TradeAction.SELL and final_bias == "BULLISH":
            return None, "veto_biais_haussier_confirme"

        return action, None

    def _record_decision_audit(self, audit_event: dict[str, object]) -> None:
        """Enregistre un evenement de decision dans la fenetre glissante.

        Args:
            audit_event (dict[str, object]): Evenement normalise a conserver.
        """
        self._decision_audit.append(self._json_safe_value(audit_event))

    def get_decision_audit_snapshot(self) -> dict[str, object]:
        """Construit un resume glissant de la chaine de decision live.

        Returns:
            dict[str, object]: Compteurs globaux et repartition par symbole.
        """
        raw_counts: Counter[str] = Counter()
        post_counts: Counter[str] = Counter()
        ensemble_modes: Counter[str] = Counter()
        degraded_fallbacks: Counter[str] = Counter()
        per_symbol: dict[str, dict[str, object]] = {}

        for event in list(self._decision_audit):
            symbol = str(event.get("symbol") or "UNKNOWN")
            raw_action = str(event.get("raw_model_action") or "HOLD")
            post_action = str(event.get("post_veto_action") or "HOLD")
            ensemble_mode = str(event.get("ensemble_mode") or "none")
            degraded_reason = str(event.get("degraded_fallback_reason") or "").strip()

            raw_counts[raw_action] += 1
            post_counts[post_action] += 1
            ensemble_modes[ensemble_mode] += 1
            if degraded_reason:
                degraded_fallbacks[degraded_reason] += 1

            symbol_state = per_symbol.setdefault(
                symbol,
                {
                    "events": 0,
                    "raw_counts": Counter(),
                    "post_counts": Counter(),
                    "ensemble_modes": Counter(),
                },
            )
            symbol_state["events"] = int(symbol_state["events"]) + 1
            symbol_state["raw_counts"][raw_action] += 1
            symbol_state["post_counts"][post_action] += 1
            symbol_state["ensemble_modes"][ensemble_mode] += 1

        formatted_symbols = {}
        for symbol, symbol_state in per_symbol.items():
            formatted_symbols[symbol] = {
                "events": int(symbol_state["events"]),
                "raw_counts": dict(symbol_state["raw_counts"]),
                "post_counts": dict(symbol_state["post_counts"]),
                "ensemble_modes": dict(symbol_state["ensemble_modes"]),
            }

        recent_events = list(self._decision_audit)[-10:]
        return {
            "window_size": len(self._decision_audit),
            "limit": self._decision_audit_limit,
            "raw_counts": dict(raw_counts),
            "post_counts": dict(post_counts),
            "ensemble_modes": dict(ensemble_modes),
            "degraded_fallbacks": dict(degraded_fallbacks),
            "ensemble_decision_stats": {
                "enabled": self._ensemble_enabled,
                "mode_counts": dict(ensemble_modes),
                "degraded_fallbacks": dict(degraded_fallbacks),
            },
            "symbols": formatted_symbols,
            "recent": recent_events,
        }

    def get_latest_trading_review(self) -> dict[str, object] | None:
        """Relit le dernier rapport journalier persiste.

        Returns:
            dict[str, object] | None: Rapport complet si present, sinon `None`.
        """
        if not self._latest_trading_review_path.exists():
            return None
        try:
            return json.loads(self._latest_trading_review_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Lecture du dernier rapport journalier impossible: %s", exc)
            return None

    def get_latest_trading_review_summary(self) -> dict[str, object]:
        """Expose un resume leger du dernier rapport journalier.

        Returns:
            dict[str, object]: Resume de disponibilite et principaux chiffres.
        """
        review = self.get_latest_trading_review()
        if review is None:
            return {
                "available": False,
                "path": str(self._latest_trading_review_path),
            }

        review_window = dict(review.get("window") or {})
        performance = dict(review.get("performance") or {})
        summary = dict(performance.get("summary") or {})
        diagnostics = list(review.get("diagnostics") or [])
        return {
            "available": True,
            "generated_at": review.get("generated_at"),
            "path": str((review.get("storage") or {}).get("latest_path") or self._latest_trading_review_path),
            "window": review_window,
            "realized_pnl": summary.get("net_profit", 0.0),
            "closed_trades": summary.get("closed_trades", 0),
            "diagnostics_count": len(diagnostics),
        }

    def _build_symbol_review(self, closed_deals: list[dict[str, object]]) -> list[dict[str, object]]:
        """Agrege la journee par symbole a partir des deals de sortie.

        Args:
            closed_deals (list[dict[str, object]]): Deals de sortie normalises.

        Returns:
            list[dict[str, object]]: Resume trie des symboles les plus actifs.
        """
        buckets: dict[str, dict[str, object]] = {}
        for deal in closed_deals:
            symbol = str(deal.get("symbol") or "INCONNU")
            net_profit = (
                float(deal.get("profit") or 0.0)
                + float(deal.get("swap") or 0.0)
                + float(deal.get("commission") or 0.0)
            )
            bucket = buckets.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "closed_deals": 0,
                    "wins": 0,
                    "losses": 0,
                    "net_profit": 0.0,
                },
            )
            bucket["closed_deals"] = int(bucket["closed_deals"]) + 1
            bucket["net_profit"] = float(bucket["net_profit"]) + net_profit
            if net_profit > 0:
                bucket["wins"] = int(bucket["wins"]) + 1
            elif net_profit < 0:
                bucket["losses"] = int(bucket["losses"]) + 1

        ranked: list[dict[str, object]] = []
        for bucket in buckets.values():
            closed_deals_count = int(bucket["closed_deals"])
            wins = int(bucket["wins"])
            ranked.append(
                {
                    "symbol": bucket["symbol"],
                    "closed_deals": closed_deals_count,
                    "wins": wins,
                    "losses": int(bucket["losses"]),
                    "win_rate": round((wins / closed_deals_count) * 100.0, 2) if closed_deals_count else 0.0,
                    "net_profit": round(float(bucket["net_profit"]), 2),
                }
            )
        ranked.sort(key=lambda item: (float(item["net_profit"]), int(item["closed_deals"])), reverse=True)
        return ranked

    def _build_close_reason_summary(self, closed_deals: list[dict[str, object]]) -> list[dict[str, object]]:
        """Resume les motifs de cloture observes sur la journee.

        Args:
            closed_deals (list[dict[str, object]]): Deals de sortie normalises.

        Returns:
            list[dict[str, object]]: Motifs de cloture tries par frequence.
        """
        reasons: Counter[str] = Counter()
        for deal in closed_deals:
            comment = str(deal.get("comment") or "Inconnu").strip() or "Inconnu"
            reasons[comment] += 1
        return [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common(10)
        ]

    def _build_symbol_risk_map(
        self,
        *,
        symbol_review: list[dict[str, object]],
        nemesis_status: dict[str, object],
    ) -> list[dict[str, object]]:
        """Construit une carte de risque exploitable symbole par symbole.

        Args:
            symbol_review (list[dict[str, object]]): Performance journaliere par symbole.
            nemesis_status (dict[str, object]): Etat courant de Nemesis.

        Returns:
            list[dict[str, object]]: Carte de risque triee par criticite.
        """
        recent_losses_by_symbol = {
            str(symbol).strip().upper(): dict(payload or {})
            for symbol, payload in dict(nemesis_status.get("recent_losses_by_symbol") or {}).items()
            if str(symbol).strip()
        }
        quarantined_symbols = {
            str(symbol).strip().upper()
            for symbol in list(nemesis_status.get("quarantined_symbols") or [])
            if str(symbol).strip()
        }
        quarantine_expires = {
            str(symbol).strip().upper(): value
            for symbol, value in dict(nemesis_status.get("quarantine_expires_at_by_symbol") or {}).items()
            if str(symbol).strip()
        }
        buckets: dict[str, dict[str, object]] = {
            str(item.get("symbol") or "").strip().upper(): dict(item)
            for item in symbol_review
            if str(item.get("symbol") or "").strip()
        }
        for symbol in set(buckets.keys()) | set(recent_losses_by_symbol.keys()) | quarantined_symbols:
            bucket = dict(buckets.get(symbol) or {})
            losses = dict(recent_losses_by_symbol.get(symbol) or {})
            closed_deals = int(bucket.get("closed_deals") or 0)
            net_profit = round(float(bucket.get("net_profit") or 0.0), 2)
            recent_losses_4h = int(losses.get("recent_losses_4h") or 0)
            recent_events_12h = int(losses.get("recent_events_12h") or 0)
            day_loss_percent = round(float(losses.get("day_loss_percent") or 0.0), 4)
            is_quarantined = symbol in quarantined_symbols
            risk_level = "normal"
            if is_quarantined:
                risk_level = "quarantaine"
            elif recent_losses_4h >= 2 or recent_events_12h >= 3 or day_loss_percent >= 0.60:
                risk_level = "alerte"
            elif net_profit < 0.0 or int(bucket.get("losses") or 0) > int(bucket.get("wins") or 0):
                risk_level = "surveillance"
            buckets[symbol] = {
                "symbol": symbol,
                "closed_deals": closed_deals,
                "wins": int(bucket.get("wins") or 0),
                "losses": int(bucket.get("losses") or 0),
                "win_rate": round(float(bucket.get("win_rate") or 0.0), 2),
                "net_profit": net_profit,
                "recent_losses_4h": recent_losses_4h,
                "recent_events_12h": recent_events_12h,
                "recent_losses_6h": int(losses.get("recent_losses_6h") or 0),
                "day_loss_amount": round(float(losses.get("day_loss_amount") or 0.0), 2),
                "day_loss_percent": day_loss_percent,
                "latest_nemesis_type": losses.get("latest_nemesis_type"),
                "quarantined": is_quarantined,
                "quarantine_expires_at": quarantine_expires.get(symbol),
                "risk_level": risk_level,
            }
        ordered = sorted(
            buckets.values(),
            key=lambda item: (
                {"quarantaine": 0, "alerte": 1, "surveillance": 2, "normal": 3}.get(
                    str(item.get("risk_level") or "normal"),
                    9,
                ),
                float(item.get("net_profit") or 0.0),
            ),
        )
        return ordered

    def _build_nemesis_learning(
        self,
        *,
        nemesis_status: dict[str, object],
    ) -> dict[str, object]:
        """Agrege les pertes labelisees pour l'apprentissage nocturne.

        Args:
            nemesis_status (dict[str, object]): Etat courant de Nemesis.

        Returns:
            dict[str, object]: Exemples recents et regroupements utiles.
        """
        recent_defeats = list(nemesis_status.get("recent_defeats") or [])
        by_type: Counter[str] = Counter()
        by_symbol: Counter[str] = Counter()
        labeled_examples: list[dict[str, object]] = []

        for defeat in recent_defeats:
            context = dict(defeat.get("context") or {})
            nemesis_type = str(defeat.get("nemesis_type") or "UNKNOWN").strip() or "UNKNOWN"
            symbol = str(defeat.get("symbol") or context.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
            by_type[nemesis_type] += 1
            by_symbol[symbol] += 1
            labeled_examples.append(
                {
                    "trade_id": defeat.get("trade_id"),
                    "timestamp": defeat.get("timestamp"),
                    "symbol": symbol,
                    "nemesis_type": nemesis_type,
                    "loss": round(float(defeat.get("loss") or 0.0), 2),
                    "action": context.get("action"),
                    "gnn_bias": context.get("gnn_bias"),
                    "final_bias": context.get("final_bias"),
                    "spread": context.get("spread"),
                    "veto_reason": context.get("veto_reason"),
                    "raw_model_action": context.get("raw_model_action"),
                    "close_reason": context.get("close_reason"),
                    "model_version": context.get("model_version"),
                }
            )

        return {
            "total_examples": len(labeled_examples),
            "by_nemesis_type": dict(by_type),
            "by_symbol": dict(by_symbol),
            "labeled_examples": labeled_examples,
        }

    def _build_runtime_fallbacks(
        self,
        *,
        runtime_status: dict[str, object],
        decision_audit: dict[str, object],
    ) -> dict[str, object]:
        """Resume les replis runtime observes pendant la session.

        Args:
            runtime_status (dict[str, object]): Etat runtime courant du banker.
            decision_audit (dict[str, object]): Audit glissant des decisions.

        Returns:
            dict[str, object]: Compteurs et contexte des fallbacks observes.
        """
        by_reason = {
            str(reason): int(count or 0)
            for reason, count in dict(decision_audit.get("degraded_fallbacks") or {}).items()
            if str(reason).strip()
        }
        fallback_events = sum(by_reason.values())
        window_size = max(0, int(decision_audit.get("window_size") or 0))
        fallback_rate = round((fallback_events / window_size) * 100.0, 2) if window_size else 0.0
        return {
            "fallback_events": fallback_events,
            "fallback_rate_percent": fallback_rate,
            "by_reason": by_reason,
            "cpu_live_mode": bool(runtime_status.get("cpu_live_mode")),
            "cortex_mode": runtime_status.get("cortex_mode"),
            "cortex_backend": runtime_status.get("cortex_backend"),
            "gnn_mode": runtime_status.get("gnn_mode"),
            "selection_policy_required": runtime_status.get("selection_policy_required"),
        }

    def _build_support_model_quality(
        self,
        *,
        runtime_status: dict[str, object],
        decision_audit: dict[str, object],
        runtime_fallbacks: dict[str, object],
    ) -> dict[str, object]:
        """Evalue la qualite operationnelle des modeles de support.

        Args:
            runtime_status (dict[str, object]): Etat runtime courant du banker.
            decision_audit (dict[str, object]): Audit glissant des decisions.
            runtime_fallbacks (dict[str, object]): Synthese des replis runtime.

        Returns:
            dict[str, object]: Vue compacte de la sante des modeles de support.
        """
        cortex_status = dict(runtime_status.get("cortex") or {})
        ensemble_modes = dict(decision_audit.get("ensemble_modes") or {})
        return {
            "cortex": {
                "mode": cortex_status.get("mode"),
                "backend": cortex_status.get("backend"),
                "required": bool(cortex_status.get("required", False)),
                "consultative": bool(cortex_status.get("consultative", runtime_status.get("cpu_live_mode", False))),
            },
            "gnn": {
                "mode": runtime_status.get("gnn_mode"),
                "window_events": int(decision_audit.get("window_size") or 0),
            },
            "vllm": {
                "mode": "disabled" if bool(runtime_status.get("cpu_live_mode")) else "live",
                "selection_policy_required": runtime_status.get("selection_policy_required"),
            },
            "ensemble_modes": ensemble_modes,
            "fallback_rate_percent": runtime_fallbacks.get("fallback_rate_percent", 0.0),
            "fallback_events": runtime_fallbacks.get("fallback_events", 0),
        }

    def _build_mutation_priors(
        self,
        *,
        diagnostics: list[dict[str, object]],
        symbol_risk_map: list[dict[str, object]],
        nemesis_learning: dict[str, object],
        runtime_fallbacks: dict[str, object],
    ) -> list[dict[str, object]]:
        """Convertit la revue du jour en priors de mutation pour la nuit.

        Args:
            diagnostics (list[dict[str, object]]): Diagnostics de la review.
            symbol_risk_map (list[dict[str, object]]): Carte de risque par symbole.
            nemesis_learning (dict[str, object]): Exemples Nemesis recents.
            runtime_fallbacks (dict[str, object]): Etat des replis runtime.

        Returns:
            list[dict[str, object]]: Priors structures pour les runs de nuit.
        """
        priors: list[dict[str, object]] = []
        diagnostic_codes = {
            str(item.get("code") or "").strip()
            for item in diagnostics
            if str(item.get("code") or "").strip()
        }
        symbol_index = {
            str(item.get("symbol") or "").strip().upper(): item
            for item in symbol_risk_map
            if str(item.get("symbol") or "").strip()
        }

        if "passivite_elevee" in diagnostic_codes:
            priors.append(
                {
                    "target": "muzero_mechanics",
                    "priority": "high",
                    "reason": "passivite_elevee",
                    "adjustments": [
                        "relacher_hold_thresholds",
                        "augmenter_split_reactivity",
                        "ameliorer_close_quality",
                    ],
                }
            )

        if "biais_directionnel" in diagnostic_codes:
            priors.append(
                {
                    "target": "muzero_directional_balance",
                    "priority": "high",
                    "reason": "biais_directionnel",
                    "adjustments": [
                        "durcir_directional_imbalance",
                        "augmenter_directional_penalty",
                        "rehausser_activity_penalty_si_entrees_insuffisantes",
                    ],
                }
            )

        gold_risk = dict(symbol_index.get("XAUUSD") or {})
        gold_nemesis = int((nemesis_learning.get("by_symbol") or {}).get("XAUUSD", 0) or 0)
        if "passivite_gold" in diagnostic_codes or gold_nemesis > 0 or gold_risk.get("risk_level") in {"alerte", "quarantaine"}:
            priors.append(
                {
                    "target": "gold_live_filters",
                    "priority": "medium",
                    "reason": "liquidity_trap_xauusd",
                    "adjustments": [
                        "verifier_quarantaine_nemesis_xauusd",
                        "revoir_entree_gold_et_veto_spread",
                        "mesurer_impact_hold_sur_xauusd",
                    ],
                }
            )

        if int(runtime_fallbacks.get("fallback_events") or 0) > 0:
            priors.append(
                {
                    "target": "support_runtime",
                    "priority": "medium",
                    "reason": "fallbacks_runtime",
                    "adjustments": [
                        "stabiliser_cortex_ollama_ou_vllm",
                        "reduire_les_replis_avant_promotion",
                        "controler_la_chaine_consultative_avant_runs_gpu_lourds",
                    ],
                }
            )

        alert_symbols = [
            item["symbol"]
            for item in symbol_risk_map
            if str(item.get("risk_level") or "") in {"alerte", "quarantaine"}
        ]
        if alert_symbols:
            priors.append(
                {
                    "target": "gnn_consultatif",
                    "priority": "medium",
                    "reason": "filtrage_contextuel_a_renforcer",
                    "symbols": alert_symbols,
                    "adjustments": [
                        "rafraichir_gnn_consultatif",
                        "mesurer_directional_precision_minimale",
                        "verifier_l_apport_reel_sur_les_vetos",
                    ],
                }
            )

        if not priors:
            priors.append(
                {
                    "target": "collecte_shadow",
                    "priority": "low",
                    "reason": "aucun_signal_critique",
                    "adjustments": [
                        "poursuivre_la_collecte_shadow",
                        "maintenir_mu_zero_prioritaire",
                    ],
                }
            )
        return priors

    def _build_review_diagnostics(
        self,
        *,
        performance: dict[str, object],
        decision_audit: dict[str, object],
        symbol_review: list[dict[str, object]],
        nemesis_status: dict[str, object],
    ) -> list[dict[str, object]]:
        """Derive un diagnostic exploitable pour les runs de nuit.

        Args:
            performance (dict[str, object]): Performance agregee de la journee.
            decision_audit (dict[str, object]): Audit glissant des decisions live.
            symbol_review (list[dict[str, object]]): Resume par symbole.
            nemesis_status (dict[str, object]): Etat courant de Nemesis.

        Returns:
            list[dict[str, object]]: Liste de constats priorises.
        """
        diagnostics: list[dict[str, object]] = []
        summary = dict(performance.get("summary") or {})
        raw_counts = Counter(dict(decision_audit.get("raw_counts") or {}))
        post_counts = Counter(dict(decision_audit.get("post_counts") or {}))
        degraded_fallbacks = dict(decision_audit.get("degraded_fallbacks") or {})
        closed_trades = int(summary.get("closed_trades") or 0)
        net_profit = float(summary.get("net_profit") or 0.0)

        if closed_trades == 0:
            diagnostics.append(
                {
                    "code": "aucun_trade_cloture",
                    "severity": "warning",
                    "message": "Aucun trade cloture sur la fenetre analysee; les comparaisons restent faibles.",
                }
            )
        elif net_profit <= 0.0:
            diagnostics.append(
                {
                    "code": "pnl_journalier_negatif",
                    "severity": "warning",
                    "message": "Le PnL realise de la journee est negatif ou nul.",
                }
            )

        post_total = sum(post_counts.values())
        if post_total > 0 and (post_counts.get("HOLD", 0) / post_total) >= 0.65:
            diagnostics.append(
                {
                    "code": "passivite_elevee",
                    "severity": "info",
                    "message": "Le taux de HOLD/VETO depasse 65%, ce qui signale une passivite notable.",
                }
            )

        directional_total = raw_counts.get("BUY", 0) + raw_counts.get("SELL", 0)
        if directional_total > 0:
            dominant_share = max(raw_counts.get("BUY", 0), raw_counts.get("SELL", 0)) / directional_total
            if dominant_share >= 0.75:
                diagnostics.append(
                    {
                        "code": "biais_directionnel",
                        "severity": "warning",
                        "message": "Les signaux directionnels sont trop concentres sur un seul sens.",
                    }
                )

        gold_state = next((item for item in symbol_review if str(item.get("symbol")).upper() == "XAUUSD"), None)
        if gold_state is None or int(gold_state.get("closed_deals") or 0) == 0:
            diagnostics.append(
                {
                    "code": "passivite_gold",
                    "severity": "info",
                    "message": "Aucune cloture Gold detectee sur la periode; le flux XAUUSD reste trop passif.",
                }
            )

        if degraded_fallbacks:
            diagnostics.append(
                {
                    "code": "fallbacks_runtime",
                    "severity": "warning",
                    "message": "Des replis runtime ont ete observes dans la chaine live.",
                    "details": degraded_fallbacks,
                }
            )

        if bool(nemesis_status.get("trading_blocked", False)):
            diagnostics.append(
                {
                    "code": "nemesis_actif",
                    "severity": "critical",
                    "message": "Nemesis bloque actuellement le trading; la revue de nuit doit integrer les defaites recentes.",
                }
            )

        return diagnostics

    def _build_review_recommendations(
        self,
        diagnostics: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Transforme les constats en recommandations de nuit.

        Args:
            diagnostics (list[dict[str, object]]): Constats produits par la revue.

        Returns:
            list[dict[str, object]]: Actions conseillees pour la fenetre de nuit.
        """
        recommendations: list[dict[str, object]] = []
        seen_codes: set[str] = set()
        for diagnostic in diagnostics:
            code = str(diagnostic.get("code") or "")
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            if code == "passivite_elevee":
                recommendations.append(
                    {
                        "action": "ajuster_mechanics_hold_split",
                        "message": "Verifier les seuils HOLD/SPLIT et reduire la passivite des veto intraday.",
                    }
                )
            elif code == "biais_directionnel":
                recommendations.append(
                    {
                        "action": "relancer_ga_mecanique",
                        "message": "Durcir l'equilibre directionnel dans la campagne GA seedee MuZero.",
                    }
                )
            elif code == "passivite_gold":
                recommendations.append(
                    {
                        "action": "revoir_xauusd",
                        "message": "Analyser les veto spread/contexte sur XAUUSD avant la prochaine campagne nocturne.",
                    }
                )
            elif code == "fallbacks_runtime":
                recommendations.append(
                    {
                        "action": "stabiliser_runtime",
                        "message": "Inspecter les connecteurs en repli avant d'autoriser des runs GPU plus lourds.",
                    }
                )
            elif code == "nemesis_actif":
                recommendations.append(
                    {
                        "action": "analyser_nemesis",
                        "message": "Passer en revue les patterns Nemesis pour eviter une repetition sur la prochaine session.",
                    }
                )
            elif code == "pnl_journalier_negatif":
                recommendations.append(
                    {
                        "action": "revoir_shortlist",
                        "message": "Ne promouvoir aucun candidat et renforcer la shortlist GA/full avant la prochaine bascule.",
                    }
                )
        if not recommendations:
            recommendations.append(
                {
                    "action": "continuer_collecte_shadow",
                    "message": "Aucun signal critique; poursuivre la collecte Shadow et la comparaison nocturne.",
                }
            )
        return recommendations

    async def generate_trading_review(
        self,
        *,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        persist: bool = True,
    ) -> dict[str, object]:
        """Genere la revue journaliere structuree du banker.

        Args:
            from_dt (datetime | None): Debut explicite de la fenetre.
            to_dt (datetime | None): Fin explicite de la fenetre.
            persist (bool): Si True, persiste le rapport sur disque.

        Returns:
            dict[str, object]: Rapport complet de revue.
        """
        review_end = to_dt or datetime.now()
        review_start = from_dt or review_end.replace(hour=0, minute=0, second=0, microsecond=0)
        account_summary = await self.mt5.get_account_summary()
        open_positions = await self.mt5.get_open_positions() or []
        closed_deals = await self.mt5.get_deal_history(review_start, review_end, closed_only=True)
        performance = await self.mt5.get_strategy_performance(review_start, review_end, limit=10)
        decision_audit = self.get_decision_audit_snapshot()
        nemesis_status = get_nemesis_system().get_status()
        runtime_status = self.get_runtime_mode_status()
        execution_mechanics = self.get_execution_mechanics_status()
        live_universe = self.get_live_universe_status()
        symbol_review = self._build_symbol_review(closed_deals)
        symbol_risk_map = self._build_symbol_risk_map(
            symbol_review=symbol_review,
            nemesis_status=nemesis_status,
        )
        close_reasons = self._build_close_reason_summary(closed_deals)
        diagnostics = self._build_review_diagnostics(
            performance=performance,
            decision_audit=decision_audit,
            symbol_review=symbol_review,
            nemesis_status=nemesis_status,
        )
        nemesis_learning = self._build_nemesis_learning(nemesis_status=nemesis_status)
        runtime_fallbacks = self._build_runtime_fallbacks(
            runtime_status=runtime_status,
            decision_audit=decision_audit,
        )
        support_model_quality = self._build_support_model_quality(
            runtime_status=runtime_status,
            decision_audit=decision_audit,
            runtime_fallbacks=runtime_fallbacks,
        )
        mutation_priors = self._build_mutation_priors(
            diagnostics=diagnostics,
            symbol_risk_map=symbol_risk_map,
            nemesis_learning=nemesis_learning,
            runtime_fallbacks=runtime_fallbacks,
        )
        recommendations = self._build_review_recommendations(diagnostics)

        review: dict[str, object] = {
            "generated_at": review_end.isoformat(),
            "runtime_profile": self._runtime_profile,
            "shadow_learning_mode": self._shadow_learning_mode,
            "window": {
                "from": review_start.isoformat(),
                "to": review_end.isoformat(),
            },
            "account": self._json_safe_value(account_summary or {}),
            "open_positions": self._json_safe_value(
                [
                    {
                        "ticket": position.ticket,
                        "symbol": position.symbol,
                        "action": position.action.value if hasattr(position.action, "value") else str(position.action),
                        "volume": float(position.volume),
                        "profit": float(position.profit),
                        "open_price": float(position.open_price),
                        "current_price": float(position.current_price),
                    }
                    for position in open_positions
                ]
            ),
            "performance": self._json_safe_value(performance),
            "symbols": symbol_review,
            "symbol_risk_map": self._json_safe_value(symbol_risk_map),
            "close_reasons": close_reasons,
            "decision_audit": self._json_safe_value(decision_audit),
            "latest_decisions": self._json_safe_value(self.latest_decisions),
            "nemesis": self._json_safe_value(nemesis_status),
            "nemesis_learning": self._json_safe_value(nemesis_learning),
            "runtime": self._json_safe_value(runtime_status),
            "runtime_fallbacks": self._json_safe_value(runtime_fallbacks),
            "support_model_quality": self._json_safe_value(support_model_quality),
            "execution_mechanics": self._json_safe_value(execution_mechanics),
            "live_universe": self._json_safe_value(live_universe),
            "intraday_mutation_allowed": self._intraday_retrain_allowed,
            "intraday_promotion_allowed": self._intraday_promotion_allowed,
            "diagnostics": diagnostics,
            "mutation_priors": self._json_safe_value(mutation_priors),
            "recommendations": recommendations,
        }

        if persist:
            self._trading_review_dir.mkdir(parents=True, exist_ok=True)
            timestamp = review_end.strftime("%Y%m%d_%H%M%S")
            archive_path = self._trading_review_dir / f"review_{timestamp}.json"
            review["storage"] = {
                "latest_path": str(self._latest_trading_review_path),
                "archive_path": str(archive_path),
            }
            payload = self._json_safe_value(review)
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            archive_path.write_text(serialized, encoding="utf-8")
            self._latest_trading_review_path.write_text(serialized, encoding="utf-8")
            logger.info("Revue journaliere ecrite dans %s.", archive_path)

        return review

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
        """Detecte les positions fermees et alimente le feedback live.

        Cette routine doit rester stricte sur l'identification du deal de
        cloture afin d'eviter de rattacher un mauvais PnL a un ticket encore
        vivant. Elle alimente ensuite le gouverneur de risque, Nemesis et le
        Shadow Learning a partir du contexte reel de la position.

        Args:
            current_positions (list): Positions encore ouvertes sur MT5.
        """
        if current_positions is None:
            return

        current_tickets = {pos.ticket for pos in current_positions}
        closed_tickets = self._known_tickets - current_tickets

        for ticket in closed_tickets:
            info = self._trade_open_info.get(ticket, {})
            if not info:
                continue

            try:
                symbol = str(info.get("symbol") or "").strip().upper()
                from_dt = info.get("open_time", datetime.now() - timedelta(days=1))
                to_dt = datetime.now() + timedelta(days=1)
                deals = await self.mt5.get_deal_history(from_dt, to_dt) or []

                close_deal = None
                for deal in deals:
                    if deal.get("position_id") != ticket:
                        continue
                    if str(deal.get("symbol") or "").strip().upper() != symbol:
                        continue
                    close_deal = deal
                    break

                close_time = datetime.now()
                if close_deal:
                    profit = (
                        float(close_deal.get("profit", 0.0) or 0.0)
                        + float(close_deal.get("swap", 0.0) or 0.0)
                        + float(close_deal.get("commission", 0.0) or 0.0)
                    )
                    exit_price = float(close_deal.get("price", info.get("entry_price", 0.0)) or 0.0)
                    close_time = close_deal.get("time") if isinstance(close_deal.get("time"), datetime) else datetime.now()
                    reason = str(close_deal.get("comment") or "SL/TP Hit")
                else:
                    profit = 0.0
                    exit_price = float(info.get("entry_price", 0.0) or 0.0)
                    reason = "Ferme (details indisponibles)"

                duration = max(
                    0.0,
                    (close_time - info.get("open_time", datetime.now())).total_seconds() / 60.0,
                )

                msg = self._fmt_close_msg(
                    symbol=symbol,
                    action=info["action"],
                    entry_price=info["entry_price"],
                    exit_price=exit_price,
                    profit=profit,
                    duration_min=int(duration),
                    reason=reason,
                )
                self.telegram.send_sync(msg)
                logger.info(
                    "Notification de cloture envoyee pour %s #%s (P&L: %.2f).",
                    symbol,
                    ticket,
                    profit,
                )

                self.risk.record_trade_result(Decimal(str(profit)))
                asyncio.create_task(self.risk.save_state())
                risk_status = await self.risk.get_current_status()

                asyncio.create_task(
                    self._send_pnl_feedback(
                        symbol=symbol,
                        action=info["action"],
                        price=exit_price,
                        pnl=profit,
                        ticket=ticket,
                    )
                )

                if profit < 0:
                    market_context = {
                        "symbol": symbol,
                        "action": info.get("action"),
                        "volatility": float(info.get("atr", 0.0) or 0.0),
                        "gnn_bias": info.get("gnn_bias"),
                        "final_bias": info.get("final_bias"),
                        "spread": float(info.get("spread", 0.0) or 0.0),
                        "veto_reason": info.get("veto_reason"),
                        "raw_model_action": info.get("raw_model_action"),
                        "close_reason": reason,
                        "model_version": info.get("model_version"),
                        "engine_name": info.get("engine_name"),
                        "selection": info.get("selection"),
                        "checkpoint": info.get("checkpoint"),
                        "day_open_balance": float(risk_status.day_open_balance),
                        "risk_governor_triggered": risk_status.kill_switch_state != "normal",
                        "trend_reversal": "reversal" in str(reason).lower(),
                        "news_event": "news" in str(reason).lower(),
                    }
                    asyncio.create_task(
                        get_nemesis_system().report_loss(
                            trade_id=str(ticket),
                            loss_amount=abs(profit),
                            market_context=market_context,
                        )
                    )

                asyncio.create_task(
                    self._send_pnl_to_accountant(
                        symbol=symbol,
                        profit=profit,
                    )
                )

                if profit >= 0.5:
                    asyncio.create_task(
                        self._viralize_trade(
                            symbol=symbol,
                            action=info["action"],
                            pnl=profit,
                        )
                    )
            except Exception as exc:
                logger.error("Erreur lors du traitement de la cloture %s: %s", ticket, exc)
            finally:
                self._trade_open_info.pop(ticket, None)

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

    async def _flatten_all_positions(self, positions: list) -> None:
        """Ferme toutes les positions ouvertes lorsque le kill switch est critique.

        Args:
            positions (list): Positions ouvertes a fermer.
        """
        if not positions:
            return

        ordered_positions = sorted(
            positions,
            key=lambda position: float(getattr(position, "profit", 0.0) or 0.0),
        )
        logger.warning(
            "Flatten immediat active: tentative de fermeture de %s positions.",
            len(ordered_positions),
        )
        for position in ordered_positions:
            result = await self.mt5.close_position(position.ticket)
            if result.get("success"):
                self.risk.note_flatten_action("immediate")
                logger.warning(
                    "Flatten immediat: position %s #%s fermee.",
                    position.symbol,
                    position.ticket,
                )
            else:
                logger.error(
                    "Flatten immediat: echec de fermeture pour %s #%s: %s",
                    position.symbol,
                    position.ticket,
                    result.get("message", "erreur inconnue"),
                )

    async def _flatten_worst_position(self, positions: list) -> bool:
        """Ferme la pire position ouverte pour alleger le drawdown.

        Args:
            positions (list): Positions candidates au flatten progressif.

        Returns:
            bool: ``True`` si une fermeture a ete declenchee.
        """
        if not positions:
            return False

        worst_position = min(
            positions,
            key=lambda position: float(getattr(position, "profit", 0.0) or 0.0),
        )
        result = await self.mt5.close_position(worst_position.ticket)
        if not result.get("success"):
            logger.error(
                "Flatten progressif: echec de fermeture pour %s #%s: %s",
                worst_position.symbol,
                worst_position.ticket,
                result.get("message", "erreur inconnue"),
            )
            return False

        self.risk.note_flatten_action("progressive")
        logger.warning(
            "Flatten progressif: fermeture de la pire position %s #%s.",
            worst_position.symbol,
            worst_position.ticket,
        )
        return True

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

    async def _daily_review_loop(self):
        """Genere une revue journaliere persistante a la fin de chaque journee."""
        while self.is_active:
            try:
                now = datetime.now()
                next_review = now.replace(hour=23, minute=58, second=0, microsecond=0)
                if now >= next_review:
                    next_review = next_review + timedelta(days=1)

                await asyncio.sleep((next_review - now).total_seconds())
                if not self.is_active:
                    break

                await self.generate_trading_review()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Erreur boucle revue journaliere: %s", exc)
                await asyncio.sleep(900)

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
                equity = Decimal(str(summary.get("equity", summary.get("balance", 100000))))
                self.risk.update_account_balance(balance, equity=equity)

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

                if self.risk.should_flatten_immediate():
                    await self._flatten_all_positions(positions)
                    await asyncio.sleep(5)
                    continue

                if self.risk.should_flatten_progressive() and self.risk.can_run_progressive_flatten():
                    flattened = await self._flatten_worst_position(positions)
                    if flattened:
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

                        if nemesis.is_symbol_quarantined(symbol):
                            self._log_pause_state(
                                f"nemesis_quarantine_{symbol}",
                                f"Auto-Trading pause sur {symbol}: quarantaine Nemesis active.",
                                cooldown_seconds=300,
                            )
                            continue
                        self._clear_pause_state(f"nemesis_quarantine_{symbol}")

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
                            "training_compat_mode": self._training_compat_mode,
                            "cortex_required": not self._cpu_live_mode,
                            "gnn_mode": "consultatif" if self._cpu_live_mode else "fusionne",
                            "price": float(current_price),
                            "timestamp": self._normalize_live_timestamp(
                                latest_candle.get("time")
                            ),
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
                        raw_model_action_id = 0
                        raw_model_action = "HOLD"
                        raw_policy = []
                        raw_model_confidence = 0.0
                        raw_model_value = 0.0
                        raw_prediction = "HOLD"
                        checkpoint_path = None
                        model_version = None
                        model_status = "unknown"
                        veto_reason = None
                        
                        try:
                            from shared.internal_auth import InternalAuth
                            observation["selection_policy"] = (
                                "champion_only"
                                if self._cpu_live_mode
                                else str(os.getenv("MUZERO_LIVE_SELECTION_POLICY", "champion_only"))
                            )
                            token = InternalAuth.generate_token("banker")
                            candidate_urls = [self._resolve_live_inference_url()]
                            if (
                                self._cpu_live_mode
                                and not self._live_inference_url.strip()
                            ):
                                candidate_urls.append(self._resolve_legacy_inference_url())
                            
                            async with aiohttp.ClientSession() as session:
                                for url_index, lab_url in enumerate(candidate_urls):
                                    async with session.post(
                                        lab_url,
                                        json=observation,
                                        headers={"X-Hive-Internal-Token": token},
                                        timeout=self._live_inference_timeout_seconds,
                                    ) as resp:
                                        if resp.status == 404 and url_index + 1 < len(candidate_urls):
                                            logger.warning(
                                                "Endpoint %s absent. Repli ponctuel vers %s.",
                                                lab_url,
                                                candidate_urls[url_index + 1],
                                            )
                                            continue

                                        if resp.status == 200:
                                            lab_result = await resp.json()
                                            raw_model_action_id = int(lab_result.get("action", 0) or 0)
                                            raw_model_value = float(lab_result.get("value", 0.0) or 0.0)
                                            raw_model_confidence = float(
                                                lab_result.get("confidence", 0.0) or 0.0
                                            )
                                            raw_policy = self._json_safe_value(
                                                lab_result.get("policy", []) or []
                                            )
                                            checkpoint_path = lab_result.get("checkpoint")
                                            raw_prediction = str(
                                                lab_result.get("prediction") or "HOLD"
                                            )
                                            raw_model_action = self._model_action_label(
                                                raw_model_action_id,
                                                raw_prediction,
                                            )
                                            model_engine = str(lab_result.get("engine", "Modele")).strip() or "Modele"
                                            engine_name = str(lab_result.get("engine_name") or "").strip() or None
                                            model_version = lab_result.get("model_version")
                                            model_status = str(lab_result.get("model_status") or "unknown")
                                            lab_selection = str(lab_result.get("selection") or "none")
                                            lab_selection_policy = str(
                                                lab_result.get("selection_policy") or "unknown"
                                            )
                                            governance = dict(lab_result.get("governance") or {})
                                            ensemble_mode = str(
                                                lab_result.get("ensemble_mode")
                                                or governance.get("mode")
                                                or ""
                                            ).strip() or None
                                            degraded_fallback_reason = str(
                                                lab_result.get("degraded_fallback_reason")
                                                or governance.get("degraded_fallback_reason")
                                                or ""
                                            ).strip() or None
                                            muzero_decision = self._json_safe_value(governance.get("muzero") or {})
                                            dreamer_decision = self._json_safe_value(governance.get("dreamer") or {})
                                            ensemble_scores = self._json_safe_value(governance.get("scores") or {})
                                            live_model_allowed, live_block_reason = self._is_live_model_allowed(lab_result)
                                            dreamer_comment = f"{model_engine} (v={raw_model_value:.2f})"
                                            mt5_order_comment = self._build_order_comment(
                                                engine_label=model_engine,
                                                live_horizon=live_horizon,
                                                selection=lab_selection,
                                                model_value=raw_model_value,
                                            )

                                            if raw_model_action_id == 1:
                                                action = TradeAction.BUY
                                                comment = f"{dreamer_comment} -> BUY"
                                            elif raw_model_action_id == 2:
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
                                            if (
                                                action is not None
                                                and self._cpu_live_mode
                                                and lab_selection_policy != "champion_only"
                                            ):
                                                logger.info(
                                                    "Entree live refusee sur %s: cpu_live exige champion_only (recu=%s).",
                                                    symbol,
                                                    lab_selection_policy,
                                                )
                                                action = None
                                                live_model_allowed = False
                                                live_block_reason = (
                                                    f"selection_policy_invalide:{lab_selection_policy or 'unknown'}"
                                                )
                                                comment = "Mode cpu_live: champion_only requis"
                                            break

                                        error_payload = (await resp.text()).strip()
                                        if len(error_payload) > 500:
                                            error_payload = f"{error_payload[:497]}..."
                                        logger.error(
                                            "Inference live en echec via %s: HTTP %s - %s",
                                            lab_url,
                                            resp.status,
                                            error_payload or "reponse vide",
                                        )
                                        action = None
                                        comment = f"Erreur inference live (HTTP {resp.status})"
                                        break
                                        
                        except Exception as e_lab:
                            # En cas d'echec reseau ou de serialisation, on force HOLD pour proteger le compte.
                            logger.error(
                                "Inference live impossible pour %s: %s - %s. Passage en attente.",
                                symbol,
                                e_lab.__class__.__name__,
                                e_lab,
                            )
                            action = None
                            comment = "Erreur service inference live"

                        action, veto_reason = self._apply_context_veto(action, last_strat)
                        if veto_reason is not None:
                            logger.info(
                                "Veto contextuel sur %s: action=%s biais_final=%s force=%s raison=%s",
                                symbol,
                                raw_model_action,
                                last_strat.get("bias", "NEUTRAL"),
                                last_strat.get("bias_strength", "weak"),
                                veto_reason,
                            )
                            comment = f"Bloque par contexte ({veto_reason})"

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
                            "raw_model_action_id": raw_model_action_id,
                            "raw_model_action": raw_model_action,
                            "raw_prediction": raw_prediction,
                            "raw_policy": raw_policy,
                            "raw_model_confidence": raw_model_confidence,
                            "raw_model_value": raw_model_value,
                            "post_veto_action": self._trade_action_label(action),
                            "veto_reason": veto_reason,
                            "cortex_bias": last_strat.get("cortex_bias", "NEUTRAL"),
                            "gnn_bias": last_strat.get("gnn_bias", "NEUTRAL"),
                            "gnn_scalp_bias": last_strat.get("gnn_scalp_bias", "NEUTRAL"),
                            "gnn_intraday_bias": last_strat.get("gnn_intraday_bias", "NEUTRAL"),
                            "gnn_swing_bias": last_strat.get("gnn_swing_bias", "NEUTRAL"),
                            "gnn_confidence": float(last_strat.get("gnn_confidence", 0.0) or 0.0),
                            "final_bias": last_strat.get("bias", bias),
                            "bias_alignment": last_strat.get("bias_alignment", "unknown"),
                            "bias_strength": last_strat.get("bias_strength", "weak"),
                            "training_compat_mode": self._training_compat_mode,
                            "cpu_live_mode": self._cpu_live_mode,
                            "comment": comment,
                            "timestamp": datetime.now().isoformat(),
                            "selection": lab_selection,
                            "checkpoint": checkpoint_path,
                            "model_version": model_version,
                            "model_status": model_status,
                            "engine_name": engine_name or ("ensemble" if ensemble_mode else "muzero"),
                            "ensemble_mode": ensemble_mode,
                            "degraded_fallback_reason": degraded_fallback_reason,
                            "muzero_decision": muzero_decision,
                            "dreamer_decision": dreamer_decision,
                            "ensemble_decision": {
                                "action": raw_model_action,
                                "confidence": raw_model_confidence,
                                "value": raw_model_value,
                            } if ensemble_mode else {},
                            "ensemble_scores": ensemble_scores,
                            "lab_selection": lab_selection,
                            "lab_selection_policy": lab_selection_policy,
                            "live_model_allowed": live_model_allowed,
                            "live_block_reason": live_block_reason,
                            "mt5_comment": mt5_order_comment,
                        })
                        if self._is_unusable_reasoning(str(decision_state.get("raw_thought", ""))):
                            decision_state["raw_thought"] = comment
                        self.latest_decisions[symbol] = decision_state
                        self._record_decision_audit({
                            "symbol": symbol,
                            "timestamp": decision_state["timestamp"],
                            "raw_model_action": raw_model_action,
                            "post_veto_action": self._trade_action_label(action),
                            "veto_reason": veto_reason,
                            "cortex_bias": decision_state.get("cortex_bias"),
                            "gnn_bias": decision_state.get("gnn_bias"),
                            "final_bias": decision_state.get("final_bias"),
                            "selection": lab_selection,
                            "training_compat_mode": self._training_compat_mode,
                            "checkpoint": checkpoint_path,
                            "model_version": model_version,
                            "model_status": model_status,
                            "engine_name": decision_state.get("engine_name"),
                            "ensemble_mode": decision_state.get("ensemble_mode"),
                            "degraded_fallback_reason": decision_state.get("degraded_fallback_reason"),
                        })
                        await self._publish_trading_context_event(symbol, live_horizon, decision_state)
                        await self._publish_trading_decision_event(symbol, live_horizon, decision_state)

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
                            requested_action = self._trade_action_label(action)
                            logger.info(
                                "Execution ignoree sur %s: volume calcule nul pour le budget risque courant.",
                                symbol,
                            )
                            if self._should_send_veto_alert("volume_zero"):
                                self.telegram.send_sync(
                                    f"*RISK VETO* | {symbol} {action.value if hasattr(action, 'value') else action} bloque\n"
                                    "Raison: risque autorise insuffisant pour calculer un volume exploitable."
                                )
                            await self._publish_execution_event(
                                symbol=symbol,
                                action=requested_action,
                                stage="risk_sizing",
                                allowed=False,
                                reason="volume_calcule_nul",
                                volume=float(dynamic_vol),
                                payload={"risk_percent": float(risk_pct)},
                            )
                            action = None
                            comment = "Risque insuffisant"

                        if action and dynamic_vol < broker_min_volume:
                            requested_action = self._trade_action_label(action)
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
                            await self._publish_execution_event(
                                symbol=symbol,
                                action=requested_action,
                                stage="broker_volume_guard",
                                allowed=False,
                                reason="volume_minimum_broker",
                                volume=float(dynamic_vol),
                                payload={"broker_min_volume": broker_min_volume},
                            )
                            action = None
                            comment = "Volume minimum broker > risque autorise"

                        if action is None:
                            continue

                        # Le mode CPU live privilegie un plafond de taille
                        # tres conservateur pour la demo pendant le training GPU.
                        max_volume_cap = (
                            self._cpu_live_max_volume
                            if self._cpu_live_mode
                            else 0.10
                        )
                        if self._cpu_live_mode:
                            symbol_cap = self._cpu_live_symbol_max_volumes.get(symbol.upper())
                            if symbol_cap is not None:
                                max_volume_cap = max(0.01, float(symbol_cap))
                        if str(decision_state.get("ensemble_mode") or "").lower() == "degraded_muzero_only":
                            # En mode degrade, on reduit la taille pour proteger la demo
                            # tant que DreamerV3 n'est pas capable de participer au vote.
                            max_volume_cap = max(0.01, round(max_volume_cap * 0.5, 2))
                        final_vol = min(max_volume_cap, dynamic_vol)

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
                                await self._publish_execution_event(
                                    symbol=symbol,
                                    action=self._trade_action_label(action),
                                    stage="position_guard",
                                    allowed=False,
                                    reason="position_deja_ouverte",
                                    volume=float(final_vol),
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
                                    requested_action = self._trade_action_label(action)
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
                                    await self._publish_execution_event(
                                        symbol=symbol,
                                        action=requested_action,
                                        stage="margin_guard",
                                        allowed=False,
                                        reason="marge_insuffisante",
                                        volume=float(final_vol),
                                        payload={
                                            "margin_required": margin_required,
                                            "free_margin": free_margin,
                                        },
                                    )
                                    action = None
                                    comment = "Marge insuffisante"
                            
                        # If action was vetoed by margin check, skip execution
                        if action is None:
                            continue

                        validation = await self.risk.validate_order(order)
                        if validation["allowed"]:
                            logger.info(f"ðŸ¤– EXEC {symbol}: {action} | {comment}")
                            await self._publish_execution_event(
                                symbol=symbol,
                                action=self._trade_action_label(action),
                                stage="risk_validation",
                                allowed=True,
                                reason="validation_ok",
                                volume=float(order.volume),
                                payload=validation,
                            )
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
                                        "raw_model_action": raw_model_action,
                                        "veto_reason": veto_reason,
                                        "gnn_bias": decision_state.get("gnn_bias"),
                                        "final_bias": decision_state.get("final_bias"),
                                        "spread": float((tick["ask"] - tick["bid"]) if isinstance(tick, dict) else 0.0),
                                        "model_version": decision_state.get("model_version"),
                                        "engine_name": decision_state.get("engine_name"),
                                        "selection": lab_selection,
                                        "checkpoint": checkpoint_path,
                                        "atr": float(features.get("ATR", 0.0) or 0.0),
                                    }
                                    self._known_tickets.add(ticket)
                                    self._last_symbol_entry_at[symbol] = now_open
                                    open_symbols.add(symbol)

                                await self._publish_execution_event(
                                    symbol=symbol,
                                    action=self._trade_action_label(action),
                                    stage="execution",
                                    allowed=True,
                                    reason="ordre_execute",
                                    volume=float(order.volume),
                                    ticket=int(ticket) if ticket else None,
                                    payload=result,
                                )
                                
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
                                await self._publish_execution_event(
                                    symbol=symbol,
                                    action=self._trade_action_label(action),
                                    stage="execution",
                                    allowed=False,
                                    reason=fail_msg,
                                    volume=float(fail_volume),
                                    payload=result,
                                )
                        else:
                            logger.warning(f"Rejected {symbol}: {validation['reason']}")
                            await self._publish_execution_event(
                                symbol=symbol,
                                action=self._trade_action_label(action),
                                stage="risk_validation",
                                allowed=False,
                                reason=str(validation.get("reason") or "validation_refusee"),
                                volume=float(order.volume),
                                payload=validation,
                            )
                            
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

