"""
Modèles de Données Partagés Pydantic (DTOs).

Ce module définit le "Langage Commun" de THE HIVE. Tous les agents (Core, Banker, etc.)
communiquent en sérialisant/désérialisant ces modèles via Redis ou API REST.

Contient :
- Énumérations (Types d'ordres, Rôles, Sévérité).
- Modèles Trading (Ordres, Positions, Risque).
- Modèles Communication (Messages Chat & Inter-Agents).
- Modèles Sécurité (Audit, Alertes).
- Modèles Système (Métriques Hardware/GPU).
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class TradeAction(str, Enum):
    """
    Direction d'un ordre de trading.

    Values:
        BUY: Achat (Long).
        SELL: Vente (Short).
    """
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """
    Type d'exécution pour un ordre de bourse.

    Values:
        MARKET: Exécution immédiate au prix du marché.
        LIMIT: Exécution à un prix spécifique ou meilleur.
        STOP: Exécution lorsque le prix atteint un seuil spécifique.
        STOP_LIMIT: Combinaison d'un ordre Stop et Limit.
    """
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSource(str, Enum):
    """
    Origine de l'ordre de trading (Auditabilité).

    Permet de savoir QUI ou QUOI a initié la transaction pour les logs de compliance.

    Values:
        VOICE: Commande vocale utilisateur.
        CHAT: Commande textuelle chat.
        API: Appel API externe.
        STRATEGY: Automatisme/Stratégie interne (ex: Hedging).
        COPY: Copy-Trading depuis un compte maître.
    """
    VOICE = "VOICE"
    CHAT = "CHAT"
    API = "API"
    STRATEGY = "STRATEGY"
    COPY = "COPY"


class IntentType(str, Enum):
    """
    Classification des intentions utilisateur (NLU).

    Utilisé par le Router pour diriger la requête vers le bon Expert.

    Values:
        TRADING_ORDER: Demande d'achat/vente -> Banker.
        POSITION_STATUS: Demande d'état des positions -> Banker.
        RISK_INQUIRY: Question sur le risque/exposition -> Banker/Risk.
        GENERAL_CHAT: Conversation banale -> Core/LLM.
        MEMORY_RECALL: Recherche d'infos passées -> Core/Memory.
        OSINT_REQUEST: Recherche d'infos sur le web -> Sentinel.
        SYSTEM_COMMAND: Ordre technique (reboot, logs) -> Builder.
        SECURITY_ALERT: Signalement de menace -> Kernel/Compliance.
    """
    TRADING_ORDER = "TRADING_ORDER"
    POSITION_STATUS = "POSITION_STATUS"
    RISK_INQUIRY = "RISK_INQUIRY"
    CHAT = "CHAT"
    GENERAL_CHAT = "CHAT"
    MEMORY_RECALL = "MEMORY_RECALL"
    OSINT_REQUEST = "OSINT_REQUEST"
    SYSTEM_COMMAND = "SYSTEM_COMMAND"
    SECURITY_ALERT = "SECURITY_ALERT"


class MessageRole(str, Enum):
    """
    Rôle de l'interlocuteur dans une conversation.

    Values:
        USER: L'utilisateur humain.
        ASSISTANT: L'IA ou l'expert.
        SYSTEM: Instructions système ou contexte.
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SecuritySeverity(str, Enum):
    """
    Niveau de criticité d'un événement de sécurité.

    Values:
        LOW: Information ou événement mineur.
        MEDIUM: Avertissement nécessitant une attention.
        HIGH: Incident sérieux, intervention requise.
        CRITICAL: Danger immédiat pour le système ou le capital (Loi 0/2).
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMessageType(str, Enum):
    """
    Type sémantique d'un message échangé entre agents via Redis.

    Values:
        REQUEST: Demande d'action ou d'information (attend une réponse).
        RESPONSE: Réponse à une requête précédente.
        EVENT: Notification d'un changement d'état (Fire-and-forget).
        ALERT: Signalement d'une anomalie ou danger.
        HEARTBEAT: Signal de vie périodique.
        SWARM_COMMAND: Ordre de coordination pour l'essaim.
    """
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    SWARM_COMMAND = "swarm_command"


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class TradeOrder(BaseModel):
    """
    Structure standardisée d'un ordre de trading.

    Attributes:
        id (UUID): Identifiant unique de l'ordre (interne).
        symbol (str): Paire ou actif (ex: XAUUSD).
        action (TradeAction): Achat ou Vente.
        volume (Decimal): Taille du lot (0.01 à 10.0).
        entry_price (Decimal | None): Prix d'entree estime au moment de la decision.
        stop_loss_price (Decimal | None): Prix de sortie en perte (Obligatoire).
        take_profit_price (Decimal | None): Prix de sortie en gain (Optionnel).
        order_type (OrderType): Market, Limit, Stop...
        source (OrderSource): Origine de la demande.
        account_id (UUID | None): Compte cible (si multi-comptes).
    """
    id: UUID = Field(default_factory=uuid4)
    symbol: str = Field(..., description="Symbole (ex: XAUUSD)")
    action: TradeAction
    volume: Decimal = Field(..., gt=0, le=10)
    entry_price: Decimal | None = Field(
        None,
        description="Prix d'entree estime utilise pour le calcul de risque",
    )
    stop_loss_price: Decimal | None = Field(None, description="Prix Stop Loss (obligatoire ROE)")
    take_profit_price: Decimal | None = None
    order_type: OrderType = OrderType.MARKET
    source: OrderSource = OrderSource.CHAT
    account_id: UUID | None = None
    magic_number: int = 12345
    comment: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {Decimal: str}


class Position(BaseModel):
    """
    Représente une position de trading actuellement ouverte sur le marché.

    Attributes:
        ticket (int): Identifiant unique de la position (MT5).
        symbol (str): L'actif tradé (ex: EURUSD).
        action (TradeAction): BUY ou SELL.
        volume (Decimal): Taille de la position en lots.
        open_price (Decimal): Prix d'ouverture.
        current_price (Decimal): Prix actuel du marché.
        profit (Decimal): Profit ou perte latent (Floating P&L).
    """
    ticket: int
    symbol: str
    action: TradeAction
    volume: Decimal
    open_price: Decimal
    current_price: Decimal
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    profit: Decimal
    swap: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    magic_number: int = 0
    open_time: datetime

    @property
    def pnl_pips(self) -> float:
        """
        Calcule le profit ou la perte approximative en pips.

        Returns:
            float: La différence en pips (positif pour un gain, négatif pour une perte).
        """
        diff = float(self.current_price - self.open_price)
        if self.action == TradeAction.SELL:
            diff = -diff
        # Simplification: 1 pip = 0.0001 pour forex, 0.1 pour or
        pip_size = 0.1 if "XAU" in self.symbol else 0.0001
        return diff / pip_size


class RiskStatus(BaseModel):
    """
    Rapport d'état des risques en temps réel (Loi 2).

    Agrège les données de tous les comptes pour donner une vision consolidée
    de l'exposition au risque.

    Attributes:
        daily_drawdown_percent (Decimal): Perte journalière en % (Max 4%).
        total_drawdown_percent (Decimal): Perte totale en % (Max 8%).
        open_positions_count (int): Nombre de trades actifs.
        trading_allowed (bool): Si False, le Kernel bloque tout nouvel ordre.
    """
    account_id: UUID
    timestamp: datetime = Field(default_factory=datetime.now)
    daily_drawdown_percent: Decimal = Decimal("0")
    total_drawdown_percent: Decimal = Decimal("0")
    open_positions_count: int = 0
    anti_tilt_active: bool = False
    anti_tilt_expires_at: datetime | None = None
    news_filter_active: bool = False
    trading_allowed: bool = True

    def check_trading_allowed(
        self,
        max_daily_dd: Decimal = Decimal("4.0"),
        max_total_dd: Decimal = Decimal("8.0"),
        max_positions: int = 3,
    ) -> tuple[bool, str | None]:
        """
        Vérifie si le trading est autorisé selon les règles de la Constitution (Loi 2).

        Args:
            max_daily_dd (Decimal): Limite de perte journalière (défaut 4%).
            max_total_dd (Decimal): Limite de perte totale (défaut 8%).
            max_positions (int): Nombre max de positions simultanées (défaut 3).

        Returns:
            tuple[bool, str | None]: (Autorisé?, Raison du refus ou None).
        """
        if self.anti_tilt_active:
            return False, "ANTI_TILT_ACTIVE"
        if self.news_filter_active:
            return False, "NEWS_FILTER_ACTIVE"
        if self.daily_drawdown_percent >= max_daily_dd:
            return False, "DAILY_DRAWDOWN_LIMIT"
        if self.total_drawdown_percent >= max_total_dd:
            return False, "TOTAL_DRAWDOWN_LIMIT"
        if self.open_positions_count >= max_positions:
            return False, "MAX_POSITIONS_REACHED"
        return True, None


class AccountBalance(BaseModel):
    """
    État financier d'un compte de trading à un instant T.

    Attributes:
        login (int): Numéro de compte.
        balance (Decimal): Solde clôturé (sans le flottant).
        equity (Decimal): Solde avec flottant (Balance + P&L latent).
        margin (Decimal): Marge utilisée pour les positions ouvertes.
        free_margin (Decimal): Marge disponible pour de nouveaux trades.
    """
    login: int
    server: str
    balance: Decimal
    equity: Decimal
    margin: Decimal = Decimal("0")
    free_margin: Decimal = Decimal("0")
    margin_level: float | None = None
    currency: str = "USD"
    leverage: int = 100
    timestamp: datetime = Field(default_factory=datetime.now)


class PropFirmAccount(BaseModel):
    """
    Représentation d'un compte financé par une Prop Firm (ex: FTMO).

    Ces comptes ont des règles spécifiques de drawdown et de cibles de profit.

    Attributes:
        id (UUID): Identifiant unique interne.
        phase (str): Phase du challenge (challenge, verification, funded).
        max_daily_loss_percent (Decimal): Limite de perte journalière (ex: 4.0).
        max_total_loss_percent (Decimal): Limite de perte totale (ex: 8.0).
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    login: int
    server: str
    broker: str
    phase: str = Field(..., description="challenge|verification|funded")
    initial_balance: Decimal
    current_balance: Decimal
    max_daily_loss_percent: Decimal = Decimal("4.0")
    max_total_loss_percent: Decimal = Decimal("8.0")
    profit_target_percent: Decimal | None = None
    copy_enabled: bool = True
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMUNICATION MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class AgentMessage(BaseModel):
    """
    Enveloppe standardisée pour les messages asynchrones inter-agents.

    Attributes:
        type (AgentMessageType): Nature du message (Request, Alert...).
        source_agent (str): L'expéditeur (ex: core).
        target_agent (str): Le destinataire (ex: banker ou all).
        payload (dict): Le contenu utile du message.
        correlation_id (UUID): Pour lier une requête à sa réponse.
    """
    id: UUID = Field(default_factory=uuid4)
    type: AgentMessageType
    source_agent: str
    target_agent: str
    action: str
    payload: dict[str, Any] = {}
    correlation_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    ttl_seconds: int = 30

    def to_redis_channel(self) -> str:
        """
        Génère le topic Redis Pub/Sub canonique pour ce message.

        Le format standard est `eva.{target_agent}.{type}s`.
        Permet le routage et le broadcast (si target='all').

        Returns:
            str: Le nom du canal Redis (ex: "eva.banker.requests").
        """
        return f"eva.{self.target_agent}.{self.type.value}s"


class SwarmDrone(BaseModel):
    """
    Représente un Agent Autonome (Drone) exécutant une tâche de fond persistante.

    Utilisé pour les missions de surveillance continue ou de calcul distribué
    (Mode Swarm).

    Attributes:
        name (str): Identifiant lisible du drone.
        parent_agent (str): L'expert responsable de ce drone.
        mission (str): Description textuelle de l'objectif.
        status (str): État actuel (active, idle, terminated).
    """
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Nom du drone (ex: GoldSurveillance)")
    parent_agent: str = Field(..., description="L'expert qui a lancé le drone")
    mission: str = Field(..., description="Description de la tâche autonome")
    status: str = "active"
    started_at: datetime = Field(default_factory=datetime.now)
    last_callback: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = {}


class ChatMessage(BaseModel):
    """
    Message individuel au sein d'une session de chat.

    Attributes:
        session_id (UUID): L'identifiant de la conversation.
        role (MessageRole): Qui parle (User, Assistant, System).
        content (str): Le texte du message.
        thoughts (str | None): Le raisonnement interne (CoT) si disponible.
    """
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: MessageRole
    content: str
    thoughts: str | None = None  # Trace de raisonnement de l'expert
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = {}


class Intent(BaseModel):
    """
    Résultat de la classification sémantique d'une requête utilisateur.

    Attributes:
        intent_type (IntentType): La catégorie d'action détectée.
        confidence (float): Score de confiance (0.0 à 1.0).
        entities (dict): Entités nommées extraites (ex: symbol=XAUUSD).
        target_expert (str): L'agent le plus qualifié pour répondre.
    """
    intent_type: IntentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: dict[str, Any] = {}
    target_expert: str = "core"
    raw_text: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class SecurityEvent(BaseModel):
    """
    Représente un événement lié à la sécurité du système (Wazuh, Sentinel).

    Attributes:
        event_type (str): Type d'incident (ex: 'ssh_login_failed').
        severity (SecuritySeverity): Niveau de risque estimé.
        source_ip (str | None): Adresse IP de l'attaquant potentiel.
        action_taken (str | None): Réponse automatique du système (ex: 'blocked').
    """
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str
    source_ip: str | None = None
    target_service: str | None = None
    severity: SecuritySeverity
    description: str = ""
    details: dict[str, Any] = {}
    action_taken: str | None = None
    resolved: bool = False


class AuditRecord(BaseModel):
    """
    Enregistrement immuable pour la Black Box (Loi 3).

    Chaque action critique du système modifie cet enregistrement, qui est ensuite
    hashé et chaîné au précédent pour former une blockchain locale infalsifiable.

    Attributes:
        agent (str): L'agent responsable de l'action.
        action (str): L'action effectuée.
        previous_hash (str): Hash de l'enregistrement précédent (Chaînage).
        record_hash (str): Hash de l'enregistrement courant.
    """
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    agent: str
    action: str
    details: dict[str, Any] = {}
    user_id: str | None = None
    session_id: UUID | None = None
    previous_hash: str = ""
    record_hash: str = ""

    def compute_hash(self, previous_hash: str = "") -> str:
        """
        Calcule l'empreinte cryptographique (SHA-256) de l'audit.

        Garantit l'intégrité des données en incluant le hash précédent.

        Args:
            previous_hash (str): Le hash du bloc précédent dans la chaîne.

        Returns:
            str: Le hash hexadécimal calculé.
        """
        import hashlib
        import json

        data = {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "agent": self.agent,
            "action": self.action,
            "details": self.details,
            "previous_hash": previous_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class BaseHealthResponse(BaseModel):
    """
    Réponse de santé de base pour tous les services (Health Check).

    Attributes:
        status (str): État global (ok, error, degraded).
        version (str): Version sémantique du service.
    """
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentStatus(BaseModel):
    """
    État de santé et versioning d'un agent de la Ruche.

    Attributes:
        agent_id (str): Identifiant unique de l'agent.
        status (str): État actuel (online, offline, error).
        last_seen (datetime): Horodatage du dernier heartbeat.
        version (str): Version du logiciel.
    """
    agent_id: str
    status: str = "online"
    last_seen: datetime = Field(default_factory=datetime.now)
    version: str = "0.1.0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GPUMetrics(BaseModel):
    """
    Métriques de santé du GPU (Loi 0 - Préservation Matérielle).

    Surveillance critique de la température pour éviter la dégradation du hardware
    local (RTX 3090).

    Attributes:
        temperature_celsius (float): Température critique.
        utilization_percent (float): Charge GPU.
    """
    name: str = "NVIDIA GeForce RTX 3090"
    temperature_celsius: float
    utilization_percent: float
    memory_used_mb: int
    memory_total_mb: int = 24576
    power_draw_watts: float = 0.0
    fan_speed_percent: int = 0

    def is_overheating(self, threshold: float = 90.0) -> bool:
        """
        Vérifie si la température dépasse le seuil de sécurité.

        Args:
            threshold (float): Limite en degrés Celsius (défaut 90.0).

        Returns:
            bool: True si surchauffe détectée.
        """
        return self.temperature_celsius >= threshold


class HardwareMetrics(BaseModel):
    """
    Métriques de performance du système hôte (Proxmox/Linux).

    Attributes:
        cpu_percent (float): Utilisation CPU globale.
        ram_used_gb (float): Mémoire vive utilisée en Go.
        disk_used_percent (float): Espace disque utilisé (%).
        gpu (GPUMetrics | None): Métriques GPU si disponible.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    cpu_percent: float
    cpu_freq_mhz: float = 0.0
    ram_used_gb: float
    ram_total_gb: float
    swap_used_gb: float = 0.0
    disk_used_percent: float = 0.0
    gpu: GPUMetrics | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class HiveError(BaseModel):
    """
    Structure d'erreur standardisée pour THE HIVE.

    Permet une gestion uniforme des exceptions à travers tous les micro-services.

    Attributes:
        code (str): Code d'erreur unique (ex: 'AUTH_FAILED').
        message (str): Description lisible de l'erreur en Français.
        severity (SecuritySeverity): Niveau de gravité.
        recoverable (bool): Indique si le client peut réessayer.
    """
    code: str
    message: str
    severity: SecuritySeverity = SecuritySeverity.MEDIUM
    category: str = "GENERAL"
    recoverable: bool = True
    details: dict[str, Any] = {}
    constitution_reference: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ConnectorMode(str, Enum):
    """
    Etat explicite d'un connecteur ou d'un moteur externe.

    Values:
        DISABLED: Connecteur inactif ou stub non exploitable.
        PAPER: Connecteur en simulation ou en mode degrade non executable.
        LIVE: Connecteur operationnel sur le chemin reel.
    """

    DISABLED = "disabled"
    PAPER = "paper"
    LIVE = "live"


class RuntimeMode(str, Enum):
    """
    Mode runtime stable de l'usine de trading.

    Values:
        DEMO_LIVE: Execution demo standard.
        TRAINING_CPU_LIVE: Trading live minimal pendant un training GPU.
        MAINTENANCE: Service en lecture seule ou arrete.
    """

    DEMO_LIVE = "demo_live"
    TRAINING_CPU_LIVE = "training_cpu_live"
    MAINTENANCE = "maintenance"


class EventEnvelope(BaseModel):
    """
    Enveloppe canonique commune a tous les evenements metier.

    Attributes:
        envelope_id (UUID): Identifiant unique de l'evenement.
        event_type (str): Type semantique stable de l'evenement.
        source (str): Service emetteur.
        created_at (datetime): Horodatage de creation.
        metadata (dict[str, Any]): Metadonnees transverses de correlation.
    """

    envelope_id: UUID = Field(default_factory=uuid4)
    event_type: str
    source: str
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradingContextEnvelope(EventEnvelope):
    """
    Etat de contexte de marche utilise pour une decision live.

    Attributes:
        runtime_mode (RuntimeMode): Mode d'execution courant.
        symbol (str): Symbole evalue.
        horizon (str): Horizon du modele sollicite.
        market_state (dict[str, Any]): Indicateurs et biais de contexte.
        connectors (dict[str, Any]): Etat des dependances de decision.
    """

    event_type: str = "trading.context"
    source: str = "banker"
    runtime_mode: RuntimeMode
    symbol: str
    horizon: str
    market_state: dict[str, Any] = Field(default_factory=dict)
    connectors: dict[str, Any] = Field(default_factory=dict)


class TradingDecisionEnvelope(EventEnvelope):
    """
    Decision brute et post-filtrage produite par le banker.

    Attributes:
        runtime_mode (RuntimeMode): Mode runtime ayant produit la decision.
        symbol (str): Symbole traite.
        horizon (str): Horizon du modele choisi.
        raw_model_action (str): Action brute du modele.
        post_veto_action (str): Action finale retenue.
        selection (str): Source du modele retenu.
        checkpoint (str | None): Checkpoint utilise pour l'inference.
        final_bias (str): Biais final applique a la decision.
        veto_reason (str | None): Motif de veto si applicable.
        connectors (dict[str, Any]): Etat des dependances de decision.
    """

    event_type: str = "trading.decision"
    source: str = "banker"
    runtime_mode: RuntimeMode
    symbol: str
    horizon: str
    engine: str | None = None
    raw_model_action: str
    post_veto_action: str
    selection: str
    checkpoint: str | None = None
    final_bias: str = "NEUTRAL"
    veto_reason: str | None = None
    ensemble_mode: str | None = None
    degraded_fallback_reason: str | None = None
    connectors: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class ExecutionEventEnvelope(EventEnvelope):
    """
    Evenement d'execution ou de refus d'ordre.

    Attributes:
        runtime_mode (RuntimeMode): Mode runtime courant.
        symbol (str): Symbole concerne.
        action (str): Action demandee.
        stage (str): Etape du pipeline d'execution.
        allowed (bool): Indique si l'etape a permis la poursuite.
        reason (str | None): Motif principal du refus ou du resultat.
        volume (float | None): Volume envoye ou refuse.
        spread_points (float | None): Spread releve au moment du controle.
        ticket (int | None): Ticket MT5 si execution reussie.
    """

    event_type: str = "trading.execution"
    source: str = "banker"
    runtime_mode: RuntimeMode
    symbol: str
    action: str
    stage: str
    allowed: bool
    reason: str | None = None
    volume: float | None = None
    spread_points: float | None = None
    ticket: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class PromotionReportEnvelope(EventEnvelope):
    """
    Rapport de promotion ou de blocage d'un challenger.

    Attributes:
        horizon (str): Horizon strategique concerne.
        family (str | None): Famille d'actifs eventuelle.
        live_champion_id (str | None): Champion live courant.
        challenger_id (str | None): Challenger evalue.
        promotion_gate (dict[str, Any]): Verdict et motif principal.
        promotion_checks (dict[str, Any]): Checks unitaires de promotion.
        metrics_by_symbol (dict[str, Any]): Metriques par actif.
        metrics_by_position_mechanics (dict[str, Any]): Metriques de sortie et de gestion de position.
        feature_profile (str | None): Profil de features actif.
        dataset_id (str | None): Identifiant immuable du dataset d'evaluation.
        top_live_symbols (list[str]): Selection live recommandee.
    """

    event_type: str = "training.promotion"
    source: str = "lab"
    engine: str = "muzero"
    horizon: str
    family: str | None = None
    live_champion_id: str | None = None
    challenger_id: str | None = None
    promotion_gate: dict[str, Any] = Field(default_factory=dict)
    promotion_checks: dict[str, Any] = Field(default_factory=dict)
    metrics_by_symbol: dict[str, Any] = Field(default_factory=dict)
    metrics_by_position_mechanics: dict[str, Any] = Field(default_factory=dict)
    feature_profile: str | None = None
    dataset_id: str | None = None
    failure_mode: str | None = None
    top_live_symbols: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class TrainingRunEnvelope(EventEnvelope):
    """
    Etat structure d'un run d'entrainement en cours ou termine.

    Attributes:
        run_id (str | None): Identifiant du run.
        horizon (str | None): Horizon courant du run.
        family (str | None): Famille d'actifs ciblee si disponible.
        feature_profile (str | None): Profil de features applique au run.
        dataset_id (str | None): Identifiant immuable du dataset courant.
        dataset_source (str | None): Source historique utilisee par le run.
        phase (str | None): Phase courante.
        current_symbol (str | None): Symbole courant.
        status (str): Etat global du run.
        arena_progress (dict[str, Any] | None): Progression Arena en direct.
        dependencies (dict[str, Any]): Dependances utiles au run.
        universe (dict[str, Any]): Resume de l'univers entraine.
    """

    event_type: str = "training.run"
    source: str = "lab"
    engine: str | None = None
    run_id: str | None = None
    horizon: str | None = None
    family: str | None = None
    feature_profile: str | None = None
    dataset_id: str | None = None
    dataset_source: str | None = None
    mechanics_profile_version: str | None = None
    ga_status: str | None = None
    ga_generation: int | None = None
    ga_trial: str | None = None
    trial_mode: str | None = None
    trial_cost_profile: str | None = None
    replay_cache_status: str | None = None
    replay_cache_key: str | None = None
    replay_cache_entries: int | None = None
    replay_cache_source: str | None = None
    shadow_buffer_size: int | None = None
    sequence_length: int | None = None
    sequence_stride: int | None = None
    world_model_steps: int | None = None
    dataset_coverage: dict[str, Any] = Field(default_factory=dict)
    phase: str | None = None
    current_symbol: str | None = None
    status: str = "idle"
    arena_progress: dict[str, Any] | None = None
    dependencies: dict[str, Any] = Field(default_factory=dict)
    universe: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
