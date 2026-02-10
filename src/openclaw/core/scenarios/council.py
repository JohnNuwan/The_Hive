"""
OpenClaw War Room Scenario: THE COUNCIL
Part of Sovereign Stack V3.0 — Sprint WR-3

Scénario de prise de décision financière avec logique de Veto.
Le Banker propose un trade, Shadow le contredit, Quant vérifie les chiffres.

Le point critique : si le risque dépasse les seuils (Loi 2), le trade est
physiquement bloqué via un Hard Kill, peu importe le score du vote.

Workflow :
    1. Banker détecte une opportunité (ex: Double Top, Break of Structure).
    2. Le Council est convoqué avec les indicateurs techniques comme preuves.
    3. Débat contradictoire (3 tours).
    4. Vote pondéré + vérification Veto automatique.
    5. Si Approved : le trade est autorisé. Si Rejected/Veto : Hard Kill.

Références :
    - CDcs War Rooms, Sprint 3 (Semaine 3)
    - Loi 2 : Protection du Capital (Risque < 2%)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from ..war_room import WarRoomSession, WarRoomVerdict, VoteChoice
from ..war_room_prompts import WarRoomType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class VetoReason(Enum):
    """Raisons possibles pour un Veto automatique (Hard Kill)."""

    RISK_TOO_HIGH = "risk_too_high"     # Drawdown > 4% ou risque > 2%
    ILLEGAL = "illegal"                  # Advocate dit "Illégal"
    INSUFFICIENT_DATA = "insufficient"   # Pas assez de données pour décider
    NONE = "none"                        # Pas de veto


@dataclass
class TradeProposal:
    """Proposition de trade soumise au Council.

    Attributes:
        symbol: Paire de trading (ex: "XAUUSD", "EURUSD").
        direction: BUY ou SELL.
        lot_size: Taille du lot.
        entry_price: Prix d'entrée visé.
        stop_loss: Niveau de Stop Loss.
        take_profit: Niveau de Take Profit.
        risk_percent: Risque en % du capital.
        indicators: Dictionnaire d'indicateurs techniques.
    """

    symbol: str
    direction: str  # "BUY" ou "SELL"
    lot_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_percent: float = 0.0
    indicators: Dict[str, Any] = field(default_factory=dict)

    def to_evidence_text(self) -> str:
        """Convertit la proposition en texte de preuves pour le débat.

        Returns:
            Texte formaté avec tous les détails du setup.
        """
        rr_ratio = 0.0
        if self.stop_loss != self.entry_price:
            risk_pips = abs(self.entry_price - self.stop_loss)
            reward_pips = abs(self.take_profit - self.entry_price)
            rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0.0

        lines = [
            f"📊 TRADE PROPOSAL: {self.direction} {self.symbol}",
            f"  Entry: {self.entry_price}",
            f"  Stop Loss: {self.stop_loss}",
            f"  Take Profit: {self.take_profit}",
            f"  Lot Size: {self.lot_size}",
            f"  Risk: {self.risk_percent:.1f}% du capital",
            f"  Risk/Reward: 1:{rr_ratio:.1f}",
        ]

        if self.indicators:
            lines.append("  📈 INDICATEURS:")
            for key, val in self.indicators.items():
                lines.append(f"    - {key}: {val}")

        return "\n".join(lines)


@dataclass
class CouncilDecision:
    """Résultat de la délibération du Council.

    Attributes:
        proposal: La proposition évaluée.
        verdict: Verdict de la War Room.
        veto: Raison du veto (NONE si pas de veto).
        trade_allowed: True si le trade est autorisé.
        report_text: Rapport final formaté.
        timestamp: Horodatage de la décision.
    """

    proposal: TradeProposal
    verdict: WarRoomVerdict
    veto: VetoReason = VetoReason.NONE
    trade_allowed: bool = False
    report_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# VETO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


# Seuils de sécurité (Loi 2 : Protection du Capital)
MAX_RISK_PERCENT = 2.0       # Risque max par trade
MAX_DAILY_DRAWDOWN = 4.0     # Drawdown journalier max (FTMO)
MIN_RR_RATIO = 1.5           # Ratio Risk/Reward minimum


def check_hard_veto(proposal: TradeProposal, verdict: WarRoomVerdict) -> VetoReason:
    """Vérifie les conditions de Veto automatique (Hard Kill).

    Le Veto est PRIORITAIRE sur le vote du Council. Même si le vote
    est 100% APPROVE, le trade est bloqué si un seuil est dépassé.

    Conditions de Veto :
        1. Risque > MAX_RISK_PERCENT (2%)
        2. Un votant ADVOCATE a voté REJECT (présomption d'illégalité)
        3. Ratio R/R insuffisant (< 1.5)

    Args:
        proposal: La proposition de trade.
        verdict: Le verdict du vote.

    Returns:
        VetoReason.NONE si pas de veto, sinon la raison.
    """
    # Veto 1 : Risque trop élevé
    if proposal.risk_percent > MAX_RISK_PERCENT:
        logger.warning(
            f"[Council] ⛔ HARD VETO: Risk {proposal.risk_percent:.1f}% "
            f"> max {MAX_RISK_PERCENT}%"
        )
        return VetoReason.RISK_TOO_HIGH

    # Veto 2 : Advocate (légal) a voté REJECT
    for vote in verdict.votes:
        if vote.expert == "ADVOCATE" and vote.choice == VoteChoice.REJECT:
            logger.warning("[Council] ⛔ HARD VETO: Advocate voted REJECT (Illegal)")
            return VetoReason.ILLEGAL

    # Veto 3 : Ratio R/R insuffisant
    if proposal.stop_loss != proposal.entry_price:
        risk_pips = abs(proposal.entry_price - proposal.stop_loss)
        reward_pips = abs(proposal.take_profit - proposal.entry_price)
        rr = reward_pips / risk_pips if risk_pips > 0 else 0.0
        if rr < MIN_RR_RATIO:
            logger.warning(
                f"[Council] ⛔ HARD VETO: R/R ratio {rr:.1f} < min {MIN_RR_RATIO}"
            )
            return VetoReason.RISK_TOO_HIGH

    return VetoReason.NONE


# ═══════════════════════════════════════════════════════════════════════════════
# COUNCIL SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════


class CouncilTradeReview:
    """Scénario de review de trade par le Council.

    Orchestre le débat contradictoire entre Banker, Shadow et Quant
    sur une proposition de trade, avec vérification automatique
    des seuils de sécurité (Hard Veto).

    Usage :
        council = CouncilTradeReview()
        decision = await council.evaluate(proposal, llm_service)
        if decision.trade_allowed:
            execute_trade(proposal)
    """

    async def evaluate(
        self,
        proposal: TradeProposal,
        llm_service,
        memory_bridge=None,
    ) -> CouncilDecision:
        """Évalue une proposition de trade via le Council.

        Args:
            proposal: La proposition de trade à évaluer.
            llm_service: Service LLM pour les agents.
            memory_bridge: MemoryBridge optionnel pour archivage.

        Returns:
            CouncilDecision avec verdict, veto éventuel, et autorisation.
        """
        logger.info(
            f"[Council] Evaluating: {proposal.direction} {proposal.symbol} "
            f"({proposal.lot_size} lots, risk={proposal.risk_percent:.1f}%)"
        )

        # Préparer les preuves (indicateurs techniques)
        evidence = proposal.to_evidence_text()

        # Lancer la session War Room COUNCIL
        session = WarRoomSession(
            room_type=WarRoomType.COUNCIL,
            subject=evidence,
        )
        verdict = await session.run_debate(
            llm_service=llm_service,
            memory_bridge=memory_bridge,
        )

        # Vérification Hard Veto (PRIORITAIRE sur le vote)
        veto = check_hard_veto(proposal, verdict)

        # Décision finale
        trade_allowed = verdict.approved and veto == VetoReason.NONE

        decision = CouncilDecision(
            proposal=proposal,
            verdict=verdict,
            veto=veto,
            trade_allowed=trade_allowed,
        )
        decision.report_text = self._generate_report(decision)

        status = "✅ AUTORISÉ" if trade_allowed else "❌ BLOQUÉ"
        if veto != VetoReason.NONE:
            status += f" (VETO: {veto.value})"

        logger.info(f"[Council] Decision: {status}")
        return decision

    def _generate_report(self, decision: CouncilDecision) -> str:
        """Génère le rapport final du Council.

        Args:
            decision: La décision à formater.

        Returns:
            Texte Markdown formaté du rapport de décision.
        """
        p = decision.proposal
        status = "✅ TRADE AUTORISÉ" if decision.trade_allowed else "❌ TRADE BLOQUÉ"

        lines = [
            f"# 🏛️ COUNCIL DECISION — Session {decision.verdict.session_id}",
            f"**Statut: {status}**",
            f"**Setup: {p.direction} {p.symbol} @ {p.entry_price}**",
            f"**Risk: {p.risk_percent:.1f}% | Score: {decision.verdict.approval_score:.0%}**",
            "",
        ]

        if decision.veto != VetoReason.NONE:
            lines.append(f"⛔ **VETO ACTIF: {decision.veto.value}**")
            lines.append("Le trade a été physiquement bloqué (Hard Kill).")
            lines.append("")

        lines.append("## Votes")
        for v in decision.verdict.votes:
            icon = "✅" if v.choice == VoteChoice.APPROVE else "❌"
            lines.append(f"  {icon} {v.role_name} ({v.expert}): {v.justification[:100]}")

        return "\n".join(lines)
