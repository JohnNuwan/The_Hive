"""
OpenClaw War Room Session Manager
Part of Sovereign Stack V3.0

Ce module implémente le système de débat éphémère ("War Room") décrit dans
le Cahier des Charges v3.0. Chaque War Room est un processus temporaire
(pas un service permanent) pour économiser la VRAM (6 Go RTX 2060).

Workflow DEFCON :
    1. Trigger   — Un agent détecte une anomalie ou opportunité à haut risque.
    2. Summoning — OpenClaw gèle les tâches de fond et convoque les experts.
    3. Debate    — 3 tours max (Thèse → Antithèse → Synthèse).
    4. Verdict   — Vote pondéré. Si Approval < 80%, l'action est avortée.
    5. Dissolution — Le résumé est archivé dans Mem0, la RAM est libérée.

Références :
    - CDcs "Module War Rooms" (THE HIVE v3.0)
    - Consensus Protocol (réduction des erreurs par débat contradictoire)
"""

import logging
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .war_room_prompts import (
    WarRoomType,
    WarRoomConfig,
    WarRoomRole,
    get_war_room_config,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class VoteChoice(Enum):
    """Choix de vote d'un participant lors du Verdict."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class DebateEntry:
    """Un argument soumis pendant un tour de débat.

    Attributes:
        role_name: Nom du rôle qui a parlé (ex: "Proposant").
        expert: Nom de l'expert E.V.A. (ex: "BANKER").
        content: Contenu de l'argument.
        round_number: Numéro du tour (1-based).
        timestamp: Horodatage de l'entrée.
    """

    role_name: str
    expert: str
    content: str
    round_number: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Vote:
    """Vote d'un participant après le débat.

    Attributes:
        role_name: Nom du rôle votant.
        expert: Nom de l'expert.
        choice: Choix (APPROVE / REJECT / ABSTAIN).
        weight: Poids du vote (défini par la config du rôle).
        justification: Raison du vote.
    """

    role_name: str
    expert: str
    choice: VoteChoice
    weight: float
    justification: str = ""


@dataclass
class WarRoomVerdict:
    """Résultat final d'une session de War Room.

    Attributes:
        session_id: Identifiant unique de la session.
        room_type: Type de War Room.
        approved: True si le seuil d'approbation est atteint.
        approval_score: Score d'approbation (0.0 → 1.0).
        votes: Liste des votes individuels.
        summary: Résumé du débat pour archivage Mem0.
        timestamp: Horodatage du verdict.
    """

    session_id: str
    room_type: WarRoomType
    approved: bool
    approval_score: float
    votes: List[Vote]
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# WAR ROOM SESSION
# ═══════════════════════════════════════════════════════════════════════════════


class WarRoomSession:
    """Gère une session de débat contradictoire entre agents.

    Cette classe orchestre le workflow DEFCON complet :
    invocation, tours de débat, vote pondéré, et dissolution.

    L'instance est conçue pour être éphémère : créée pour un débat,
    puis détruite après archivage du résumé dans Mem0.

    Attributes:
        session_id: Identifiant UUID unique de la session.
        config: Configuration de la War Room (type, rôles, seuils).
        subject: Sujet du débat.
        transcript: Liste chronologique de tous les arguments.
    """

    def __init__(self, room_type: WarRoomType, subject: str):
        """Initialise une nouvelle session de War Room.

        Args:
            room_type: Type de War Room à instancier (COUNCIL, DOJO, etc.).
            subject: Sujet ou contexte du débat (ex: "Trade XAUUSD 0.1 lot").
        """
        self.session_id: str = str(uuid.uuid4())[:8]
        self.config: WarRoomConfig = get_war_room_config(room_type)
        self.subject: str = subject
        self.transcript: List[DebateEntry] = []
        self._votes: List[Vote] = []

        logger.info(
            f"[WarRoom:{self.config.name}] Session {self.session_id} créée. "
            f"Sujet: {subject}"
        )

    async def run_debate(self, llm_service, memory_bridge=None) -> WarRoomVerdict:
        """Exécute le débat complet et retourne le verdict.

        Workflow :
            1. Pour chaque tour (max_rounds), chaque rôle prend la parole.
            2. Après les tours, chaque rôle vote (APPROVE/REJECT).
            3. Le score pondéré est calculé.
            4. Le résumé est archivé dans Mem0 si disponible.

        Args:
            llm_service: Instance du service LLM pour la génération.
            memory_bridge: Instance optionnelle du MemoryBridge pour archivage.

        Returns:
            WarRoomVerdict contenant le résultat du débat.
        """
        logger.info(
            f"[WarRoom:{self.config.name}] ═══ DÉBUT DU DÉBAT ═══ "
            f"({self.config.max_rounds} tours max)"
        )

        # ─── Phase 1 : Débat (Thèse → Antithèse → Synthèse) ───
        for round_num in range(1, self.config.max_rounds + 1):
            round_labels = {1: "THÈSE", 2: "ANTITHÈSE", 3: "SYNTHÈSE"}
            label = round_labels.get(round_num, f"TOUR {round_num}")
            logger.info(
                f"[WarRoom:{self.config.name}] ─── Tour {round_num}: {label} ───"
            )

            for role in self.config.roles:
                argument = await self._generate_argument(
                    llm_service, role, round_num
                )
                entry = DebateEntry(
                    role_name=role.name,
                    expert=role.expert,
                    content=argument,
                    round_number=round_num,
                )
                self.transcript.append(entry)
                logger.info(
                    f"[WarRoom:{self.config.name}] "
                    f"{role.name} ({role.expert}): {argument[:100]}..."
                )

        # ─── Phase 2 : Vote ───
        logger.info(f"[WarRoom:{self.config.name}] ─── VOTE ───")
        for role in self.config.roles:
            vote = await self._collect_vote(llm_service, role)
            self._votes.append(vote)
            logger.info(
                f"[WarRoom:{self.config.name}] "
                f"{role.name} vote: {vote.choice.value} "
                f"(poids: {vote.weight}x) — {vote.justification[:80]}"
            )

        # ─── Phase 3 : Calcul du verdict ───
        verdict = self._compute_verdict()

        # ─── Phase 4 : Archivage Mem0 ───
        if memory_bridge:
            try:
                await memory_bridge.add(
                    content=verdict.summary,
                    metadata={
                        "type": "war_room_verdict",
                        "room": self.config.name,
                        "session_id": self.session_id,
                        "approved": verdict.approved,
                    },
                )
                logger.info(
                    f"[WarRoom:{self.config.name}] "
                    f"Résumé archivé dans Mem0."
                )
            except Exception as e:
                logger.warning(
                    f"[WarRoom:{self.config.name}] "
                    f"Archivage Mem0 échoué: {e}"
                )

        status = "✅ APPROUVÉ" if verdict.approved else "❌ REJETÉ"
        logger.info(
            f"[WarRoom:{self.config.name}] ═══ VERDICT: {status} "
            f"(score: {verdict.approval_score:.0%}) ═══"
        )

        return verdict

    async def _generate_argument(
        self,
        llm_service,
        role: WarRoomRole,
        round_number: int,
    ) -> str:
        """Génère l'argument d'un participant pour un tour donné.

        Le prompt inclut le contexte du débat (arguments précédents)
        pour permettre une conversation cohérente.

        Args:
            llm_service: Service LLM pour la génération.
            role: Le rôle du participant.
            round_number: Le numéro du tour en cours.

        Returns:
            L'argument généré sous forme de texte.
        """
        # Construire le contexte avec les arguments précédents
        context_parts = [f"SUJET DU DÉBAT: {self.subject}\n"]

        for entry in self.transcript:
            context_parts.append(
                f"[Tour {entry.round_number}] {entry.role_name} "
                f"({entry.expert}): {entry.content}"
            )

        context = "\n".join(context_parts)

        # Import local pour éviter les dépendances circulaires
        from shared import ChatMessage, Role as MsgRole

        messages = [
            ChatMessage(
                role=MsgRole.USER,
                content=(
                    f"Contexte du débat (Tour {round_number}):\n{context}\n\n"
                    f"C'est ton tour de parler en tant que {role.name}. "
                    f"Sois concis (3-5 phrases max)."
                ),
            )
        ]

        response, _ = await llm_service.generate_response(
            messages=messages,
            system_prompt=role.system_prompt,
            role="general",
            max_tokens=500,
            temperature=0.7,
        )
        return response

    async def _collect_vote(
        self,
        llm_service,
        role: WarRoomRole,
    ) -> Vote:
        """Collecte le vote d'un participant après le débat.

        Le participant reçoit l'intégralité du transcript et doit
        voter APPROVE, REJECT, ou ABSTAIN avec une justification.

        Args:
            llm_service: Service LLM pour la génération.
            role: Le rôle du votant.

        Returns:
            Vote contenant le choix, le poids, et la justification.
        """
        transcript_text = "\n".join(
            f"[Tour {e.round_number}] {e.role_name}: {e.content}"
            for e in self.transcript
        )

        from shared import ChatMessage, Role as MsgRole

        messages = [
            ChatMessage(
                role=MsgRole.USER,
                content=(
                    f"TRANSCRIPT DU DÉBAT:\n{transcript_text}\n\n"
                    f"En tant que {role.name} ({role.expert}), "
                    f"vote APPROVE ou REJECT avec une justification courte. "
                    f"Réponds EXACTEMENT au format:\n"
                    f"VOTE: [APPROVE/REJECT]\n"
                    f"RAISON: [ta justification]"
                ),
            )
        ]

        response, _ = await llm_service.generate_response(
            messages=messages,
            system_prompt=role.system_prompt,
            role="general",
            max_tokens=200,
            temperature=0.3,  # Basse pour un vote déterministe
        )

        # Parser le vote
        choice = VoteChoice.ABSTAIN
        justification = response

        response_upper = response.upper()
        if "APPROVE" in response_upper:
            choice = VoteChoice.APPROVE
        elif "REJECT" in response_upper:
            choice = VoteChoice.REJECT

        # Extraire la justification
        if "RAISON:" in response:
            justification = response.split("RAISON:")[-1].strip()

        return Vote(
            role_name=role.name,
            expert=role.expert,
            choice=choice,
            weight=role.weight,
            justification=justification,
        )

    def _compute_verdict(self) -> WarRoomVerdict:
        """Calcule le verdict final à partir des votes pondérés.

        Formule : score = Σ(poids_approve) / Σ(poids_total)
        L'action est approuvée si score ≥ approval_threshold (80% par défaut).

        Returns:
            WarRoomVerdict avec le score, le résumé, et le statut.
        """
        total_weight = 0.0
        approve_weight = 0.0

        for vote in self._votes:
            if vote.choice != VoteChoice.ABSTAIN:
                total_weight += vote.weight
            if vote.choice == VoteChoice.APPROVE:
                approve_weight += vote.weight

        # Éviter la division par zéro
        score = approve_weight / total_weight if total_weight > 0 else 0.0
        approved = score >= self.config.approval_threshold

        # Générer le résumé
        summary = (
            f"[{self.config.name}] Session {self.session_id} — "
            f"Sujet: {self.subject} — "
            f"Verdict: {'APPROUVÉ' if approved else 'REJETÉ'} "
            f"({score:.0%}). "
            f"Votes: {', '.join(f'{v.role_name}={v.choice.value}' for v in self._votes)}"
        )

        return WarRoomVerdict(
            session_id=self.session_id,
            room_type=self.config.type,
            approved=approved,
            approval_score=score,
            votes=self._votes,
            summary=summary,
        )

    def get_transcript_text(self) -> str:
        """Retourne le transcript complet du débat sous forme lisible.

        Returns:
            Texte formaté de tous les arguments par tour.
        """
        lines = [f"═══ {self.config.name} — Session {self.session_id} ═══"]
        lines.append(f"Sujet: {self.subject}\n")

        current_round = 0
        for entry in self.transcript:
            if entry.round_number != current_round:
                current_round = entry.round_number
                labels = {1: "THÈSE", 2: "ANTITHÈSE", 3: "SYNTHÈSE"}
                label = labels.get(current_round, f"TOUR {current_round}")
                lines.append(f"\n─── Tour {current_round}: {label} ───")
            lines.append(f"  [{entry.role_name}] {entry.content}")

        return "\n".join(lines)
