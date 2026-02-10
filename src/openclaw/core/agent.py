"""
OpenClaw Agent Core
Part of Sovereign Stack V3.0

Ce module définit le cerveau de l'agent (Cognitive Kernel).
Il implémente la boucle OODA (Observe, Orient, Decide, Act).

L'agent est la brique fondamentale de l'architecture OpenClaw.
Chaque agent possède :
    - Un rôle (general, planner, coder, etc.)
    - Un objectif (goal)
    - Un accès à la mémoire (Mem0 / Neo4j via MemoryBridge)
    - Un accès au LLM (vLLM / Ollama via LLMService)
    - Un historique de session (short-term memory)

Références :
    - Boyd's OODA Loop (Observe, Orient, Decide, Act)
    - ReAct (Reasoning + Acting) — à implémenter en v2
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from shared import ChatMessage, Role
from shared.memory_bridge import get_memory_bridge
from eva_core.services.llm import get_llm_service

logger = logging.getLogger(__name__)


class OpenClawAgent:
    """Agent autonome capable d'utiliser des outils (Skills) et de planifier.

    Chaque agent est un participant du système multi-agent OpenClaw.
    Il peut être utilisé seul (via `run()`) ou dans une équipe (`AgentTeam`)
    ou une War Room (`WarRoomSession`).

    Attributes:
        name: Nom de l'agent (ex: "Planner", "Executor", "BANKER").
        role: Rôle pour le Council (model swapping) — "general", "coder", "planner".
        goal: Objectif de haut niveau injecté dans le system prompt.
        tools: Liste des Skills disponibles pour cet agent.
        memory: Pont mémoire (Mem0 + Neo4j).
        llm: Service LLM (vLLM / Ollama).
        short_term_history: Historique des échanges de la session courante.
    """

    def __init__(
        self,
        name: str = "OpenClaw",
        role: str = "general",
        goal: str = "T'améliorer continuellement.",
        tools: Optional[List[Any]] = None,
    ):
        """Initialise un agent OpenClaw.

        Args:
            name: Nom d'affichage de l'agent.
            role: Rôle pour le Model Swapping (Council).
            goal: Objectif injecté dans le system prompt.
            tools: Liste optionnelle de Skills disponibles.
        """
        self.name = name
        self.role = role
        self.goal = goal
        self.tools = tools or []

        self.memory = get_memory_bridge()
        self.llm = get_llm_service()

        # BUG FIX: L'historique est maintenant réinitialisé à chaque appel run()
        self.short_term_history: List[ChatMessage] = []

    async def run(self, input_task: str) -> str:
        """Lance la boucle OODA sur une tâche donnée.

        Workflow :
            1. OBSERVE  — Récupère le contexte mémoriel pertinent.
            2. ORIENT   — Génère un plan d'action via le LLM.
            3. DECIDE   — Sélectionne la meilleure approche.
            4. ACT      — Exécute et retourne la réponse.

        Args:
            input_task: Description textuelle de la tâche à réaliser.

        Returns:
            La réponse finale générée par l'agent.
        """
        logger.info(f"[{self.name}] Starting task: {input_task}")

        # Réinitialiser l'historique pour chaque nouvelle tâche
        # (évite l'accumulation de contexte entre appels successifs)
        self.short_term_history = []

        # 1. OBSERVE : Récupération du contexte
        context = await self._observe(input_task)

        # Initialisation de l'historique de session
        self.short_term_history.append(
            ChatMessage(
                role=Role.USER,
                content=f"Tâche : {input_task}\nContexte: {context}",
            )
        )

        # 2. ORIENT : Analyse de la situation (Planification)
        plan = await self._plan(input_task)
        logger.info(f"[{self.name}] Plan: {plan}")

        # 3. DECIDE & ACT : Exécution
        # TODO: Implémenter la boucle ReAct (Reasoning + Acting) avec appel d'outils
        response, thoughts = await self.llm.generate_response(
            messages=self.short_term_history,
            system_prompt=f"Tu es {self.name}. Ton but est : {self.goal}.\nPlan d'action : {plan}",
            role=self.role,
        )

        if thoughts:
            logger.info(f"[{self.name}] Thoughts: {thoughts}")

        logger.info(f"[{self.name}] Final Response: {response}")
        return response

    async def _observe(self, task: str) -> str:
        """Phase OBSERVE : récupère le contexte mémoriel pertinent.

        Interroge le MemoryBridge (Mem0 + Neo4j) pour trouver
        les souvenirs les plus proches de la tâche courante.

        Args:
            task: La tâche pour laquelle chercher du contexte.

        Returns:
            Chaîne de contexte formatée, ou message par défaut.
        """
        try:
            context = await self.memory.search(task, limit=3)
            return "\n".join(context) if context else "Aucun contexte préalable."
        except Exception as e:
            logger.warning(f"[{self.name}] Memory search failed: {e}")
            return "Aucun contexte préalable (erreur mémoire)."

    async def _plan(self, task: str) -> str:
        """Phase ORIENT : génère un plan d'action de haut niveau.

        Utilise le LLM en mode "planner" pour décomposer la tâche
        en étapes claires et actionnables.

        Args:
            task: La tâche à planifier.

        Returns:
            Plan textuel généré par le LLM.
        """
        messages = [
            ChatMessage(
                role=Role.SYSTEM,
                content="Tu es un planificateur expert. Décompose la tâche user en étapes claires.",
            ),
            ChatMessage(role=Role.USER, content=f"Tâche: {task}"),
        ]
        response, _ = await self.llm.generate_response(messages, role="planner")
        return response

    async def participate_in_debate(
        self,
        system_prompt: str,
        debate_context: str,
        round_number: int,
    ) -> str:
        """Participe à un tour de débat dans une War Room.

        L'agent reçoit un system prompt spécifique à son rôle dans le débat
        (ex: "Tu es l'opposant") et doit argumenter en contexte.

        Args:
            system_prompt: L'instruction système du rôle assigné.
            debate_context: Le contexte complet du débat (arguments précédents).
            round_number: Numéro du tour en cours.

        Returns:
            L'argument de l'agent pour ce tour.
        """
        messages = [
            ChatMessage(
                role=Role.USER,
                content=(
                    f"Contexte du débat (Tour {round_number}):\n{debate_context}\n\n"
                    f"C'est ton tour de parler. Sois concis (3-5 phrases max)."
                ),
            )
        ]

        response, _ = await self.llm.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            role="general",
            max_tokens=500,
            temperature=0.7,
        )
        return response
