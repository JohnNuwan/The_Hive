"""
OpenClaw Agent Core
Part of Sovereign Stack V3.0

Ce module définit le cerveau de l'agent (Cognitive Kernel).
Il implémente la boucle OODA (Observe, Orient, Decide, Act).
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from shared import ChatMessage, Role
from shared.memory_bridge import get_memory_bridge
from eva_core.services.llm import get_llm_service

logger = logging.getLogger(__name__)

class OpenClawAgent:
    """
    Agent autonome capable d'utiliser des outils (Skills) et de planifier.
    """
    
    def __init__(
        self, 
        name: str = "OpenClaw", 
        role: str = "general",
        goal: str = "T'améliorer continuellement.",
        tools: List[Any] = None
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.tools = tools or []
        
        self.memory = get_memory_bridge()
        self.llm = get_llm_service()
        
        self.short_term_history: List[ChatMessage] = []

    async def run(self, input_task: str):
        """Lance la boucle d'exécution sur une tâche donnée"""
        logger.info(f"[{self.name}] Starting task: {input_task}")
        
        # 1. OBSERVE : Récupération du contexte
        context = await self.memory.search(input_task, limit=3)
        context_str = "\n".join(context) if context else "Aucun contexte préalable."
        
        # Initialisation de l'historique de session
        self.short_term_history.append(ChatMessage(role=Role.USER, content=f"Tâche : {input_task}\nContexte: {context_str}"))

        # 2. ORIENT : Analyse de la situation (Planification)
        plan = await self._plan(input_task)
        logger.info(f"[{self.name}] Plan: {plan}")
        
        # 3. DECIDE & ACT : Exécution pas à pas
        # Pour ce MVP, on fait une simple génération de réponse basée sur le plan
        # TODO: Implémenter la boucle ReAct (Reasoning + Acting) avec appel d'outils
        
        response, thoughts = await self.llm.generate_response(
            messages=self.short_term_history,
            system_prompt=f"Tu es {self.name}. Ton but est : {self.goal}.\nPlan d'action : {plan}",
            role=self.role
        )
        
        if thoughts:
            logger.info(f"[{self.name}] Thoughts: {thoughts}")
            
        logger.info(f"[{self.name}] Final Response: {response}")
        return response

    async def _plan(self, task: str) -> str:
        """Génère un plan d'action de haut niveau"""
        messages = [
            ChatMessage(role=Role.SYSTEM, content="Tu es un planificateur expert. Décompose la tâche user en étapes claires."),
            ChatMessage(role=Role.USER, content=f"Tâche: {task}")
        ]
        response, _ = await self.llm.generate_response(messages, role="planner")
        return response
