"""
OpenClaw Agent Team
Part of Sovereign Stack V3.0

Orchestration d'équipe : Planner -> Executor.
"""

import logging
import asyncio
from typing import List, Dict, Any

from .agent import OpenClawAgent
from shared import ChatMessage, Role

logger = logging.getLogger(__name__)

class AgentTeam:
    """
    Equipe d'agents collaborant sur une tâche.
    Pattern: Plan-and-Execute.
    """
    
    def __init__(self, use_mocks: bool = False):
        self.planner = OpenClawAgent(
            name="Planner", 
            role="planner", 
            goal="Décomposer les requêtes complexes en étapes atomiques (To-Do List)."
        )
        self.executor = OpenClawAgent(
            name="Executor", 
            role="coder", # ou "general" selon besoin
            goal="Exécuter chaque étape avec précision en utilisant les Skills."
        )
        # TODO: Support mocks injection if needed provided by init
        
    async def run(self, user_request: str) -> str:
        """Exécute une requête via l'équipe"""
        logger.info(f"[Team] New Request: {user_request}")
        
        # 1. PLANIFICATION
        logger.info("[Team] Phase 1: Planning...")
        # On demande au planner de faire une liste
        plan_response = await self.planner.run(
            f"Analyse cette demande et liste les étapes d'exécution (Format: 1. ..., 2. ...): {user_request}"
        )
        
        # Parsing basique du plan (à améliorer avec Structured Output)
        steps = self._parse_plan(plan_response)
        logger.info(f"[Team] Generated {len(steps)} steps.")
        
        # 2. EXECUTION
        logger.info("[Team] Phase 2: Execution...")
        context = []
        final_results = []
        
        for i, step in enumerate(steps):
            logger.info(f"[Team] Executing Step {i+1}: {step}")
            
            # L'executor reçoit le contexte des étapes précédentes
            context_str = "\n".join(context)
            step_prompt = f"Contexte précédent: {context_str}\n\nAction à réaliser: {step}"
            
            result = await self.executor.run(step_prompt)
            context.append(f"Step {i+1} Result: {result}")
            final_results.append(result)
            
        # 3. SYNTHESE (Optionnel, ou retour du dernier résultat)
        return "\n\n".join(final_results)

    def _parse_plan(self, plan_text: str) -> List[str]:
        """Extrait les lignes numérotées comme étapes"""
        steps = []
        for line in plan_text.split('\n'):
            line = line.strip()
            # Détection simple "1. " ou "- "
            if (line.startswith("1.") or line.startswith("-") or line.startswith("*")) and len(line) > 5:
                 steps.append(line)
        
        if not steps:
            # Fallback: toute la réponse est une seule étape
            steps = [plan_text]
            
        return steps
