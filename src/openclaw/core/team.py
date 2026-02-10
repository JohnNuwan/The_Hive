"""
OpenClaw Agent Team
Part of Sovereign Stack V3.0

Orchestre plusieurs agents pour résoudre des tâches complexes.
Pattern principal : Plan-and-Execute.

Architecture :
    1. User Request → Planner → [Task 1, Task 2, Task 3]
    2. Pour chaque Task N :
        - Executor (avec Skills) → Exécute Task N → Result N
        - Mise à jour du contexte cumulé
    3. Final Answer → User

Intégration War Room :
    L'équipe peut convoquer une War Room (DEFCON) pour les décisions
    critiques nécessitant un débat contradictoire multi-experts.
"""

import logging
import re
from typing import List, Dict, Any, Optional

from .agent import OpenClawAgent
from .war_room import WarRoomSession, WarRoomVerdict
from .war_room_prompts import WarRoomType
from shared import ChatMessage, Role

logger = logging.getLogger(__name__)


class AgentTeam:
    """Équipe d'agents collaborant sur une tâche via Plan-and-Execute.

    L'équipe est composée d'un Planner (cerveau) et d'un Executor (bras).
    Le Planner décompose la requête en étapes atomiques, puis l'Executor
    les réalise séquentiellement en accumulant le contexte.

    Attributes:
        planner: Agent dédié à la planification.
        executor: Agent dédié à l'exécution des étapes.
    """

    def __init__(self):
        """Initialise l'équipe Planner + Executor."""
        self.planner = OpenClawAgent(
            name="Planner",
            role="planner",
            goal="Décomposer les requêtes complexes en étapes atomiques (To-Do List).",
        )
        self.executor = OpenClawAgent(
            name="Executor",
            role="coder",
            goal="Exécuter chaque étape avec précision en utilisant les Skills.",
        )

    async def run(self, user_request: str) -> str:
        """Exécute une requête complexe via l'équipe Plan-and-Execute.

        Workflow :
            1. Le Planner génère une liste d'étapes numérotées.
            2. L'Executor traite chaque étape séquentiellement.
            3. Le contexte cumulé est transmis entre les étapes.

        Args:
            user_request: La requête utilisateur à traiter.

        Returns:
            Les résultats de toutes les étapes, concaténés.
        """
        logger.info(f"[Team] New Request: {user_request}")

        # ─── Phase 1 : PLANIFICATION ───
        logger.info("[Team] Phase 1: Planning...")
        plan_response = await self.planner.run(
            f"Analyse cette demande et liste les étapes d'exécution "
            f"(Format: 1. ..., 2. ..., 3. ...): {user_request}"
        )

        steps = self._parse_plan(plan_response)
        logger.info(f"[Team] Generated {len(steps)} steps.")

        # ─── Phase 2 : EXÉCUTION ───
        logger.info("[Team] Phase 2: Execution...")
        context: List[str] = []
        final_results: List[str] = []

        for i, step in enumerate(steps, 1):
            logger.info(f"[Team] Executing Step {i}/{len(steps)}: {step}")

            # L'executor reçoit le contexte des étapes précédentes
            context_str = "\n".join(context) if context else "Première étape."
            step_prompt = f"Contexte précédent:\n{context_str}\n\nAction à réaliser: {step}"

            result = await self.executor.run(step_prompt)
            context.append(f"Step {i} Result: {result}")
            final_results.append(result)

        # ─── Phase 3 : SYNTHÈSE ───
        logger.info("[Team] Phase 3: Synthesis complete.")
        return "\n\n".join(final_results)

    async def convene_war_room(
        self,
        room_type: WarRoomType,
        subject: str,
        llm_service=None,
        memory_bridge=None,
    ) -> WarRoomVerdict:
        """Convoque une War Room pour un débat contradictoire.

        Cette méthode gèle les tâches courantes, instancie une session
        de War Room éphémère, exécute le débat DEFCON, et retourne le verdict.

        Args:
            room_type: Type de War Room (COUNCIL, DOJO, HIGH_COURT, QUIET_ROOM).
            subject: Sujet ou contexte du débat.
            llm_service: Service LLM (si None, utilise celui du planner).
            memory_bridge: MemoryBridge pour archivage (si None, utilise celui du planner).

        Returns:
            WarRoomVerdict contenant le résultat du débat.
        """
        logger.info(f"[Team] ⚔️ CONVOCATION WAR ROOM: {room_type.value}")

        # Utiliser les services du planner si non fournis
        llm = llm_service or self.planner.llm
        memory = memory_bridge or self.planner.memory

        session = WarRoomSession(room_type=room_type, subject=subject)
        verdict = await session.run_debate(llm_service=llm, memory_bridge=memory)

        if verdict.approved:
            logger.info(f"[Team] ✅ War Room verdict: APPROVED ({verdict.approval_score:.0%})")
        else:
            logger.warning(f"[Team] ❌ War Room verdict: REJECTED ({verdict.approval_score:.0%})")

        return verdict

    def _parse_plan(self, plan_text: str) -> List[str]:
        """Extrait les étapes numérotées d'un plan textuel.

        Supporte les formats suivants :
            - "1. Step one"  (numérotation)
            - "- Step one"   (tirets)
            - "* Step one"   (étoiles)

        BUG FIX: L'ancienne version ne matchait que "1." et ignorait
        les lignes "2.", "3.", etc. Corrigé avec une regex robuste.

        Args:
            plan_text: Le texte brut retourné par le Planner.

        Returns:
            Liste des étapes extraites. Si aucune n'est trouvée,
            retourne le texte entier comme étape unique (fallback).
        """
        steps: List[str] = []

        for line in plan_text.split("\n"):
            line = line.strip()
            if not line or len(line) < 5:
                continue

            # Match numérotation (1. , 2. , 10. , etc.)
            if re.match(r"^\d+\.\s", line):
                steps.append(line)
            # Match tirets et étoiles
            elif line.startswith("- ") or line.startswith("* "):
                steps.append(line)

        if not steps:
            # Fallback : toute la réponse est une seule étape
            logger.warning("[Team] Plan parsing failed, using full text as single step.")
            steps = [plan_text]

        return steps
