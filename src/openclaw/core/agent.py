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
import uuid
from typing import List, Dict, Any, Optional

from shared import ChatMessage, Role
from shared.memory_bridge import get_memory_bridge
from eva_core.services.llm import get_llm_service
from openclaw.skills.registry import get_skill, get_available_skills, load_all_skills

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

        # Charger tous les skills disponibles dans le registre si des outils sont demandés
        if self.tools:
            load_all_skills()

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

        # Générer un session_id pour cette exécution
        session_id = uuid.uuid4()

        # 1. OBSERVE : Récupération du contexte
        context = await self._observe(input_task)

        # Initialisation de l'historique de session
        self.short_term_history.append(
            ChatMessage(
                session_id=session_id,
                role=Role.USER,
                content=f"Tâche : {input_task}\nContexte: {context}",
            )
        )

        # 2. ORIENT : Analyse de la situation (Planification)
        plan = await self._plan(input_task, session_id)
        logger.info(f"[{self.name}] Plan: {plan}")

        # 3. DECIDE & ACT : Exécution
        # Boucle ReAct (Reasoning + Acting)
        max_iterations = 5
        final_response = "Désolé, je n'ai pas pu aboutir à une conclusion."

        # Construction du prompt système avec les outils
        tools_desc = self._get_tools_description()

        system_prompt = (
            f"Tu es {self.name}. Ton but est : {self.goal}.\n"
            f"Plan d'action : {plan}\n\n"
            f"Tu as accès aux outils suivants :\n{tools_desc}\n\n"
            "Utilise le format suivant pour raisonner et agir :\n"
            "Question: la tâche à accomplir\n"
            "Thought: je dois réfléchir à ce que je dois faire\n"
            "Action: le nom de l'outil à utiliser (parmi la liste ci-dessus)\n"
            "Action Input: l'entrée pour l'outil\n"
            "Observation: le résultat de l'outil\n"
            "... (répète Thought/Action/Action Input/Observation N fois)\n"
            "Thought: je connais maintenant la réponse finale\n"
            "Final Answer: la réponse finale à la tâche originale\n\n"
            "Si tu n'as pas besoin d'outils, donne directement la Final Answer."
        )

        for i in range(max_iterations):
            logger.info(f"[{self.name}] ReAct Iteration {i+1}/{max_iterations}")

            response, thoughts = await self.llm.generate_response(
                messages=self.short_term_history,
                system_prompt=system_prompt,
                role=self.role,
            )

            # Log thoughts
            if thoughts:
                logger.info(f"[{self.name}] Thoughts: {thoughts}")

            # Parsing de la réponse pour détecter une action
            action_line = None
            action_input_line = None
            lines = response.split('\n')

            for line in lines:
                if line.startswith("Action:"):
                    action_line = line
                elif line.startswith("Action Input:"):
                    action_input_line = line
                    # On suppose que Action Input suit Action assez rapidement
                    # Si on a déjà trouvé Action, on break ici
                    if action_line:
                        break

            if action_line and action_input_line:
                tool_name = action_line.replace("Action:", "").strip()
                tool_input = action_input_line.replace("Action Input:", "").strip()

                logger.info(f"[{self.name}] Tool Call: {tool_name}('{tool_input}')")

                # Exécuter l'outil
                observation = await self._execute_tool(tool_name, tool_input)
                logger.info(f"[{self.name}] Observation: {observation[:100]}...") # Log tronqué

                # Ajouter l'échange à l'historique
                self.short_term_history.append(
                    ChatMessage(
                        session_id=session_id,
                        role=Role.ASSISTANT,
                        content=response,
                        thoughts=thoughts
                    )
                )

                self.short_term_history.append(
                    ChatMessage(
                        session_id=session_id,
                        role=Role.USER,
                        content=f"Observation: {observation}"
                    )
                )

            elif "Final Answer:" in response:
                # Extraction de la réponse finale
                final_parts = response.split("Final Answer:")
                final_response = final_parts[-1].strip()
                logger.info(f"[{self.name}] Final Answer found.")

                self.short_term_history.append(
                    ChatMessage(
                        session_id=session_id,
                        role=Role.ASSISTANT,
                        content=response,
                        thoughts=thoughts
                    )
                )
                return final_response
            else:
                # Pas d'action ni de Final Answer explicite.
                # On considère que c'est la réponse finale.
                logger.info(f"[{self.name}] No action detected, assuming final response.")
                self.short_term_history.append(
                    ChatMessage(
                        session_id=session_id,
                        role=Role.ASSISTANT,
                        content=response,
                        thoughts=thoughts
                    )
                )
                return response

        logger.warning(f"[{self.name}] ReAct loop limit reached.")
        return final_response

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

    async def _plan(self, task: str, session_id: uuid.UUID = None) -> str:
        """Phase ORIENT : génère un plan d'action de haut niveau.

        Utilise le LLM en mode "planner" pour décomposer la tâche
        en étapes claires et actionnables.

        Args:
            task: La tâche à planifier.
            session_id: ID de session pour les messages.

        Returns:
            Plan textuel généré par le LLM.
        """
        if session_id is None:
            session_id = uuid.uuid4()

        messages = [
            ChatMessage(
                session_id=session_id,
                role=Role.SYSTEM,
                content="Tu es un planificateur expert. Décompose la tâche user en étapes claires.",
            ),
            ChatMessage(
                session_id=session_id,
                role=Role.USER,
                content=f"Tâche: {task}"
            ),
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
        session_id = uuid.uuid4()
        messages = [
            ChatMessage(
                session_id=session_id,
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

    def _get_tools_description(self) -> str:
        """Retourne la description textuelle des outils disponibles."""
        if not self.tools:
            return "Aucun outil disponible."

        descriptions = []
        for tool_name in self.tools:
            if isinstance(tool_name, str):
                try:
                    # Récupérer la fonction du skill
                    skill_func = get_skill(tool_name)
                    # Utiliser _skill_description injecté par le décorateur @skill
                    desc = getattr(skill_func, "_skill_description", "Pas de description.")
                    descriptions.append(f"- {tool_name}: {desc}")
                except KeyError:
                    logger.warning(f"Tool '{tool_name}' not found in registry.")
            else:
                # Si c'est déjà un callable
                name = getattr(tool_name, "_skill_name", str(tool_name))
                desc = getattr(tool_name, "_skill_description", "Pas de description.")
                descriptions.append(f"- {name}: {desc}")

        return "\n".join(descriptions)

    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Exécute un outil par son nom et retourne le résultat."""
        try:
            skill_func = get_skill(tool_name)
            # Vérifier si la fonction est asynchrone
            if asyncio.iscoroutinefunction(skill_func):
                result = await skill_func(tool_input)
            else:
                # Exécuter dans un thread pour ne pas bloquer si c'est sync et long
                result = await asyncio.to_thread(skill_func, tool_input)
            return str(result)
        except KeyError:
            return f"Error: Tool '{tool_name}' not found."
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"
