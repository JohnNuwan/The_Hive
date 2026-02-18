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
import re
from typing import List, Dict, Any, Optional
from uuid import uuid4

from shared import ChatMessage, MessageRole as Role
from shared.memory_bridge import get_memory_bridge
from eva_core.services.llm import get_llm_service
from openclaw.skills.registry import SKILL_REGISTRY, get_skill, load_all_skills

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

        # Assure que tous les skills sont chargés si nécessaire
        if not SKILL_REGISTRY:
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
        self.short_term_history = []
        session_id = uuid4()

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
        plan = await self._plan(input_task)
        logger.info(f"[{self.name}] Plan: {plan}")

        # 3. DECIDE & ACT : Exécution (ReAct Loop)
        tool_desc = self._get_tool_descriptions()
        system_prompt = (
            f"Tu es {self.name}. Ton but est : {self.goal}.\n"
            f"Plan d'action : {plan}\n\n"
            f"You have access to the following tools:\n{tool_desc}\n\n"
            "To use a tool, please use the following format:\n"
            "Thought: Do I need to use a tool? Yes\n"
            "Action: [The name of the tool to use]\n"
            "Action Input: [The input to the action]\n"
            "Observation: [The result of the tool]\n\n"
            "When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n"
            "Thought: Do I need to use a tool? No\n"
            "Final Answer: [your response here]"
        )

        max_loops = 5
        final_response = "I couldn't complete the task within the limit."

        for i in range(max_loops):
            logger.info(f"[{self.name}] ReAct Loop {i+1}/{max_loops}")

            response, thoughts = await self.llm.generate_response(
                messages=self.short_term_history,
                system_prompt=system_prompt,
                role=self.role,
            )

            # Si des pensées sont retournées séparément (par ex. <thought>...), on les logue
            if thoughts:
                logger.info(f"[{self.name}] Thoughts: {thoughts}")

            # Reconstruire le contenu complet pour l'historique si nécessaire
            full_response_content = response
            if thoughts:
                full_response_content = f"<thought>{thoughts}</thought>\n{response}"

            self.short_term_history.append(
                ChatMessage(
                    session_id=session_id,
                    role=Role.ASSISTANT,
                    content=full_response_content
                )
            )

            # Parsing de la réponse ReAct
            # On cherche "Action:" et "Action Input:"
            # Ou "Final Answer:"

            # Note: On utilise re.DOTALL pour capturer sur plusieurs lignes si besoin
            final_answer_match = re.search(r"Final Answer:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
            action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", response, re.IGNORECASE)
            # Input peut être multiline, on prend jusqu'à la fin ou jusqu'à "Observation:"
            input_match = re.search(r"Action Input:\s*(.*)", response, re.IGNORECASE | re.DOTALL)

            if final_answer_match:
                final_response = final_answer_match.group(1).strip()
                break

            if action_match:
                tool_name = action_match.group(1).strip()
                # Nettoyer l'input (enlever d'éventuels Observation: qui auraient été générés par le LLM par erreur)
                raw_input = input_match.group(1).strip() if input_match else ""
                # Si l'input contient "Observation:", on coupe avant
                if "Observation:" in raw_input:
                    raw_input = raw_input.split("Observation:")[0].strip()

                tool_input = raw_input

                logger.info(f"[{self.name}] Executing Tool: {tool_name} with Input: {tool_input}")
                observation = await self._execute_tool(tool_name, tool_input)
                logger.info(f"[{self.name}] Observation: {observation}")

                # Ajout de l'observation à l'historique pour le prochain tour
                self.short_term_history.append(
                    ChatMessage(
                        session_id=session_id,
                        role=Role.USER,
                        content=f"Observation: {observation}"
                    )
                )
            else:
                # Pas d'action, pas de final answer explicite
                # Si la réponse ne semble pas être structurée ReAct, on assume que c'est la réponse finale
                if "Action:" not in response and "Final Answer:" not in response:
                    final_response = response
                    break

        logger.info(f"[{self.name}] Final Response: {final_response}")
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

    async def _plan(self, task: str) -> str:
        """Phase ORIENT : génère un plan d'action de haut niveau.

        Utilise le LLM en mode "planner" pour décomposer la tâche
        en étapes claires et actionnables.

        Args:
            task: La tâche à planifier.

        Returns:
            Plan textuel généré par le LLM.
        """
        session_id = uuid4()
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
        session_id = uuid4()
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

    def _get_tool_descriptions(self) -> str:
        """Génère la description textuelle des outils disponibles."""
        if not self.tools:
            return "No tools available."

        descriptions = []
        for tool_spec in self.tools:
            try:
                # Si c'est juste le nom (str)
                if isinstance(tool_spec, str):
                    tool_func = get_skill(tool_spec)
                    name = tool_spec
                    desc = tool_func._skill_description
                # Si c'est déjà une fonction/callable
                elif callable(tool_spec):
                    name = getattr(tool_spec, "_skill_name", tool_spec.__name__)
                    desc = getattr(tool_spec, "_skill_description", tool_spec.__doc__)
                else:
                    name = str(tool_spec)
                    desc = "Unknown tool"

                descriptions.append(f"- {name}: {desc}")
            except KeyError:
                logger.warning(f"Tool '{tool_spec}' not found in registry.")

        return "\n".join(descriptions)

    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Exécute un outil par son nom avec l'input donné."""
        try:
            # Nettoyage basique des quotes
            tool_input = tool_input.strip().strip('"').strip("'")

            tool_func = get_skill(tool_name)

            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(tool_input)
            else:
                # Pour les fonctions bloquantes, on pourrait utiliser run_in_executor
                # mais ici les skills sont supposés rapides ou safe.
                result = tool_func(tool_input)

            return str(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error executing tool '{tool_name}': {str(e)}"
