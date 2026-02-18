"""
Strategy Orchestrator — THE HIVE
Deep logic for Mixture of Experts (MoE) routing and intention analysis.
"""

import logging
import uuid
from typing import Any
from shared import Intent, IntentType, ChatMessage, MessageRole
from eva_core.services.llm import get_llm_service

logger = logging.getLogger(__name__)

class StrategyOrchestrator:
    """
    Advanced orchestrator that decides which Expert(s) should handle a request.
    Uses semantic analysis and system state to optimize the 'Swarm' response.
    """

    def __init__(self):
        self.llm = get_llm_service()
        self.experts_manifest = {
            "banker": "Financial trades, risk management, account equity, and broker integration.",
            "sentinel": "System health, hardware metrics, security alerts, and institutional sentiment.",
            "shadow": "OSINT, investigation, information retrieval, and reconnaissance.",
            "accountant": "Corporate accounting, tax compliance (URSSAF), and invoice management.",
            "rwa": "Real World Assets, tokenization, and bridging physical assets to the blockchain.",
            "lab": "Research and development, experimental code, and data science simulations.",
            "compliance": "Legal regulations, KYC/AML, and corporate documentation.",
            "substrate": "Energy management, circadian rhythm optimization, and lifestyle automation."
        }

    async def route_request(self, message: str, history: list = None) -> Intent:
        """
        Analyzes the message and returns a high-confidence Intent with a target Expert.
        """
        logger.info(f"Orchestrating strategy for: {message[:50]}...")

        # Construct a prompt for the 'Orchestrator' persona
        system_prompt = f"""
        You are the THE HIVE Strategy Orchestrator. 
        Your job is to route user requests to the most appropriate Expert in the Mixture of Experts (MoE) cluster.
        
        EXPERTS MANIFEST:
        {self._format_manifest()}

        Classify the user intent and choose the target expert.

        Intent Types:
        - TRADING_ORDER: Buying/selling assets.
        - POSITION_STATUS: Checking positions.
        - RISK_INQUIRY: Risk management questions.
        - GENERAL_CHAT: General conversation.
        - MEMORY_RECALL: Asking about past events.
        - OSINT_REQUEST: Information gathering.
        - SYSTEM_COMMAND: System control.
        - SECURITY_ALERT: Security issues.

        Return your decision in JSON format:
        {{
            "intent_type": "TRADING_ORDER|GENERAL_CHAT|...",
            "target_expert": "expert_name",
            "confidence": 0.0-1.0,
            "entities": {{"key": "value"}}
        }}
        """

        try:
            # Construct messages properly
            messages = [
                ChatMessage(
                    session_id=uuid.uuid4(),
                    role=MessageRole.USER,
                    content=message
                )
            ]

            # We use the LLM to perform the high-level semantic routing
            # This is much more 'divine' than simple keyword matching
            response_tuple = await self.llm.generate_response(
                messages=messages,
                system_prompt=system_prompt,
            )
            
            # Unpack response (response_text, thoughts)
            response_text = response_tuple[0] if isinstance(response_tuple, tuple) else response_tuple

            # Clean markdown code blocks if present
            clean_response = response_text.strip()
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()

            import json
            data = json.loads(clean_response)

            intent_val = data.get("intent_type", "GENERAL_CHAT")
            try:
                intent_type = IntentType(intent_val)
            except ValueError:
                intent_type = IntentType.GENERAL_CHAT
            
            return Intent(
                intent_type=intent_type,
                target_expert=data.get("target_expert", "core"),
                confidence=float(data.get("confidence", 0.5)),
                entities=data.get("entities", {})
            )
            
        except Exception as e:
            logger.error(f"Strategy Orchestration failed: {e}. Falling back to default routing.")
            return Intent(intent_type=IntentType.GENERAL_CHAT, target_expert="core", confidence=0.1)

    def _format_manifest(self) -> str:
        return "\n".join([f"- {name}: {desc}" for name, desc in self.experts_manifest.items()])
