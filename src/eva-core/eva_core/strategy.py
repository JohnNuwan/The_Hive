"""
Strategy Orchestrator — THE HIVE
Deep logic for Mixture of Experts (MoE) routing and intention analysis.
"""

import logging
from typing import Any
from shared import Intent, IntentType
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
        Return your decision in JSON format:
        {{
            "intent_type": "TRADE|INTEL|RESEARCH|MANAGEMENT|CHAT",
            "target_expert": "expert_name",
            "confidence": 0.0-1.0,
            "entities": {{"key": "value"}}
        }}
        """

        try:
            # We use the LLM to perform the high-level semantic routing
            # This is much more 'divine' than simple keyword matching
            response = await self.llm.generate_response(
                messages=[{"role": "user", "content": message}],
                system_prompt=system_prompt,
                json_mode=True
            )
            
            import json
            data = json.loads(response)
            
            return Intent(
                intent_type=IntentType(data.get("intent_type", "CHAT")),
                target_expert=data.get("target_expert", "core"),
                confidence=float(data.get("confidence", 0.5)),
                entities=data.get("entities", {})
            )
            
        except Exception as e:
            logger.error(f"Strategy Orchestration failed: {e}. Falling back to default routing.")
            return Intent(intent_type=IntentType.CHAT, target_expert="core", confidence=0.1)

    def _format_manifest(self) -> str:
        return "\n".join([f"- {name}: {desc}" for name, desc in self.experts_manifest.items()])
