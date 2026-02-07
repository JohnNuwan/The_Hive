"""
Routeur d'Intent - Classification et Routage LangGraph
Classe les intentions utilisateur et route vers l'expert approprié
"""

import logging
import re
from typing import Any

from shared import Intent, IntentType

logger = logging.getLogger(__name__)


# Patterns de détection d'intent (version simplifiée, à remplacer par LLM)
INTENT_PATTERNS: dict[IntentType, list[str]] = {
    IntentType.TRADING_ORDER: [
        r"(achète|acheter|buy|vends|vendre|sell|ouvre|ferme|close)",
        r"(lot|lots|position)",
        r"(gold|xauusd|eurusd|gbpusd|nasdaq|us30)",
    ],
    IntentType.POSITION_STATUS: [
        r"(position|positions|trades?|ordres?)",
        r"(statut|status|état|ouverte?s?)",
        r"(profit|perte|p&l|pnl)",
    ],
    IntentType.RISK_INQUIRY: [
        r"(risque|risk|drawdown|dd)",
        r"(limite|limit|max)",
        r"(anti.?tilt|kill.?switch)",
    ],
    IntentType.MEMORY_RECALL: [
        r"(rappelle|souviens|mémoire|historique)",
        r"(rappel|dernier|précédent|hier|semaine)",
    ],
    IntentType.OSINT_REQUEST: [
        r"(recherche|trouve|scan|osint)",
        r"(information|info|données|data)",
    ],
    IntentType.SECURITY_ALERT: [
        r"(sécurité|securite|alerte|menace)",
        r"(intrusion|attaque|hack)",
    ],
    IntentType.SYSTEM_COMMAND: [
        r"(système|system|config|configuration)",
        r"(redémarre|restart|stop|arrête)",
    ],
}

# Mapping intent -> expert cible
INTENT_TO_EXPERT: dict[IntentType, str] = {
    IntentType.TRADING_ORDER: "banker",
    IntentType.POSITION_STATUS: "banker",
    IntentType.RISK_INQUIRY: "banker",
    IntentType.OSINT_REQUEST: "shadow",
    IntentType.SECURITY_ALERT: "sentinel",
    IntentType.SYSTEM_COMMAND: "keeper",
    IntentType.MEMORY_RECALL: "core",
    IntentType.GENERAL_CHAT: "core",
}


class IntentRouter:
    """
    Routeur d'Intent pour EVA Core.
    
    En mode simple, utilise des patterns regex.
    En mode avancé (production), utilise le LLM pour classification.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        logger.info(f"IntentRouter initialisé (use_llm={use_llm})")

    async def classify(self, text: str) -> Intent:
        """
        Classifie l'intention de l'utilisateur.
        
        Args:
            text: Message de l'utilisateur
            
        Returns:
            Intent avec type, confiance et entités extraites
        """
        text_lower = text.lower()

        if self.use_llm:
            return await self._classify_with_llm(text)

        return self._classify_with_patterns(text_lower)

    def _classify_with_patterns(self, text: str) -> Intent:
        """Classification basée sur des patterns regex"""
        best_intent = IntentType.GENERAL_CHAT
        best_score = 0.0
        entities: dict[str, Any] = {}

        for intent_type, patterns in INTENT_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 0.3

            if score > best_score:
                best_score = score
                best_intent = intent_type

        # Extraction d'entités pour trading
        if best_intent == IntentType.TRADING_ORDER:
            entities = self._extract_trading_entities(text)

        # Confiance basée sur le score
        confidence = min(0.9, 0.5 + best_score) if best_score > 0 else 0.95

        return Intent(
            intent_type=best_intent,
            confidence=confidence,
            entities=entities,
            target_expert=INTENT_TO_EXPERT.get(best_intent, "core"),
            raw_text=text,
        )

    def _extract_trading_entities(self, text: str) -> dict[str, Any]:
        """Extrait les entités d'un ordre de trading"""
        entities: dict[str, Any] = {}

        # Action (BUY/SELL)
        if re.search(r"(achète|acheter|buy|ouvre|long)", text, re.IGNORECASE):
            entities["action"] = "BUY"
        elif re.search(r"(vends|vendre|sell|short)", text, re.IGNORECASE):
            entities["action"] = "SELL"

        # Symbole
        symbols = {
            r"(gold|xauusd|or)": "XAUUSD",
            r"(eurusd|eur/usd|euro.dollar)": "EURUSD",
            r"(gbpusd|gbp/usd|livre.dollar)": "GBPUSD",
            r"(usdjpy|usd/jpy|dollar.yen)": "USDJPY",
            r"(nasdaq|nas100|nq)": "NAS100",
            r"(us30|dow.jones|dji)": "US30",
        }
        for pattern, symbol in symbols.items():
            if re.search(pattern, text, re.IGNORECASE):
                entities["symbol"] = symbol
                break

        # Volume (lots)
        volume_match = re.search(r"(\d+\.?\d*)\s*(lot|lots)", text, re.IGNORECASE)
        if volume_match:
            entities["volume"] = float(volume_match.group(1))

        # Stop Loss
        sl_match = re.search(r"(sl|stop.?loss|stop)\s*[àa]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
        if sl_match:
            entities["stop_loss"] = float(sl_match.group(2))

        # Take Profit
        tp_match = re.search(r"(tp|take.?profit|profit)\s*[àa]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
        if tp_match:
            entities["take_profit"] = float(tp_match.group(2))

        return entities

    async def _classify_with_llm(self, text: str) -> Intent:
        """Classification utilisant le LLM via LangChain/Ollama"""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from shared import get_settings
        import json

        settings = get_settings()
        
        # Initialisation du LLM spécialisé pour la classification
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=f"http://{settings.ollama_host}:{settings.ollama_port}",
            temperature=0,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le routeur d'intentions d'E.V.A., une IA de trading et sécurité.
            Ta mission est d'analyser le message utilisateur et de retourner un JSON strict.
            
            Types d'intentions (intent_type) :
            - TRADING_ORDER : Passer un ordre d'achat ou vente.
            - POSITION_STATUS : Consulter l'état des trades ou profits.
            - RISK_INQUIRY : Question sur le drawdown, le risque ou les limites.
            - MEMORY_RECALL : Rappel de faits passés ou historique.
            - OSINT_REQUEST : Recherche d'infos externes (web, news).
            - SECURITY_ALERT : Alerte sur une menace ou intrusion.
            - SYSTEM_COMMAND : Commande technique (logs, reboot).
            - GENERAL_CHAT : Conversation normale ou question diverse.

            Pour TRADING_ORDER, extrais les entités: action (BUY/SELL), symbol (XAUUSD, EURUSD, etc.), volume (float), stop_loss (float), take_profit (float).
            
            Réponds UNIQUEMENT avec un JSON au format:
            {{
                "intent": "NOM_INTENT",
                "confidence": 0.0-1.0,
                "entities": {{ ... }},
                "explanation": "brève raison"
            }}
            """),
            ("user", "{input}")
        ])

        try:
            chain = prompt | llm
            response = await chain.ainvoke({"input": text})
            
            # Nettoyage de la réponse si le LLM ajoute du texte autour
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            intent_val = data.get("intent", "GENERAL_CHAT")
            # Validation de l'enum
            try:
                intent_type = IntentType(intent_val)
            except ValueError:
                intent_type = IntentType.GENERAL_CHAT

            return Intent(
                intent_type=intent_type,
                confidence=data.get("confidence", 0.9),
                entities=data.get("entities", {}),
                target_expert=INTENT_TO_EXPERT.get(intent_type, "core"),
                raw_text=text
            )

        except Exception as e:
            logger.error(f"Férocité LLM Router error: {e}")
            # Fallback sur patterns en cas d'erreur
            return self._classify_with_patterns(text.lower())
