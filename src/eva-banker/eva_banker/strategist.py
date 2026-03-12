import logging
import asyncio
import json
import os
import unicodedata
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

from shared.redis_client import get_redis_client

from shared.llm_client import LLMClient
from shared import get_settings
from eva_banker.services.mt5 import MT5Service
from shared.indicators import IndicatorFactory

logger = logging.getLogger(__name__)

class Strategist:
    """
    The Cortex: High-Level Strategic planner using LLM (Gemma 3).
    Analyzes Macro Trends (M15/H1) to guide the Subconscious (Dreamer).
    """
    def __init__(self, mt5_service: MT5Service):
        self.mt5 = mt5_service
        settings = get_settings()
        self.cortex_model = self._resolve_cortex_model(settings)
        self.cortex = LLMClient(model=self.cortex_model)
        logger.info("Cortex bancaire initialis? avec le mod?le: %s", self.cortex_model)
        self.latest_strategy: Dict[str, Any] = {}
        self.last_update: datetime = datetime.min

    @staticmethod
    def _resolve_cortex_model(settings) -> str:
        """
        R?sout le mod?le LLM du Cortex selon le backend actif.

        Priorit?:
            1) `BANKER_CORTEX_MODEL` si d?fini.
            2) Si backend vLLM: `COUNCIL_MODEL_BANKER` si compatible, sinon `vllm_model`.
            3) Si backend Ollama: `ollama_model`.
        """
        direct_model = os.getenv("BANKER_CORTEX_MODEL", "").strip()
        if direct_model:
            return direct_model

        if settings.llm_backend == "vllm":
            candidate = os.getenv("COUNCIL_MODEL_BANKER", settings.council_model_banker).strip()
            # Les tags de type `modele:tag` sont en g?n?ral des IDs Ollama.
            if candidate and ":" in candidate and "/" not in candidate:
                logger.warning(
                    "COUNCIL_MODEL_BANKER=%s ressemble ? un mod?le Ollama; fallback vers vLLM_MODEL=%s",
                    candidate,
                    settings.vllm_model,
                )
                return settings.vllm_model
            return candidate or settings.vllm_model

        return settings.ollama_model

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        """Retire les balises Markdown autour d'un JSON potentiel.

        Args:
            content (str): Texte brut retourne par le LLM.

        Returns:
            str: Texte nettoye, sans balises Markdown parasites.
        """
        cleaned = (content or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return cleaned.strip()

    @staticmethod
    def _sanitize_reasoning_text(content: str) -> str:
        """Normalise un texte libre pour l'affichage console et Telegram.

        Args:
            content (str): Texte a nettoyer.

        Returns:
            str: Texte ASCII compact, sans repetition immediate evidente.
        """
        cleaned = " ".join(str(content or "").replace("\n", " ").split()).strip(" -:")
        ascii_cleaned = (
            unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
        )
        return ascii_cleaned.strip(" -:")

    @classmethod
    def _parse_cortex_response(cls, response: str) -> tuple[str, str]:
        """Extrait un biais et une synthese courte depuis la reponse du Cortex.

        Args:
            response (str): Reponse brute du modele.

        Returns:
            tuple[str, str]: Biais normalise et raison exploitable en francais.
        """
        cleaned = cls._strip_json_fence(response)
        if not cleaned:
            return "NEUTRAL", "Synthese Cortex indisponible."

        payload_text = cleaned
        if "{" in cleaned and "}" in cleaned:
            payload_text = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            parsed = {}

        bias_source = str(parsed.get("bias", cleaned)).upper()
        reason = str(parsed.get("reason", cleaned)).strip()
        details = str(parsed.get("details", "")).strip()

        bias = "NEUTRAL"
        if "BULLISH" in bias_source:
            bias = "BULLISH"
        elif "BEARISH" in bias_source:
            bias = "BEARISH"
        elif "RANGING" in bias_source or "RANGE" in bias_source:
            bias = "RANGING"

        normalized_reason = cls._sanitize_reasoning_text(reason)
        normalized_details = cls._sanitize_reasoning_text(details)
        if (
            normalized_details
            and normalized_details.lower() != normalized_reason.lower()
            and normalized_details.lower() not in normalized_reason.lower()
        ):
            normalized_reason = f"{normalized_reason}. {normalized_details}".strip(". ")
        if not normalized_reason:
            normalized_reason = "Synthese Cortex indisponible."

        return bias, normalized_reason

    async def analyze_market_context(self, symbol: str) -> Dict[str, Any]:
        """Analyse le contexte M15 et produit un biais macro exploitable.

        Args:
            symbol (str): Symbole a analyser.

        Returns:
            Dict[str, Any]: Biais du Cortex, biais GNN, biais final et synthese.
        """
        logger.info(f"ðŸ§  Cortex: Analyzing Macro Context for {symbol}...")
        
        # 1. Fetch M15 Data (Macro View)
        candles = await self.mt5.get_recent_candles(symbol, timeframe=15, count=100)
        if not candles:
            logger.warning("Cortex: No M15 data available.")
            return {"action": "NEUTRAL", "reason": "No Data"}

        # 2. Calculate Macro Indicators
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["tick_volume"] for c in candles]
        
        rsi = IndicatorFactory.rsi(closes, 14).iloc[-1]
        trend_ema = IndicatorFactory.ema(closes, 50).iloc[-1]
        current_price = closes[-1]
        
        # New Advanced Indicators (VWAP, ADX, Fibs)
        vwap = IndicatorFactory.vwap(highs, lows, closes, volumes).iloc[-1]
        adx_data = IndicatorFactory.adx(highs, lows, closes, 14)
        adx_val = adx_data["adx"].iloc[-1]
        fibs = IndicatorFactory.get_fibonacci_levels(highs, lows, 50)
        
        trend = "BULLISH" if current_price > trend_ema else "BEARISH"
        
        # 3. Formulate Prompt for Gemma
        context = {
            "symbol": symbol,
            "timeframe": "M15",
            "price": current_price,
            "trend_50ema": trend,
            "rsi_14": round(rsi, 2),
            "vwap": round(vwap, 2) if not __import__("math").isnan(vwap) else current_price,
            "adx_trend_strength": round(adx_val, 2),
            "fib_382": round(fibs.get("fib_382", 0), 2),
            "fib_618": round(fibs.get("fib_618", 0), 2),
            "last_5_candles": [round(c["close"], 2) for c in candles[-5:]]
        }
        
        prompt = (
            f"Analyse la structure M15 de {symbol}. "
            f"La tendance 50 EMA est {trend}. RSI={rsi:.1f}. ADX={adx_val:.1f}. "
            f"Le prix vaut {current_price:.2f} pour une VWAP a {context['vwap']:.2f}. "
            f"Les niveaux de Fibonacci clefs sont {context['fib_382']:.2f} et {context['fib_618']:.2f}. "
            "Determine le biais strategique parmi BULLISH, BEARISH ou RANGING. "
            "Tiens compte des rejets de VWAP et du seuil ADX > 25 pour la poursuite de tendance. "
            "Reponds en JSON strict sur une seule ligne, sans Markdown, avec "
            "{\"bias\":\"BULLISH|BEARISH|RANGING\",\"reason\":\"premiere phrase courte en francais\","
            "\"details\":\"seconde phrase courte en francais avec le niveau ou le signal cle\"}."
        )

        # 4. Ask Cortex
        response = await self.cortex.analyze(json.dumps(context), prompt)
        
        # 5. Parse Response
        cortex_bias, cortex_reason = self._parse_cortex_response(response)

        # 6. Ask Proxmox MTF-GNN via REST API
        gnn_intraday = "NEUTRAL"
        gnn_scalp = "NEUTRAL"
        gnn_swing = "NEUTRAL"
        gnn_confidence = 0.0
        try:
            import aiohttp
            import os
            lab_host = os.getenv("LAB_HOST", "localhost")
            url = f"http://{lab_host}:8600/gnn/predict"
            
            # Send multi-timeframe payload using horizon-tagged keys
            # Each value is a [seq_len] list of recent closes (simplified feature vector)
            payload = {
                "assets_data": {
                    f"{symbol}_5": [closes[-15:]],   # M5 context = Scalping
                    f"{symbol}_60": [closes[-15:]],  # H1 context = Intraday
                    f"{symbol}_1440": [closes[-15:]], # D1 context = Swing
                }
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # New MTF response format
                        gnn_scalp = data.get("scalp", {}).get("bias", "NEUTRAL")
                        gnn_intraday = data.get("intraday", {}).get("bias", "NEUTRAL")
                        gnn_swing = data.get("swing", {}).get("bias", "NEUTRAL")
                        gnn_confidence = data.get("intraday", {}).get("confidence", 0.0)
                    else:
                        logger.warning(f"âš ï¸ GNN a retournÃ© HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"âš ï¸ Impossible de joindre le MTF-GNN: {e.__class__.__name__}")
        
        # Reference the intraday bias for Cortex matching (M15 analysis runs on H1-like horizon)
        gnn_bias = gnn_intraday
            
        # 7. Merge Biases (Swing override on extreme conviction, else intraday GNN cross-check)
        final_bias = cortex_bias
        if gnn_swing != "NEUTRAL" and gnn_swing != cortex_bias:
            # D1 macro context forces RANGING if contradicting the Cortex's M15 reading
            final_bias = "RANGING"
        elif gnn_bias != "NEUTRAL" and gnn_confidence > 0.5:
            if gnn_bias == cortex_bias:
                pass  # Synergy!
            else:
                # GNN intraday contradicts Cortex M15: moderate by forcing RANGING
                final_bias = "RANGING" if gnn_confidence < 0.75 else gnn_bias

        strategy = {
            "symbol": symbol,
            "cortex_bias": cortex_bias,
            "gnn_bias": gnn_bias,
            "bias": final_bias,
            "raw_thought": cortex_reason,
            "raw_response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.latest_strategy[symbol] = strategy
        
        # --- COMMAND LINE LOGGING ---
        from colorama import Fore, Style
        sym_color = Fore.CYAN if "XAU" in symbol else (Fore.YELLOW if "BTC" in symbol else Fore.WHITE)
        
        def get_color(b):
            if b == "BULLISH": return Fore.GREEN
            if b == "BEARISH": return Fore.RED
            if b == "RANGING": return Fore.MAGENTA
            return Fore.LIGHTBLACK_EX
            
        cb_color = get_color(cortex_bias)
        gb_color = get_color(gnn_bias)
        fb_color = get_color(final_bias)
        
        # --- RICH CONSOLE OUTPUT ---
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text
            
            console = Console()
            
            text = Text()
            text.append("ðŸ“Š Technical Context:\n", style="bold yellow")
            text.append(f"  â€¢ Price: {current_price:.2f} (Trend: {trend})\n", style="white")
            text.append(f"  â€¢ RSI: {rsi:.1f} | ADX: {adx_val:.1f} | VWAP: {vwap:.2f}\n", style="white")
            text.append(f"  â€¢ Fibs: {fibs.get('fib_382', 0):.2f} / {fibs.get('fib_618', 0):.2f}\n\n", style="white")
            
            text.append("ðŸ§  LLM Reasoning:\n", style="bold cyan")
            text.append(f"{cortex_reason}\n\n", style="italic white")
            
            text.append(f"Cortex Bias: ", style="bold")
            text.append(f"[{cortex_bias}]\n", style=cb_color.replace('\x1b[', '').replace('m', '').lower() if hasattr(cb_color, 'replace') else "white")
            
            text.append(f"GNN Bias: ", style="bold")
            text.append(f"[{gnn_bias}]\n", style=gb_color.replace('\x1b[', '').replace('m', '').lower() if hasattr(gb_color, 'replace') else "white")
            
            text.append(f"Final Bias: ", style="bold")
            text.append(f"[{final_bias}]", style=fb_color.replace('\x1b[', '').replace('m', '').lower() if hasattr(fb_color, 'replace') else "white")
            
            panel = Panel(
                text,
                title=f"The Cortex: {symbol} (M15)",
                border_style="magenta",
                expand=False
            )
            console.print(panel)
            
        except ImportError:
            # Fallback if rich is somehow missing
            logger.info(
                f"ðŸ§  Cortex Strategy -> {sym_color}{symbol:<8}{Style.RESET_ALL} | "
                f"Cortex: {cb_color}[{cortex_bias}]{Style.RESET_ALL} | "
                f"GNN: {gb_color}[{gnn_bias}]{Style.RESET_ALL} -> "
                f"Final: {fb_color}[{final_bias}]{Style.RESET_ALL} "
            )
        
        # PUBLISH TO AGENT FEED (UI)
        try:
            redis = get_redis_client()
            asyncio.create_task(redis.publish("eva.cortex.feed", {
                "id": str(uuid.uuid4()),
                "source_agent": "Strategist",
                "action": f"Cortex M15 Macro Bias for {symbol}: {final_bias} (GNN: {gnn_bias}) -> {cortex_reason[:100]}...",
                "timestamp": datetime.now().isoformat(),
                "type": "thought"
            }))
        except Exception as e_redis:
            logger.debug(f"Failed to publish Cortex thought to Feed: {e_redis}")
            
        # ----------------------------
        
        return strategy

    def get_bias(self, symbol: str) -> str:
        return self.latest_strategy.get(symbol, {}).get("bias", "NEUTRAL")

    async def get_micro_reasoning(self, symbol: str, action: str, indicators: dict) -> str:
        """
        Genere une synthese tres courte en francais pour un trade.

        Args:
            symbol (str): Symbole traite.
            action (str): Action candidate.
            indicators (dict): Indicateurs utiles au commentaire.

        Returns:
            str: Une phrase courte exploitable dans Telegram.
        """
        rsi = indicators.get("RSI", 50)
        adx = indicators.get("adx", 25)
        macd = indicators.get("MACD_Hist", 0)

        prompt = (
            f"Explique en une seule phrase courte, en francais, pourquoi ouvrir un {action} sur {symbol}. "
            f"Contexte: RSI={rsi:.1f}, ADX={adx:.1f}, MACD hist={macd:.4f}. "
            "Reponds uniquement avec la phrase finale, sans JSON, sans puce et sans titre."
        )

        try:
            reasoning = await self.cortex.analyze(f"Action: {action} on {symbol}", prompt)
            cleaned = " ".join(str(reasoning).replace("*", " ").split())
            return (
                cleaned.split(". ")[0][:200].strip(" -:")
                or "Le signal technique reste coherent avec le contexte courant."
            )
        except Exception as e:
            logger.warning("Generation du micro-raisonnement impossible: %s", e)
            return "Le signal technique reste coherent avec le contexte courant."
