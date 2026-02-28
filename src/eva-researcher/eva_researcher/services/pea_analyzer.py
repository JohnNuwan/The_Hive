import logging
import yfinance as yf
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

class PEAAnalyzerService:
    """Service d'Analyse Fondamentale pour Actions PEA/Européennes"""
    
    # Panier d'actions européennes fortes, éligibles PEA
    HIVE_PEA_BASKET = [
        "MC.PA",   # LVMH
        "AI.PA",   # Air Liquide
        "SU.PA",   # Schneider Electric
        "OR.PA",   # L'Oréal
        "TTE.PA",  # TotalEnergies
        "SAN.PA",  # Sanofi
        "BNP.PA",  # BNP Paribas
        "DG.PA",   # Vinci
        "RMS.PA",  # Hermès
        "ASML.AS"  # ASML Holding (Pays-Bas)
    ]

    def __init__(self, research_service):
        self.research = research_service
    
    async def analyze_basket(self) -> dict[str, Any]:
        """Analyse le panier complet et génère un rapport global via LLM"""
        start = datetime.now()
        results = []
        
        for symbol in self.HIVE_PEA_BASKET:
            try:
                metrics = await self.fetch_stock_metrics(symbol)
                # On limite la recherche de news pour ne pas timeout, ou on prend juste le top récent
                news = await self.research._web_search(f"{metrics['company_name']} stock latest news", max_results=2)
                
                news_text = "\n".join([f"- {n.title}" for n in news])
                
                results.append({
                    "symbol": symbol,
                    "metrics": metrics,
                    "recent_news": news_text
                })
            except Exception as e:
                logger.error(f"Error analyzing PEA stock {symbol}: {e}")
        
        # Consolidation et inférence LLM
        synthesis = await self._synthesize_pea_report(results)
        
        elapsed = int((datetime.now() - start).total_seconds() * 1000)
        
        return {
            "basket": self.HIVE_PEA_BASKET,
            "analysis": results,
            "llm_synthesis": synthesis,
            "execution_time_ms": elapsed
        }

    async def fetch_stock_metrics(self, symbol: str) -> dict[str, Any]:
        """Récupère les ratios financiers via yfinance"""
        import asyncio
        loop = asyncio.get_running_loop()
        
        def _fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "company_name": info.get("shortName", symbol),
                "sector": info.get("sector", "Unknown"),
                "current_price": info.get("currentPrice", 0.0),
                "pe_ratio": info.get("trailingPE", None),
                "forward_pe": info.get("forwardPE", None),
                "dividend_yield": info.get("dividendYield", 0.0) * 100 if info.get("dividendYield") else 0.0,
                "target_mean_price": info.get("targetMeanPrice", None),
                "52_week_high": info.get("fiftyTwoWeekHigh", None),
                "52_week_low": info.get("fiftyTwoWeekLow", None),
                "roe": info.get("returnOnEquity", 0.0) * 100 if info.get("returnOnEquity") else 0.0,
                "free_cashflow": info.get("freeCashflow", None)
            }
            
        return await loop.run_in_executor(None, _fetch)

    async def _synthesize_pea_report(self, analyzed_stocks: list[dict]) -> str:
        """Génère la recommandation d'investissement via le module de recherche LLM"""
        import json
        
        # Préparation du contexte compressé pour le LLM
        context_lines = []
        for stock in analyzed_stocks:
            m = stock["metrics"]
            pe = f"P/E: {m['pe_ratio']:.1f}" if m['pe_ratio'] else "P/E: N/A"
            div = f"Div: {m['dividend_yield']:.1f}%"
            target = f"Target: {m['target_mean_price']}"
            context_lines.append(f"[{stock['symbol']} - {m['company_name']}]: Price {m['current_price']}, {pe}, {div}, {target}. News: {stock['recent_news']}")
        
        context = "\n".join(context_lines)
        
        prompt = (
            "Tu es un analyste financier expert (The Sage/Researcher). Analyse ces actions européennes (PEA) et leurs fondamentaux actuels :\n"
            f"{context}\n\n"
            "Fournis un rapport stratégique en 3 points pour le portefeuille long-terme de 'The Hive' :\n"
            "1. Les 2 actions les plus sous-évaluées/prometteuses (Strong Buy).\n"
            "2. Un commentaire sur les rendements de dividendes.\n"
            "3. Les risques macro-économiques potentiels (d'après les news).\n"
            "Rends ton analyse incisive, professionnelle et orientée investissement à 1-3 ans."
        )
        
        # On utilise le LLM existant du ResearchService
        return await self.research._synthesize("PEA Fundamental Analysis", [{"title": "Data", "summary": prompt}])
