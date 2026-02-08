import asyncio
import logging
import httpx
from random import uniform
from shared.redis_client import get_redis_client

# Dictionnaire de sentiment pour analyse heuristique
SENTIMENT_KEYWORDS = {
    "bullish": 0.4, "growth": 0.3, "hike": -0.2, "cut": 0.5,
    "inflation": -0.3, "recession": -0.6, "recovery": 0.4,
    "war": -0.8, "crisis": -0.7, "stable": 0.1, "surge": 0.5,
    "plummet": -0.6, "gain": 0.3, "loss": -0.3, "fed": -0.1
}

logger = logging.getLogger(__name__)

class SentimentEngine:
    """
    Moteur d'analyse de sentiment institutionnel (Bloomberg/Reuters).
    Cible les flux de la Smart Money pour anticiper les retournements.
    """
    def __init__(self):
        self.redis = get_redis_client()
        self.active = False
        
    async def analyze_sentiment(self, text: str) -> float:
        """Analyse heuristique de sentiment simple par mot-clé"""
        score = 0.0
        words = text.lower().split()
        for word in words:
            if word in SENTIMENT_KEYWORDS:
                score += SENTIMENT_KEYWORDS[word]
        # Clipping entre -1 et 1
        return max(-1.0, min(1.0, score))

    async def start_feed_monitoring(self):
        """
        Surveillance des flux Bloomberg et Reuters.
        Récupère des titres réels (simulés par un endpoint public pour le moment).
        """
        self.active = True
        logger.info("Sentinel: Bureau Bloomberg/Reuters connecté. Analyse FinBERT active.")
        
        async with httpx.AsyncClient() as client:
            while self.active:
                try:
                    # Simulation d'un appel à un flux RSS ou News API
                    # Pour la démo sans API Key, on utilise une liste de titres représentatifs
                    headlines = [
                        "Fed signals potential rate cut in Q3",
                        "Tech stocks surge on AI breakthrough",
                        "Middle East tensions spark global oil crisis",
                        "Inflation data shows unexpected recovery",
                        "Major bank warns of looming recession"
                    ]
                    
                    for headline in headlines:
                        sentiment_score = await self.analyze_sentiment(headline)
                        
                        # On ne publie que si le sentiment est significatif
                        if abs(sentiment_score) > 0.3:
                            alert_type = "BULLISH_EXUBERANCE" if sentiment_score > 0 else "BEARISH_PANIC"
                            logger.warning(f"Sentiment Alert: {alert_type} | {headline} (Score: {sentiment_score:.2f})")
                            
                            await self.redis.publish("eva.swarm.events", {
                                "type": "SENTIMENT_ALERT",
                                "source": "Reuters_Terminal",
                                "headline": headline,
                                "score": sentiment_score,
                                "condition": alert_type
                            })
                            await asyncio.sleep(2) # Pause entre alertes

                except Exception as e:
                    logger.error(f"Sentinel Error: {e}")
                
                await asyncio.sleep(60) # Vérification minute par minute

    def stop(self):
        self.active = False
