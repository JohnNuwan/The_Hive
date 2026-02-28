"""
Market Researcher — Scores content niches by trend relevance.
Uses Ollama to analyze which content categories are performing well
on adult platforms (OnlyFans, MYM, Fansly) right now.
"""
import logging
import asyncio
import httpx
from eva_muse.niches import NICHE_CATALOG, NicheProfile
from shared import get_settings

logger = logging.getLogger(__name__)


async def score_niches() -> dict[str, float]:
    """
    Asks the LLM to score each niche from 0.0 to 1.0 based on current market trends.
    Returns a dict: { niche_id: score }
    """
    settings = get_settings()
    niche_list = "\n".join([f"- {n.id}: {n.label} ({n.description})" for n in NICHE_CATALOG.values()])
    
    prompt = f"""You are an adult content market analyst with deep knowledge of OnlyFans, MYM, and Fansly trends in 2024-2025.

For each content niche below, output a trend score between 0.0 (declining/saturated) and 1.0 (exploding/high demand).
Output ONLY a JSON object with niche_id as keys and a float score as values. No explanation.

Niches to score:
{niche_list}

Output format example:
{{"fitness": 0.7, "dominatrice": 0.85, "girlfriend": 0.9}}
"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                f"http://{settings.ollama_host}:{settings.ollama_port}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                }
            )
            raw = res.json().get("response", "{}").strip()
            
            # Extract JSON from the response
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                import json
                scores = json.loads(match.group())
                logger.info(f"Niche trend scores: {scores}")
                return scores
            else:
                logger.warning("Could not parse niche scores from LLM response")
                return _default_scores()
                
    except Exception as e:
        logger.error(f"Market researcher error: {e}")
        return _default_scores()


def _default_scores() -> dict[str, float]:
    """Fallback scores when LLM is unavailable."""
    return {
        "girlfriend": 0.90,
        "fitness": 0.75,
        "milf": 0.80,
        "rousse": 0.70,
        "dominatrice": 0.85,
        "soumise": 0.78,
        "pied": 0.65,
        "petite": 0.72,
        "cosplay": 0.68,
        "furry": 0.55,
    }


async def get_recommended_niche_order() -> list[str]:
    """Returns niche IDs sorted by trend score (highest first)."""
    scores = await score_niches()
    sorted_niches = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [niche_id for niche_id, _ in sorted_niches]
