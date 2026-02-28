"""
The Muse — Niche Profile System
Defines all content niches for the autonomous OnlyFans/MYM content factory.
Each niche has its own prompt templates, LoRA recommendations, and posting schedule.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NicheProfile:
    """A content niche with its generation and scheduling parameters."""
    id: str
    label: str
    description: str
    base_prompt: str      # Main positive prompt
    negative_prompt: str  # Always-on negative prompt
    recommended_loras: list[dict]  # [{"filename": "x.safetensors", "strength": 0.8}]
    telegram_channel: Optional[str] = None  # Specific channel, or None for default
    post_interval_hours: float = 8.0        # How often to auto-post
    enabled: bool = True
    is_nsfw: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# CATALOG — Add/remove niches here without touching any other code
# ═══════════════════════════════════════════════════════════════════════════════

NICHE_CATALOG: dict[str, NicheProfile] = {

    "fitness": NicheProfile(
        id="fitness",
        label="🏋️ Fitness & Athletic",
        description="Sport, athletic wear, energetic, gym aesthetic",
        base_prompt="1girl, athletic body, gym wear, sports bra, leggings, sweaty skin, natural lighting, high quality, 8k, photorealistic, toned body, confident expression",
        negative_prompt="ugly, low quality, disfigured, bad anatomy, watermark",
        recommended_loras=[
            {"filename": "fitness_athletic_v2.safetensors", "strength": 0.75}
        ],
        post_interval_hours=8.0,
        is_nsfw=False,
    ),

    "girlfriend": NicheProfile(
        id="girlfriend",
        label="💕 Girlfriend Experience",
        description="Sweet, intimate, girlfriend experience, candid selfie style",
        base_prompt="1girl, casual home outfit, natural makeup, warm smile, bedroom background, soft natural light, candid selfie, relatable, girlfriend vibe, high quality, photorealistic",
        negative_prompt="ugly, low quality, professional studio, harsh lighting, watermark",
        recommended_loras=[
            {"filename": "gfe_realistic_v3.safetensors", "strength": 0.8}
        ],
        post_interval_hours=6.0,
        is_nsfw=False,
    ),

    "dominatrice": NicheProfile(
        id="dominatrice",
        label="⛓️ Dominatrice / Maîtresse",
        description="BDSM dominant aesthetic, leather, latex, power dynamic",
        base_prompt="1girl, dominant expression, latex outfit, leather gloves, power pose, dramatic lighting, dark background, intense gaze, high quality, photorealistic, 8k",
        negative_prompt="ugly, low quality, submissive expression, warmth, watermark",
        recommended_loras=[
            {"filename": "bdsm_dominant_lora.safetensors", "strength": 0.85}
        ],
        post_interval_hours=12.0,
        is_nsfw=True,
    ),

    "soumise": NicheProfile(
        id="soumise",
        label="🎀 Douce & Soumise",
        description="Shy, submissive, delicate, soft aesthetic",
        base_prompt="1girl, shy smile, pastel lingerie, soft bedroom lighting, innocent expression, delicate pose, kawaii aesthetic, high quality, photorealistic",
        negative_prompt="ugly, low quality, dominant, aggressive, watermark",
        recommended_loras=[
            {"filename": "soft_submissive_v2.safetensors", "strength": 0.8}
        ],
        post_interval_hours=10.0,
        is_nsfw=True,
    ),

    "pied": NicheProfile(
        id="pied",
        label="🦶 Foot Fetish",
        description="Elegant feet, pedicure, foot close-up, beach or indoor settings",
        base_prompt="close up of elegant female feet, perfect pedicure, painted toenails, smooth skin, natural lighting, beach sand or silk sheets, aesthetic composition, high quality",
        negative_prompt="ugly feet, rough skin, dirty, low quality, watermark",
        recommended_loras=[
            {"filename": "foot_fetish_detail.safetensors", "strength": 0.9}
        ],
        post_interval_hours=12.0,
        is_nsfw=False,
    ),

    "rousse": NicheProfile(
        id="rousse",
        label="🦊 Rousse & Taches de Rousseur",
        description="Red hair, natural freckles, pale skin, artistic style",
        base_prompt="1girl, long red hair, natural freckles, pale skin, green eyes, natural outdoor lighting, artistic portrait, soft bokeh, high quality, photorealistic, 8k",
        negative_prompt="ugly, low quality, dark hair, watermark",
        recommended_loras=[
            {"filename": "redhead_freckles_lora.safetensors", "strength": 0.85}
        ],
        post_interval_hours=8.0,
        is_nsfw=False,
    ),

    "petite": NicheProfile(
        id="petite",
        label="🌸 Petite & Cute",
        description="Small frame, cute aesthetic, youthful energy",
        base_prompt="1girl, petite frame, cute outfit, playful expression, pastel colors, soft lighting, youthful energy, high quality, photorealistic",
        negative_prompt="ugly, tall, muscular, harsh, low quality, watermark",
        recommended_loras=[
            {"filename": "petite_cute_v1.safetensors", "strength": 0.75}
        ],
        post_interval_hours=8.0,
        is_nsfw=False,
    ),

    "milf": NicheProfile(
        id="milf",
        label="👑 MILF & Mature Elegance",
        description="Mature, confident, elegant, experienced woman energy",
        base_prompt="1woman, 35 to 45 years old, confident posture, elegant outfit, mature beauty, sophisticated makeup, warm lighting, luxury interior background, high quality, photorealistic, 8k",
        negative_prompt="young, teenage, childish, low quality, watermark",
        recommended_loras=[
            {"filename": "mature_elegant_v2.safetensors", "strength": 0.8}
        ],
        post_interval_hours=10.0,
        is_nsfw=False,
    ),

    "cosplay": NicheProfile(
        id="cosplay",
        label="🎮 Cosplay & Anime",
        description="Gaming characters, anime aesthetics, costume detail",
        base_prompt="1girl, detailed cosplay costume, anime-inspired character, vibrant colors, dramatic lighting, high quality craftsmanship, photorealistic, convention setting or studio, 8k",
        negative_prompt="ugly, low quality, poor costume, bad lighting, watermark",
        recommended_loras=[
            {"filename": "cosplay_detail_lora.safetensors", "strength": 0.8}
        ],
        post_interval_hours=12.0,
        is_nsfw=False,
    ),

    "furry": NicheProfile(
        id="furry",
        label="🦊 Furry Anthro",
        description="Anthropomorphic character art, furry fandom aesthetic",
        base_prompt="anthro female character, detailed fur texture, expressive eyes, furry costume, vibrant digital art style, high quality illustration, smooth rendering",
        negative_prompt="ugly, poorly drawn, bad anatomy, low quality, watermark",
        recommended_loras=[
            {"filename": "furry_anthro_v3.safetensors", "strength": 0.9}
        ],
        post_interval_hours=12.0,
        is_nsfw=False,
    ),
}


def get_niche(niche_id: str) -> NicheProfile | None:
    return NICHE_CATALOG.get(niche_id)

def list_niches() -> list[NicheProfile]:
    return list(NICHE_CATALOG.values())

def get_enabled_niches() -> list[NicheProfile]:
    return [n for n in NICHE_CATALOG.values() if n.enabled]
