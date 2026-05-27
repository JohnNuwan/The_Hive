"""Client Discord Webhook pour la distribution des alertes multi-salons THE HIVE."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from datetime import datetime
from typing import Optional, Any

import requests

logger = logging.getLogger(__name__)

# Table des Webhooks Discord extraite de Liste_salon.csv
DISCORD_WEBHOOKS = {
    "général": "https://discord.com/api/webhooks/1502188367603826791/RLdlnyoWy1UCxJVQNdBx30QaKcpIRcCMk1kk4-vOZdu4qiAF8Phg-qzEGqrhJL505jdo",
    "disclaimer": "https://discord.com/api/webhooks/1502188520419098634/GCSMymITilR0fWJPknC-YaYuFSnrnoDMy4HfFpIN390LQtExfR9ixZ34qGP_U5bTugJq",
    "analyse_technique": "https://discord.com/api/webhooks/1502188727726772286/jCdR3QOSTT5mw6m-tPIsXPCvocFQxwIyYZetpF16YUttdrhRHc9w0eNXiI6VfhYOE_lV",
    "scalping": "https://discord.com/api/webhooks/1502188930706178168/8YeYiFttdqBk1cPU0RVn79fnY4ZasUa7r8VeReNPB75-7z0N1yHzXdhFmXlpYPi0vtn4",
    "intraday": "https://discord.com/api/webhooks/1502189038851981365/vqnbLuYzM4bAQZqZc4ls0Azf5mNNnM7ZBrFfkD0n9lunQclI5tUcaghGQSbjjs0IrNcp",
    "swing": "https://discord.com/api/webhooks/1502189103075033189/_1BTTSlZYrCpy40cpWubZFauDama980GQG3vtV8-y2zUF2EB0Nbq0YDBM9HCtlgbfSQ3",
    "certification": "https://discord.com/api/webhooks/1502189335917887500/JD80CapjFALADHgNZCMUh2FtRQktGO0sT_SEzMcp70jJWRQiaA_ghLUO2380IUPknzBN",
    "eva": "https://discord.com/api/webhooks/1504560956787921057/7EnYrEYKdZsewo9gTA0F3mxRXw4EnrghsZaAPLMRMZmXRiV0VrZPdzQ40cGVUCTu9GlD",
}


def dispatch_channel(text: str, category: Optional[str] = None) -> str:
    """Détermine le salon cible en fonction de la catégorie et du contenu textuel.

    Args:
        text (str): Le contenu textuel du message.
        category (Optional[str]): Une catégorie explicite passée par le notifier.

    Returns:
        str: Le nom du salon Discord cible.
    """
    if category:
        cat_norm = str(category).lower().strip()
        if cat_norm in {"general", "général"}:
            return "général"
        if cat_norm in {"disclaimer", "danger", "warning", "emergency", "alert", "danger"}:
            return "disclaimer"
        if cat_norm in {"analyse_technique", "technical", "analyse", "hermes", "technical_analysis"}:
            return "analyse_technique"
        if cat_norm in {"scalp", "scalping"}:
            return "scalping"
        if cat_norm in {"intraday"}:
            return "intraday"
        if cat_norm in {"swing"}:
            return "swing"
        if cat_norm in {"certification", "compliance", "audit", "drawdown", "ftmo", "ftuk"}:
            return "certification"
        if cat_norm in {"eva", "chat_eva", "talk_eva", "talk-eva"}:
            return "eva"

    text_upper = text.upper()

    # 1. 🚨 DANGER / URGENCE SYSTEME (Priorité absolue)
    if any(k in text_upper for k in ["🚨", "EMERGENCY", "CRITICAL", "DANGER", "KILL SWITCH", "FATAL", "DRAWDOWN DEPASSE"]):
        return "disclaimer"

    # 2. ANALYSE TECHNIQUE / HERMES TECHNICAL REPORT (Priorité sur les mots-clés de marché et de comptes)
    if any(k in text_upper for k in ["ANALYSE TECHNIQUE", "CHARTIST", "CHARTISTE", "FIBONACCI", "SUPPORT & RESISTANCE", "RSI", "MACD", "EMA-200", "ADX", "VWAP", "ATR", "HERMES REPORT", "LOSS AUDITOR DIAGNOSIS"]):
        return "analyse_technique"

    # 3. COMPLIANCE / FTMO / AUDITS
    if any(k in text_upper for k in ["FTMO", "FTUK", "COMPLIANCE", "AUDITOR", "AUDIT", "CERTIFICATION", "MUTATION", "COMPTE DISSOLU"]):
        return "certification"

    # SCALPING (Signaux MuZero Scalp / Dreamer Scalp)
    if any(k in text_upper for k in ["SCALP", "SCALPING", "MUZERO_SCALP", "DREAMER_SCALP", "M5"]):
        return "scalping"

    # INTRADAY
    if any(k in text_upper for k in ["INTRADAY", "H1"]):
        return "intraday"

    # SWING
    if any(k in text_upper for k in ["SWING", "D1"]):
        return "swing"

    # E.V.A COGNITIVE CHAT
    if any(k in text_upper for k in ["EVA CORE", "EVA COGNITIVE", "ELECTRONIC VIRTUAL ASSISTANT", "EVA TALK"]):
        return "eva"

    return "général"


def build_discord_embed(text: str, channel: str, image_url: Optional[str] = None) -> dict[str, Any]:
    """Construit un Embed Discord enrichi et esthétique pour les notifications.

    Args:
        text (str): Le contenu textuel.
        channel (str): Le salon Discord cible.
        image_url (Optional[str]): URL d'image ou d'attachement à inclure.

    Returns:
        dict[str, Any]: La charge utile d'Embed Discord.
    """
    # Harmonie de couleurs (Palette premium)
    colors = {
        "général": 0x2F3542,            # Gris ardoise élégant
        "disclaimer": 0xFF4757,         # Rouge vif d'alarme
        "analyse_technique": 0x54A0FF,  # Bleu néon d'analyse
        "scalping": 0xFFA502,           # Orange/Or MuZero
        "intraday": 0x2ED573,           # Vert émeraude Intraday
        "swing": 0x3CAEA3,              # Cyan profond Swing
        "certification": 0x9B59B6,       # Violet de certification
        "eva": 0x10AC84,                 # Vert menthe néon E.V.A
    }

    titles = {
        "général": "ℹ️ INFO - THE HIVE",
        "disclaimer": "🚨 ALERT SYSTEM - EMERGENCY",
        "analyse_technique": "📊 TECH ANALYSIS - HERMES COGNITIVE",
        "scalping": "⚡ DYNAMIC SIGNAL - SCALPING",
        "intraday": "🕒 POSITION SIGNAL - INTRADAY",
        "swing": "📈 POSITION SIGNAL - SWING",
        "certification": "📋 COMPLIANCE & drawdown CHECK",
        "eva": "🌿 E.V.A - SUPERVISOR COGNITIVE CORE",
    }

    color = colors.get(channel, 0x2F3542)
    title = titles.get(channel, "THE HIVE SYSTEM MESSAGE")

    embed = {
        "title": title,
        "description": text,
        "color": color,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "footer": {
            "text": "THE HIVE Sentinel • Notification Dispatcher",
            "icon_url": "https://cdn.discordapp.com/embed/avatars/0.png"
        }
    }

    if image_url:
        embed["image"] = {"url": image_url}

    return embed


class DiscordClient:
    """Envoie des messages formatés et des photos sur les salons Discord configurés."""

    def __init__(self) -> None:
        """Initialise le client de notifications Discord."""
        self.enabled = len(DISCORD_WEBHOOKS) > 0
        logger.info("DiscordClient initialisé avec %d salons de destination.", len(DISCORD_WEBHOOKS))

    def _send_sync_internal(
        self,
        message: str,
        *,
        category: Optional[str] = None,
    ) -> None:
        """Envoie de manière synchrone un message sous forme d'Embed vers le salon approprié.

        Args:
            message (str): Corps du message.
            category (Optional[str]): Catégorie optionnelle pour forcer le routage.
        """
        if not self.enabled:
            return

        channel = dispatch_channel(message, category)
        webhook_url = DISCORD_WEBHOOKS.get(channel)
        if not webhook_url:
            logger.warning("Aucun Webhook trouvé pour le salon : %s", channel)
            return

        embed = build_discord_embed(message, channel)
        payload = {
            "embeds": [embed]
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=15)
            if response.status_code not in (200, 204):
                logger.error(
                    "Erreur d'envoi Discord (%s) : %s - %s",
                    channel,
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.error("Erreur de connexion Discord (%s) : %s", channel, exc)

    async def send_message(
        self,
        message: str,
        *,
        category: Optional[str] = None,
    ) -> None:
        """Envoie asynchrone d'un message vers le salon approprié.

        Args:
            message (str): Corps du message.
            category (Optional[str]): Catégorie optionnelle pour forcer le routage.
        """
        if not self.enabled:
            return
        await asyncio.to_thread(
            self._send_sync_internal,
            message,
            category=category,
        )

    def _send_photo_sync_internal(
        self,
        photo: bytes,
        caption: str,
        *,
        category: Optional[str] = None,
    ) -> None:
        """Envoie de manière synchrone une photo accompagnée d'une légende vers le salon approprié.

        Args:
            photo (bytes): Données binaires de l'image.
            caption (str): Légende textuelle associée.
            category (Optional[str]): Catégorie optionnelle.
        """
        if not self.enabled:
            return

        channel = dispatch_channel(caption, category)
        webhook_url = DISCORD_WEBHOOKS.get(channel)
        if not webhook_url:
            logger.warning("Aucun Webhook de photo trouvé pour le salon : %s", channel)
            return

        # Envoi en multipart (fichier + embed)
        # NOTE: payload_json DOIT être une chaîne JSON brute (non URL-encodée)
        import json as _json
        files = {"file": ("chart.png", photo, "image/png")}
        embed = build_discord_embed(caption, channel, image_url="attachment://chart.png")
        data = {
            "payload_json": _json.dumps({"embeds": [embed]})
        }

        try:
            response = requests.post(webhook_url, data=data, files=files, timeout=30)
            if response.status_code not in (200, 204):
                logger.error(
                    "Erreur d'envoi Photo Discord (%s) : %s - %s",
                    channel,
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.error("Erreur de connexion Photo Discord (%s) : %s", channel, exc)

    async def send_photo(
        self,
        photo: bytes,
        caption: str,
        *,
        category: Optional[str] = None,
    ) -> None:
        """Envoie asynchrone d'une photo accompagnée d'une légende vers le salon approprié.

        Args:
            photo (bytes): Données binaires de l'image.
            caption (str): Légende textuelle associée.
            category (Optional[str]): Catégorie optionnelle.
        """
        if not self.enabled:
            return
        await asyncio.to_thread(
            self._send_photo_sync_internal,
            photo,
            caption,
            category=category,
        )

    def send_sync(
        self,
        message: str,
        *,
        category: Optional[str] = None,
    ) -> None:
        """Déclenche un envoi synchrone tolérant à l'absence de boucle d'événement.

        Args:
            message (str): Corps du message.
            category (Optional[str]): Catégorie optionnelle.
        """
        if not self.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                None,
                functools.partial(
                    self._send_sync_internal,
                    message,
                    category=category,
                ),
            )
        except RuntimeError:
            self._send_sync_internal(message, category=category)
        except Exception as exc:
            logger.error("Erreur Discord synchrone : %s", exc)
