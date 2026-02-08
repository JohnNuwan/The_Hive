"""
Telegram Notifier Service — THE HIVE
Broadbands critical events to the Master's mobile.
"""

import httpx
import logging
import asyncio
from typing import Optional
from shared.config import get_settings

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """
    Handles sending messages to Telegram via Bot API.
    Supports a mock mode if credentials are missing.
    """
    
    def __init__(self):
        settings = get_settings()
        self.bot_token = getattr(settings, "telegram_bot_token", None)
        self.chat_id = getattr(settings, "telegram_chat_id", None)
        self.enabled = self.bot_token and self.chat_id
        
        if not self.enabled:
            logger.warning("🔔 Telegram Notifier in MOCK MODE (No credentials found in .env)")
        else:
            logger.info("✅ Telegram Notifier INITIALIZED")

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Sends a notification message."""
        if not self.enabled:
            logger.info(f"🎭 [MOCK TELEGRAM] {text}")
            return True

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    return True
                logger.error(f"❌ Telegram API Error: {response.text}")
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram alert: {e}")
        
        return False

    async def notify_emergency(self, source: str, message: str):
        """High priority emergency alert."""
        text = f"🚨 *THE HIVE EMERGENCY* 🚨\n\n*Source:* `{source}`\n*Alert:* {message}\n\n_System status under investigation._"
        await self.send_message(text)

    async def notify_trade(self, symbol: str, profit: float, ticket: int):
        """Trade execution alert."""
        emoji = "💰" if profit > 0 else "📉"
        text = f"{emoji} *TRADE CLOSED* {emoji}\n\n*Symbol:* `{symbol}`\n*Result:* `{profit:.2f} €`\n*Ticket:* `{ticket}`"
        await self.send_message(text)

    async def notify_self_healing(self, service: str, event: str):
        """Phoenix Protocol event alert."""
        text = f"🔥 *PHOENIX PROTOCOL* 🔥\n\n*Service:* `{service}`\n*Event:* {event}\n\n_Autonomous recovery in progress._"
        await self.send_message(text)
