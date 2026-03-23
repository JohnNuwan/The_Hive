import requests
import logging
import os
import asyncio
from typing import Optional
from shared.config import get_settings

logger = logging.getLogger(__name__)

class TelegramClient:
    """
    Client for sending notifications to a Telegram Channel/Group.
    Supports Topics (Threads) via message_thread_id.
    """
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None, topic_id: Optional[int] = None):
        self.settings = get_settings()
        self.token = token or self.settings.telegram_bot_token
        self.chat_id = chat_id or self.settings.telegram_chat_id
        # Topic ID (Thread) support - Reading from .env directly as it might not be in Settings schema yet
        # Or even better, let's stick to os.getenv for the NEW optional var
        env_topic = os.getenv("TELEGRAM_TOPIC_ID")
        self.topic_id = topic_id if topic_id else (int(env_topic) if env_topic else None)
        
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        
        if not self.token or not self.chat_id:
            logger.warning(f"⚠️ Telegram credentials missing. Token={bool(self.token)} Chat={bool(self.chat_id)}")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"📢 Telegram Bot initialized (Chat: {self.chat_id} | Topic: {self.topic_id})")

    def _send_sync_internal(self, message: str):
        """Internal synchronous method to send a message."""
        if not self.enabled: return
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        if self.topic_id:
            payload["message_thread_id"] = self.topic_id

        try:
            resp = requests.post(self.base_url, json=payload, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Telegram Send Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Telegram Connection Error: {e}")

    async def send_message(self, message: str):
        """Sends a text message to the configured chat/topic."""
        if not self.enabled: return
        await asyncio.to_thread(self._send_sync_internal, message)

    def _send_photo_sync_internal(self, photo: bytes, caption: str):
        if not self.enabled: return
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        
        data = {"chat_id": self.chat_id, "caption": caption, "parse_mode": "Markdown"}
        if self.topic_id:
            data["message_thread_id"] = self.topic_id
            
        files = {"photo": ("image.png", photo, "image/png")}
        
        try:
            resp = requests.post(url, data=data, files=files, timeout=30)
            if resp.status_code != 200:
                logger.error(f"Telegram SendPhoto Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Telegram Connection Error: {e}")

    async def send_photo(self, photo: bytes, caption: str):
        """Sends a photo to the configured chat/topic."""
        if not self.enabled: return
        await asyncio.to_thread(self._send_photo_sync_internal, photo, caption)

    def send_sync(self, message: str):
        """Synchronous wrapper for sending messages (fire & forget)."""
        if not self.enabled: return
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._send_sync_internal, message)
        except RuntimeError:
            self._send_sync_internal(message)
        except Exception as e:
            logger.error(f"Telegram Sync Error: {e}")
