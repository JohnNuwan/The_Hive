"""Tests du client Telegram partage."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from shared.telegram_client import TelegramClient


class TelegramClientTests(unittest.TestCase):
    """Verifie le mode texte brut par defaut pour Telegram."""

    @patch("shared.telegram_client.get_settings")
    @patch("shared.telegram_client.requests.post")
    def test_send_sync_omits_parse_mode_by_default(
        self,
        post_mock: MagicMock,
        settings_mock: MagicMock,
    ) -> None:
        """N'envoie pas de `parse_mode` si aucun format n'est demande."""

        settings_mock.return_value = SimpleNamespace(
            telegram_bot_token="token",
            telegram_chat_id="chat",
        )
        post_mock.return_value = SimpleNamespace(status_code=200, text="ok")

        client = TelegramClient()
        client._send_sync_internal("manual_muzero_scalp_multi_universe_full")

        payload = post_mock.call_args.kwargs["json"]
        self.assertNotIn("parse_mode", payload)
        self.assertEqual(payload["text"], "manual_muzero_scalp_multi_universe_full")

    @patch("shared.telegram_client.get_settings")
    @patch("shared.telegram_client.requests.post")
    def test_send_sync_preserves_explicit_parse_mode(
        self,
        post_mock: MagicMock,
        settings_mock: MagicMock,
    ) -> None:
        """Conserve un mode de parse explicite quand il est demande."""

        settings_mock.return_value = SimpleNamespace(
            telegram_bot_token="token",
            telegram_chat_id="chat",
        )
        post_mock.return_value = SimpleNamespace(status_code=200, text="ok")

        client = TelegramClient()
        client._send_sync_internal("message", parse_mode="HTML")

        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(payload["parse_mode"], "HTML")

    @patch("shared.telegram_client.get_settings")
    @patch("shared.telegram_client.requests.post")
    def test_send_sync_repairs_common_mojibake_text(
        self,
        post_mock: MagicMock,
        settings_mock: MagicMock,
    ) -> None:
        """Repare les chaines UTF-8 decodees a tort avant l'envoi Telegram."""

        settings_mock.return_value = SimpleNamespace(
            telegram_bot_token="token",
            telegram_chat_id="chat",
        )
        post_mock.return_value = SimpleNamespace(status_code=200, text="ok")

        client = TelegramClient()
        client._send_sync_internal("📈 *E.V.A | Bilan AprÃ¨s-Midi*")

        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(payload["text"], "📈 *E.V.A | Bilan Après-Midi*")

    @patch("shared.telegram_client.get_settings")
    @patch("shared.telegram_client.requests.post")
    def test_send_photo_repairs_caption_mojibake_text(
        self,
        post_mock: MagicMock,
        settings_mock: MagicMock,
    ) -> None:
        """Repare egalement les captions d'images Telegram."""

        settings_mock.return_value = SimpleNamespace(
            telegram_bot_token="token",
            telegram_chat_id="chat",
        )
        post_mock.return_value = SimpleNamespace(status_code=200, text="ok")

        client = TelegramClient()
        client._send_photo_sync_internal(b"png", "NEWS FILTER ACTIVÃ‰")

        payload = post_mock.call_args.kwargs["data"]
        self.assertEqual(payload["caption"], "NEWS FILTER ACTIVÉ")


if __name__ == "__main__":
    unittest.main()
