"""Client Telegram minimal pour les notifications THE HIVE."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from typing import Optional

import requests

from shared.config import get_settings

logger = logging.getLogger(__name__)

_MOJIBAKE_MARKERS = (
    "Ã",
    "â",
    "€",
    "œ",
    "ž",
    "Ÿ",
    "ï",
    "‰",
    "™",
    "š",
    "‹",
    "›",
    "“",
    "”",
    "•",
    "–",
    "—",
)


def _count_mojibake_markers(text: str) -> int:
    """Compte les marqueurs usuels de texte UTF-8 mal decode.

    Args:
        text (str): Texte a analyser.

    Returns:
        int: Nombre de marqueurs detectes.
    """
    return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)


def _attempt_redecode_text(text: str, encoding: str) -> str | None:
    """Tente de reparer un texte UTF-8 decode avec le mauvais codec.

    Args:
        text (str): Texte possiblement corrompu.
        encoding (str): Codec a reutiliser pour reconstituer les octets.

    Returns:
        str | None: Texte repare si la tentative reussit, sinon ``None``.
    """
    try:
        return text.encode(encoding).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def _normalize_telegram_text(text: str) -> str:
    """Normalise un texte Telegram avant envoi.

    Cette normalisation corrige le cas courant d'un texte UTF-8 lu comme
    ``latin-1`` ou ``cp1252``. Si aucune amelioration nette n'est detectee,
    le texte d'origine est conserve.

    Args:
        text (str): Texte brut a transmettre.

    Returns:
        str: Texte nettoye pour Telegram.
    """
    normalized = str(text or "")
    if _count_mojibake_markers(normalized) == 0:
        return normalized

    chunks: list[str] = []
    current_chunk: list[str] = []
    allowed_marker_chars = set(_MOJIBAKE_MARKERS)

    def flush_chunk() -> None:
        if not current_chunk:
            return
        segment = "".join(current_chunk)
        current_chunk.clear()
        baseline_score = _count_mojibake_markers(segment)
        if baseline_score == 0:
            chunks.append(segment)
            return

        best_candidate = segment
        best_score = baseline_score
        for encoding in ("latin-1", "cp1252"):
            candidate = _attempt_redecode_text(segment, encoding)
            if candidate is None:
                continue
            candidate_score = _count_mojibake_markers(candidate)
            if candidate_score < best_score:
                best_candidate = candidate
                best_score = candidate_score
        chunks.append(best_candidate)

    for char in normalized:
        if ord(char) <= 255 or char in allowed_marker_chars:
            current_chunk.append(char)
            continue
        flush_chunk()
        chunks.append(char)

    flush_chunk()
    return "".join(chunks)


class TelegramClient:
    """Envoie des messages Telegram vers un chat ou un topic configure.

    Le mode de parse est desactive par defaut pour eviter les erreurs de
    rendu sur les identifiants techniques contenant des underscores.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        topic_id: Optional[int] = None,
    ) -> None:
        """Initialise le client Telegram.

        Args:
            token (Optional[str]): Token du bot a utiliser.
            chat_id (Optional[str]): Identifiant du chat cible.
            topic_id (Optional[int]): Identifiant du topic cible.
        """
        self.settings = get_settings()
        self.token = token or self.settings.telegram_bot_token
        self.chat_id = chat_id or self.settings.telegram_chat_id
        env_topic = os.getenv("TELEGRAM_TOPIC_ID")
        self.topic_id = topic_id if topic_id else (int(env_topic) if env_topic else None)
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

        if not self.token or not self.chat_id:
            logger.warning(
                "Identifiants Telegram absents. Token=%s Chat=%s",
                bool(self.token),
                bool(self.chat_id),
            )
            self.enabled = False
        else:
            self.enabled = True
            logger.info(
                "Telegram initialise (chat=%s, topic=%s).",
                self.chat_id,
                self.topic_id,
            )

    def _build_message_payload(
        self,
        message: str,
        *,
        parse_mode: str | None = None,
    ) -> dict[str, object]:
        """Construit la charge utile d'un message texte.

        Args:
            message (str): Corps du message.
            parse_mode (str | None): Mode de rendu Telegram explicite.

        Returns:
            dict[str, object]: Charge utile prete pour l'API Telegram.
        """
        payload: dict[str, object] = {
            "chat_id": self.chat_id,
            "text": _normalize_telegram_text(message),
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if self.topic_id:
            payload["message_thread_id"] = self.topic_id
        return payload

    def _send_sync_internal(
        self,
        message: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        """Envoie un message texte de facon synchrone.

        Args:
            message (str): Texte a transmettre.
            parse_mode (str | None): Mode de rendu Telegram optionnel.
        """
        if not self.enabled:
            return

        payload = self._build_message_payload(message, parse_mode=parse_mode)
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(
                    "Erreur Telegram sendMessage: %s - %s",
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.error("Erreur de connexion Telegram: %s", exc)

    async def send_message(
        self,
        message: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        """Envoie un message texte asynchrone.

        Args:
            message (str): Texte a transmettre.
            parse_mode (str | None): Mode de rendu Telegram optionnel.
        """
        if not self.enabled:
            return
        await asyncio.to_thread(
            self._send_sync_internal,
            message,
            parse_mode=parse_mode,
        )

    def _send_photo_sync_internal(
        self,
        photo: bytes,
        caption: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        """Envoie une photo de facon synchrone.

        Args:
            photo (bytes): Contenu binaire de l'image.
            caption (str): Legende associee.
            parse_mode (str | None): Mode de rendu Telegram optionnel.
        """
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        data: dict[str, object] = {
            "chat_id": self.chat_id,
            "caption": _normalize_telegram_text(caption),
        }
        if parse_mode:
            data["parse_mode"] = parse_mode
        if self.topic_id:
            data["message_thread_id"] = self.topic_id
        files = {"photo": ("image.png", photo, "image/png")}

        try:
            response = requests.post(url, data=data, files=files, timeout=30)
            if response.status_code != 200:
                logger.error(
                    "Erreur Telegram sendPhoto: %s - %s",
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.error("Erreur de connexion Telegram: %s", exc)

    async def send_photo(
        self,
        photo: bytes,
        caption: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        """Envoie une photo asynchrone.

        Args:
            photo (bytes): Contenu binaire de l'image.
            caption (str): Legende associee.
            parse_mode (str | None): Mode de rendu Telegram optionnel.
        """
        if not self.enabled:
            return
        await asyncio.to_thread(
            self._send_photo_sync_internal,
            photo,
            caption,
            parse_mode=parse_mode,
        )

    def send_sync(
        self,
        message: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        """Declenche un envoi synchrone tolérant a l'absence de boucle.

        Args:
            message (str): Texte a transmettre.
            parse_mode (str | None): Mode de rendu Telegram optionnel.
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
                    parse_mode=parse_mode,
                ),
            )
        except RuntimeError:
            self._send_sync_internal(message, parse_mode=parse_mode)
        except Exception as exc:
            logger.error("Erreur Telegram synchrone: %s", exc)
