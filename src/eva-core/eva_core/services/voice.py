"""
Voice Service — Interface vocale (STT / TTS) pour EVA
═════════════════════════════════════════════════════

Fonctionnalités :
- Transcription audio → texte (Speech-to-Text via SpeechRecognition / Whisper)
- Synthèse texte → audio (Text-to-Speech stub, extensible vers Coqui/Piper)

Les imports sont conditionnels pour permettre le fonctionnement sans les deps lourdes.
"""

import asyncio
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS CONDITIONNELS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None


class VoiceService:
    """
    Service d'interface vocale pour EVA.

    Si SpeechRecognition est installé, la transcription fonctionne.
    Sinon, le service retourne des messages d'erreur gracieux.
    """

    def __init__(self):
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            logger.info("✅ Voice Service initialisé (SpeechRecognition disponible)")
        else:
            self.recognizer = None
            logger.warning(
                "⚠️ SpeechRecognition non installé. "
                "Le service vocal est en mode dégradé."
            )

    @property
    def is_available(self) -> bool:
        return self.recognizer is not None

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcrit un flux audio en texte.

        Args:
            audio_data: Bytes bruts du fichier audio (WAV, FLAC, etc.)

        Returns:
            Le texte transcrit, ou un message d'erreur.
        """
        if not self.is_available:
            return "[Service vocal désactivé — SpeechRecognition non installé]"

        try:
            # Exécuter la transcription dans un thread séparé (CPU-bound)
            text = await asyncio.to_thread(self._transcribe_sync, audio_data)
            logger.info(f"🎤 Transcription: '{text[:80]}...'")
            return text
        except Exception as e:
            logger.error(f"Erreur transcription: {e}")
            return f"[Erreur de transcription: {e}]"

    def _transcribe_sync(self, audio_data: bytes) -> str:
        """Transcription synchrone (exécutée dans un thread)."""
        audio_file = io.BytesIO(audio_data)
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)

        try:
            # Utiliser Google Speech Recognition (gratuit, limité)
            text = self.recognizer.recognize_google(audio, language="fr-FR")
            return text
        except sr.UnknownValueError:
            return "[Audio incompréhensible]"
        except sr.RequestError as e:
            return f"[Erreur service de reconnaissance: {e}]"

    async def synthesize_speech(self, text: str) -> bytes:
        """
        Synthétise du texte en audio (TTS).

        Stub actuel — en production, connecter à Coqui TTS, Piper, ou gTTS.

        Args:
            text: Le texte à convertir en parole.

        Returns:
            Bytes audio (format WAV).
        """
        logger.info(f"🔊 TTS (stub): '{text[:60]}...'")
        # Placeholder — retourne un silence WAV minimal
        # En production : utiliser gTTS, Coqui TTS, ou Piper
        await asyncio.sleep(0.1)
        return b""  # Vide — le frontend détecte et affiche le texte
