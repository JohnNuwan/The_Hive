#!/usr/bin/env python3
"""THE HIVE — Bot Passerelle de Discussion avec l'IA E.V.A.

Ce script agit comme un pont bidirectionnel entre le salon Discord `#parler-avec-eva`
et le coordinateur cognitif Hermes. Il écoute les messages de l'opérateur,
les transmet à Hermes avec un prompt de personnalité supervisée (E.V.A),
et renvoie les réponses formatées en Embeds de couleur Vert Menthe.
"""

from __future__ import annotations

import os
import sys
import logging
import requests
from pathlib import Path

# UTF-8 encoding support for Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Chargement de dotenv si disponible
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("discord_eva_bot")

# Vérification de la disponibilité du package discord.py
try:
    import discord
except ImportError:
    logger.error("=" * 60)
    logger.error("Le package 'discord.py' est requis pour faire fonctionner ce bot.")
    logger.error("Veuillez l'installer dans votre environnement local avec la commande :")
    logger.error("      pip install discord.py requests python-dotenv")
    logger.error("=" * 60)
    sys.exit(1)

# Variables de configuration
HERMES_CHAT_URL = os.getenv("HERMES_CHAT_URL", "http://192.168.1.6:9500/chat")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
EVA_CHANNEL_ID = int(os.getenv("DISCORD_EVA_CHANNEL_ID", "1504560956787921057"))

SYSTEM_PROMPT = (
    "Tu es E.V.A (Electronic Virtual Assistant), l'intelligence artificielle centrale "
    "et la superviseure suprême de la flotte de trading autonome THE HIVE.\n"
    "Tu es en contact direct et confidentiel avec l'opérateur humain du système.\n"
    "Réponds de manière sereine, hautement stratégique, concise, intelligente et "
    "professionnelle en Français. Tu as accès aux rapports des autres experts (technique, "
    "trading, macro, dev) et tu coordonnes les briques autonomes."
)


def fetch_hive_training_status() -> dict[str, Any]:
    """Se connecte en SSH au serveur Proxmox pour récupérer le statut réel des entraînements."""
    import paramiko
    import json
    import re
    
    result = {
        "status": "inconnu",
        "current_symbol": "inconnu",
        "progress": "inconnu",
        "buffer_entries": 0,
        "gnn_status": "inconnu"
    }
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect('192.168.1.6', username='aza', password='Kumara-42/600', timeout=5)
        
        # 1. Lecture de nightly_training_summary.json
        stdin, stdout, stderr = ssh.exec_command('cat /home/aza/The_Hive/data/checkpoints/nightly_training_summary.json')
        summary_content = stdout.read().decode('utf-8', errors='replace').strip()
        if summary_content:
            try:
                summary = json.loads(summary_content)
                result["status"] = summary.get("status", "running")
                steps = summary.get("steps", [])
                for step in steps:
                    if step.get("name") == "gnn":
                        result["gnn_status"] = step.get("status", "ok")
            except Exception:
                pass

        # 2. Lecture de training_status.json
        stdin, stdout, stderr = ssh.exec_command('cat /home/aza/The_Hive/data/checkpoints/training_status.json')
        status_content = stdout.read().decode('utf-8', errors='replace').strip()
        if status_content:
            try:
                status = json.loads(status_content)
                result["buffer_entries"] = status.get("replay_cache_entries", 0)
            except Exception:
                pass
                
        # 3. Lecture du fichier de log de collecte pour trouver la partie en cours
        cmd_log = "tail -n 100 /home/aza/The_Hive/data/checkpoints/training_run.log | grep -a 'partie' | tail -n 1"
        stdin, stdout, stderr = ssh.exec_command(cmd_log)
        log_line = stdout.read().decode('utf-8', errors='replace').strip()
        if log_line:
            match = re.search(r'([A-Za-z0-9\.\_]+)\s+partie\s+(\d+/\d+)', log_line)
            if match:
                result["current_symbol"] = match.group(1)
                result["progress"] = match.group(2)
                
        # Fallback pour le symbole/partie si le log n'est pas dispo
        if result["current_symbol"] == "inconnu":
            stdin, stdout, stderr = ssh.exec_command('cat /home/aza/The_Hive/data/checkpoints/training_digest_state.json')
            digest_content = stdout.read().decode('utf-8', errors='replace').strip()
            if digest_content:
                try:
                    digest = json.loads(digest_content)
                    curr_step = digest.get("snapshot", {}).get("current_step", {})
                    if curr_step:
                        result["current_symbol"] = curr_step.get("symbol", "inconnu")
                        result["progress"] = f"{curr_step.get('part_index', 0)}/{curr_step.get('part_total', 100)}"
                except Exception:
                    pass

    except Exception as exc:
        logger.error("Erreur de connexion SSH lors de la récupération du statut : %s", exc)
    finally:
        ssh.close()
        
    return result


class EVABotClient(discord.Client):
    """Client de Gateway Discord pour le dialogue bidirectionnel avec E.V.A."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise le client de discussion E.V.A."""
        intents = discord.Intents.default()
        intents.message_content = True  # Requis pour lire le contenu des messages
        super().__init__(*args, intents=intents, **kwargs)

    async def on_ready(self) -> None:
        """Déclenché lorsque le bot se connecte avec succès à la Gateway Discord."""
        logger.info("=" * 40)
        logger.info("🌿 E.V.A CONNECTÉE ET PRÊTE À DIALOGUER")
        logger.info("Nom d'utilisateur Bot : %s", self.user)
        logger.info("Salon ciblé (ID) : %d", EVA_CHANNEL_ID)
        logger.info("Expert cognitif sous-jacent : %s", HERMES_CHAT_URL)
        logger.info("=" * 40)
        
        # Définition de l'activité sur Discord
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching, 
                name="la flotte THE HIVE"
            )
        )

        # Vérification proactive du dernier message non répondu au démarrage
        try:
            channel = self.get_channel(EVA_CHANNEL_ID)
            if not channel:
                channel = await self.fetch_channel(EVA_CHANNEL_ID)
            
            if channel:
                logger.info("Recherche proactive de messages récents non répondus dans #%s...", channel.name)
                async for message in channel.history(limit=10):
                    # Trouver le dernier message d'un humain
                    if not message.author.bot:
                        # Si le dernier message du salon est de l'opérateur et n'a pas reçu de réponse d'un bot après lui
                        has_bot_reply = False
                        async for newer_msg in channel.history(after=message, limit=10):
                            if newer_msg.author.bot:
                                has_bot_reply = True
                                break
                        
                        if not has_bot_reply:
                            logger.info("Trouvé un message non répondu de %s: %s", message.author.name, message.content)
                            # Simuler la réception de ce message pour le traiter
                            await self.on_message(message)
                        else:
                            logger.info("Le dernier message de %s a déjà reçu une réponse.", message.author.name)
                        break
            else:
                logger.warning("Impossible de trouver le salon avec l'ID %d", EVA_CHANNEL_ID)
        except Exception as exc:
            logger.error("Erreur lors de la vérification proactive des messages : %s", exc)

    async def on_message(self, message: discord.Message) -> None:
        """Déclenché lors de la réception d'un nouveau message sur le serveur.

        Args:
            message (discord.Message): L'objet message reçu.
        """
        # Ne pas répondre à soi-même ou à d'autres bots
        if message.author.bot or message.author == self.user:
            return

        # Filtrer uniquement les messages du salon dédié à E.V.A
        if message.channel.id != EVA_CHANNEL_ID:
            return

        logger.info("Message reçu de l'opérateur %s : %s", message.author.name, message.content)

        # Déclenchement du statut "En train d'écrire..." pour le réalisme et l'attente
        async with message.channel.typing():
            try:
                # Récupération proactive des données réelles du système en tâche de fond
                import asyncio
                training_status = await asyncio.to_thread(fetch_hive_training_status)
                
                # Formatage du contexte technique pour l'injection
                system_context_str = (
                    f"Statut de l'entraînement : {training_status['status']}\n"
                    f"Étape GNN : {training_status['gnn_status']}\n"
                    f"Symbole actif MuZero Scalp : {training_status['current_symbol']}\n"
                    f"Progression de la collecte : {training_status['progress']}\n"
                    f"Nombre d'épisodes en cache JAX : {training_status['buffer_entries']}"
                )
                
                # Injection dans le prompt système pour guider la personnalité d'E.V.A.
                enriched_system_prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"### STATUT EN TEMPS RÉEL DU SYSTÈME THE HIVE (À utiliser obligatoirement pour répondre) :\n"
                    f"{system_context_str}"
                )

                # Préparation de la requête pour Hermes
                payload = {
                    "message": message.content,
                    "expert": "coordinator",
                    "system_prompt": enriched_system_prompt,
                    "context": {
                        "operator_name": message.author.display_name,
                        "channel_name": message.channel.name,
                    },
                    "temperature": 0.3,
                    "max_tokens": 1200
                }

                # Appel asynchrone tolérant à la latence du LLM local
                import asyncio
                response = await asyncio.to_thread(
                    requests.post, 
                    HERMES_CHAT_URL, 
                    json=payload, 
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    eva_reply = result.get("message", result.get("response", "Aucune réponse reçue."))
                else:
                    logger.error("Erreur API Hermes (%d) : %s", response.status_code, response.text)
                    eva_reply = f"⚠️ *Désolée opérateur, l'API Hermes a retourné une erreur (HTTP {response.status_code}).*"

            except Exception as exc:
                logger.error("Échec de connexion au noyau cognitif Hermes : %s", exc)
                eva_reply = "⚠️ *Désolée opérateur, je ne parviens pas à joindre mon noyau cognitif Hermes pour le moment.*"

        # Construction d'un Embed Vert Menthe élégant pour la réponse d'E.V.A
        embed = discord.Embed(
            title="🌿 E.V.A — SUPERVISOR RESPONSE",
            description=eva_reply,
            color=0x10AC84, # Vert menthe néon de la charte E.V.A
            timestamp=discord.utils.utcnow()
        )
        embed.set_footer(
            text=f"Dialogue direct • Opérateur : {message.author.display_name}",
            icon_url=message.author.display_avatar.url if message.author.display_avatar else None
        )

        try:
            await message.reply(embed=embed)
            logger.info("Réponse envoyée avec succès à %s.", message.author.name)
        except Exception as exc:
            logger.error("Impossible de publier la réponse sur Discord : %s", exc)


def main() -> None:
    """Routine d'entrée du Bot Passerelle E.V.A."""
    if not DISCORD_BOT_TOKEN:
        logger.error("=" * 60)
        logger.error("La variable 'DISCORD_BOT_TOKEN' est vide dans l'environnement / .env.")
        logger.error("Pour lier E.V.A :")
        logger.error("1. Créez une application sur https://discord.com/developers/applications")
        logger.error("2. Créez un Bot, donnez-lui l'intent 'MESSAGE_CONTENT'.")
        logger.error("3. Copiez son TOKEN et insérez-le dans votre fichier .env comme ceci :")
        logger.error("      DISCORD_BOT_TOKEN=\"votre_token_secret\"")
        logger.error("4. Relancez ensuite ce script.")
        logger.error("=" * 60)
        sys.exit(1)

    # Lancement du client de Gateway Discord
    client = EVABotClient()
    try:
        client.run(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.error("Échec d'authentification : Le token DISCORD_BOT_TOKEN fourni est invalide.")
    except Exception as exc:
        logger.error("Erreur d'exécution du bot : %s", exc)


if __name__ == "__main__":
    main()
