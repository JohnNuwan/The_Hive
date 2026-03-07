import asyncio
import os
import json
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from shared import get_settings
from shared.llm_client import LLMClient
from eva_muse.services.comfy_client import ComfyUIClient

logger = logging.getLogger(__name__)

class MuseScheduler:
    """
    Orchestrateur asynchrone pour générer du contenu automatiquement à intervalles réguliers.
    Mode actuel : "Sandboxed" -> Le texte et les images sont sauvegardés localement.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler()
        self.llm_client = LLMClient()
        self.comfy_client = ComfyUIClient()
        self.output_dir = "/mnt/black_box/muse_review" if os.path.exists("/mnt/black_box") else os.path.join(os.getcwd(), "data", "muse_review")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def start(self):
        """Démarre le planificateur."""
        # 1. Routine "Lifestyle/Finance" toutes les 4 heures
        self.scheduler.add_job(
            self.generate_standard_content,
            trigger="interval",
            hours=4,
            id="muse_standard_gen"
        )
        
        # 2. Routine "Adult/NSFW" (OnlyFans) toutes les 6 heures (Seulement si activé)
        # On passe un booléen pour garder l'option de désactiver facilement
        self.scheduler.add_job(
            self.generate_nsfw_content,
            trigger="interval",
            hours=6,
            id="muse_nsfw_gen"
        )
        
        self.scheduler.start()
        logger.info(f"📅 Muse Scheduler started. Out directory: {self.output_dir}")
        
    def stop(self):
        self.scheduler.shutdown()
        logger.info("🛑 Muse Scheduler stopped.")
        
    async def generate_standard_content(self):
        """Génère du contenu standard (Twitter/Insta Lifestyle/Trading)"""
        logger.info("🏭 Démarrage routine Standard Content...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "STANDARD"
        
        # 1. Appeler le LLM pour un concept d'image et un copy
        prompt_sys = (
            "Tu es Muse, la directrice artistique de l'influenceuse virtuelle Athena (une belle tradeuse crypto de 25 ans). "
            "Génère un concept de photo (en anglais brut pour l'IA d'image) et un texte de post Twitter/X (en français, max 200 rep) "
            "Renvoie UNIQUEMENT un JSON avec les clés 'image_prompt' et 'caption'."
        )
        prompt_user = "Crée un post sur le BTC qui s'approche d'un nouveau sommet, photo d'Athena au travail."
        
        try:
            response = await self.llm_client.analyze(context=prompt_sys, prompt=prompt_user)
            # Extraire le JSON (Le LLM peut mettre du markdown autour)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
                
            data = json.loads(json_str)
            image_prompt = data.get("image_prompt", "Athena, 25 years old, crypto trader, looking at charts, realistic, 8k, bokeh")
            caption = data.get("caption", "Prête pour le nouveau sommet du BTC ! 🚀 #Crypto #Trading")
            
        except Exception as e:
            logger.error(f"Erreur LLM génération: {e}")
            image_prompt = "Athena, 25 years old, crypto trader, looking at charts, realistic, 8k, bokeh"
            caption = "Prête pour le nouveau sommet du BTC ! 🚀 #Crypto #Trading"

        await self._render_and_save(prefix, timestamp, image_prompt, caption, is_nsfw=False)


    async def generate_nsfw_content(self):
        """Génère du contenu privé / HOT (ex: Telegram VIP / OnlyFans)"""
        logger.info("🔥 Démarrage routine NSFW Content...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "NSFW"
        
        # 1. Appeler le LLM pour un concept d'image HOT et un copy
        prompt_sys = (
            "Tu es Muse, la directrice artistique de l'influenceuse virtuelle Athena. "
            "Génère un concept de photo très suggestif / sexy pour contenu privé (en anglais brut pour l'IA d'image, insiste sur la lingerie, pose suggestive, etc) "
            "et un texte aguicheur pour ses fans (en français). "
            "Renvoie UNIQUEMENT un JSON avec les clés 'image_prompt' et 'caption'."
        )
        prompt_user = "Crée un post privé du soir pour remercier les abonnés."
        
        try:
            response = await self.llm_client.analyze(context=prompt_sys, prompt=prompt_user)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
                
            data = json.loads(json_str)
            image_prompt = data.get("image_prompt", "Athena, 25 years old, wearing seductive lingerie, bedroom, moody lighting, realistic, 8k, intimate")
            caption = data.get("caption", "Bonne nuit mes abeilles... merci pour tout ce soutien. 💋")
            
        except Exception as e:
            logger.error(f"Erreur LLM génération NSFW: {e}")
            image_prompt = "Athena, 25 years old, wearing sexy lingerie, luxury bedroom, moody lighting, realistic, intimate, 8k"
            caption = "Bonne nuit mes abeilles... merci pour tout ce soutien. 💋"

        await self._render_and_save(prefix, timestamp, image_prompt, caption, is_nsfw=True)


    async def _render_and_save(self, prefix: str, timestamp: str, prompt: str, caption: str, is_nsfw: bool):
        """Envoie le prompt à ComfyUI et sauvegarde les résultats au format Review."""
        logger.info(f"[{prefix}] Impression de l'image (Prompt: {prompt})")

        # Sauver d'abord le texte pour pouvoir le lire même si ComfyUI Crash (ex: Serveur éteint)
        base_filename = os.path.join(self.output_dir, f"{timestamp}_{prefix}")
        with open(f"{base_filename}.txt", "w", encoding="utf-8") as f:
            f.write(f"--- CAPTION ---\n{caption}\n\n--- PROMPT ---\n{prompt}")

        # 2. Construire le Workflow ComfyUI (Basic Text2Image pour le moment)
        workflow = self._build_basic_workflow(prompt)
        
        # 3. Récupérer l'image
        try:
            images = await self.comfy_client.generate_from_workflow(workflow)
            if not images:
                logger.error(f"[{prefix}] ComfyUI n'a renvoyé aucune image.")
                return
                
            img_bytes = images[0]
            
            # 4. Sauvegarder l'image
            with open(f"{base_filename}.png", "wb") as f:
                f.write(img_bytes)
                
            logger.info(f"✅ [{prefix}] Contenu multimédia généré et sauvegardé pour Review: {base_filename}")

        except Exception as e:
            logger.error(f"[{prefix}] Erreur lors de la génération ComfyUI (Image annulée): {e}")

    def _build_basic_workflow(self, prompt: str) -> dict:
        """Génère un workflow ComfyUI simpliste de Text2Image"""
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(datetime.now().timestamp()), # Random seed
                    "steps": 25,
                    "cfg": 7,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"} # À modifier selon le modèle sur ton serveur
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "height": 1024, "width": 1024}
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 1]}
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "ugly, bad anatomy, bad hands, fake, cartoon, drawing, painting", "clip": ["4", 1]}
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "MuseFactory", "images": ["8", 0]}
            }
        }
