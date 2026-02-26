import os
import re
import json
import logging
from datetime import datetime
from pydantic import BaseModel

import aiohttp
from openclaw.skills.git_ops import git_commit

logger = logging.getLogger(__name__)

class CodeRequest(BaseModel):
    prompt: str
    filename: str
    language: str = "python"


class CodeFactoryService:
    """
    Service responsable de la génération de code autonome (The Code Factory).
    Dialogue avec le LLM via eva-core (ou Ollama local) pour transformer un
    cahier des charges en fichier source brut, l'enregistrer et le commiter.
    """
    def __init__(self):
        self.output_dir = os.path.join(os.getcwd(), "data", "factory_output")
        os.makedirs(self.output_dir, exist_ok=True)
        # Using the centralized Debian 13 server for Ollama LLM
        self.llm_url = "http://192.168.1.5:11434/api/generate"
        self.model = "gemma3:4b" # Standard E.V.A programming model

    async def generate_code(self, request: CodeRequest) -> dict:
        """
        Envoie le prompt au LLM, extrait le code, le sauvegarde et le commit.
        """
        logger.info(f"🏭 Code Factory : Démarrage du job [{request.filename}]")
        
        system_prompt = (
            f"Tu es E.V.A., une développeuse Senior. Écris uniquement du code {request.language} "
            "complet, fonctionnel et de production. Pas de bla-bla. "
            "Enferme ton code dans un bloc markdown (```python ... ```)."
        )
        
        full_prompt = f"{system_prompt}\n\nCahier des charges:\n{request.prompt}"
        
        try:
            # 1. Génération par le LLM (Ollama)
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False
                }
                async with session.post(self.llm_url, json=payload, timeout=120) as resp:
                    if resp.status != 200:
                        return {"status": "error", "message": f"LLM backend failed: {resp.status}"}
                    
                    data = await resp.json()
                    response_text = data.get("response", "")
            
            # 2. Extraction du code (Regex)
            # Match ```python \n code... \n ```
            pattern = rf"```(?:{request.language})?\s*(.*?)```"
            matches = re.findall(pattern, response_text, re.DOTALL)
            
            if not matches:
                # Fallback: take the whole response if no markdown blocks are found
                clean_code = response_text.strip()
            else:
                clean_code = matches[0].strip()
                
            # 3. Sauvegarde Disque
            file_path = os.path.join(self.output_dir, request.filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(clean_code)
                
            logger.info(f"✅ Fichier généré avec succès : {file_path}")
            
            # 4. GitOps : Commit Automatique (via OpenClaw Skill)
            commit_msg = f"Factory: Generated {request.filename} (Auto-Coded)"
            git_result = await git_commit(commit_msg)
            
            return {
                "status": "success",
                "filename": request.filename,
                "path": file_path,
                "git_status": git_result
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur Code Factory : {e}")
            return {"status": "error", "message": str(e)}
