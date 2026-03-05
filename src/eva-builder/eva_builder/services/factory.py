import os
import re
import json
import logging
import asyncio
from datetime import datetime
from pydantic import BaseModel

import aiohttp
from openclaw.skills.git_ops import git_commit
from eva_builder.services.bmad_prompts import PM_PROMPT, ARCHITECT_PROMPT, DEVELOPER_PROMPT

logger = logging.getLogger(__name__)

class CodeRequest(BaseModel):
    prompt: str
    filename: str
    language: str = "python"


class CodeFactoryService:
    """
    Service responsable de la génération de code autonome (The Code Factory).
    Utilise la méthodologie BMAD (Brainstorm, Map, Act, Document) via 
    une Digital Factory multi-agents séquentielle (PM -> Architect -> Developer).
    """
    def __init__(self):
        self.output_dir = os.path.join(os.getcwd(), "data", "factory_output")
        os.makedirs(self.output_dir, exist_ok=True)
        # Using the centralized Debian 13 server for Ollama LLM
        self.llm_url = "http://192.168.1.5:11434/api/generate"
        self.model = "gemma3:4b" # Standard E.V.A programming model

    async def _call_llm(self, system_prompt: str, user_prompt: str, session: aiohttp.ClientSession) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False
        }
        async with session.post(self.llm_url, json=payload, timeout=300) as resp:
            if resp.status != 200:
                raise Exception(f"LLM backend failed: {resp.status}")
            data = await resp.json()
            return data.get("response", "")

    async def generate_code(self, request: CodeRequest) -> dict:
        """
        Génère un Micro-SaaS ou un script via la pipeline BMAD.
        Phase 1: PM (Product Requirement Document)
        Phase 2: Architect (Document d'Architecture Technique)
        Phase 3: Developer (Implémentation)
        """
        logger.info(f"🏭 Code Factory : Démarrage du job [{request.filename}] (BMAD Pipeline)")
        
        project_name = request.filename.split('.')[0] if '.' in request.filename else "project"
        project_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        try:
            async with aiohttp.ClientSession() as session:
                # ─── PHASE 1 : Product Manager (PRD) ───
                logger.info("🏭 BMAD Phase 1: Product Manager (Drafting PRD)...")
                pm_user = f"Cahier des charges initial:\n{request.prompt}"
                prd_text = await self._call_llm(PM_PROMPT, pm_user, session)
                
                prd_path = os.path.join(project_dir, "prd.md")
                with open(prd_path, "w", encoding="utf-8") as f:
                    f.write(prd_text)
                
                # ─── PHASE 2 : Architect (DAT) ───
                logger.info("🏭 BMAD Phase 2: Architect (Drafting DAT)...")
                arch_user = f"Voici le Product Requirement Document (PRD):\n\n{prd_text}"
                dat_text = await self._call_llm(ARCHITECT_PROMPT, arch_user, session)
                
                dat_path = os.path.join(project_dir, "architecture.md")
                with open(dat_path, "w", encoding="utf-8") as f:
                    f.write(dat_text)

                # ─── PHASE 3 : Developer (Code) ───
                logger.info("🏭 BMAD Phase 3: Developer (Coding)...")
                dev_user = (
                    f"Voici le PRD:\n{prd_text}\n\n"
                    f"Voici le DAT:\n{dat_text}\n\n"
                    f"Tâche : Écris le code source complet ({request.language}) "
                    f"correspondant à ces documents. Nom du fichier principal attendu : {request.filename}"
                )
                dev_text = await self._call_llm(DEVELOPER_PROMPT, dev_user, session)
                
                # Extraction du code (Regex)
                pattern = r"```(?:[\w+-]+)?\s*\n(.*?)```"
                matches = re.findall(pattern, dev_text, re.DOTALL)
                
                if not matches:
                    clean_code = dev_text.strip()
                else:
                    # Prendre le bloc de code le plus long (généralement le Main, pas les petits extraits)
                    clean_code = max(matches, key=len).strip()
                    
                code_path = os.path.join(project_dir, request.filename)
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(clean_code)
                
            logger.info(f"✅ Pipeline BMAD terminée. Fichiers générés dans : {project_dir}")
            
            # 4. GitOps : Commit Automatique (via OpenClaw Skill)
            commit_msg = f"Factory: Generated '{project_name}' via BMAD (PRD + DAT + Code)"
            git_result = await git_commit(commit_msg)
            
            return {
                "status": "success",
                "filename": request.filename,
                "project_dir": project_dir,
                "git_status": git_result
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur Code Factory BMAD : {e}")
            return {"status": "error", "message": str(e)}
