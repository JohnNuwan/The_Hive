"""Service de generation BMAD pour The Builder."""

import logging
import os
import re

import aiohttp
from pydantic import BaseModel

from eva_builder.services.bmad_prompts import ARCHITECT_PROMPT, DEVELOPER_PROMPT, PM_PROMPT

logger = logging.getLogger(__name__)

try:
    from openclaw.skills.git_ops import git_commit as openclaw_git_commit

    OPENCLAW_AVAILABLE = True
except Exception as exc:  # pragma: no cover - fallback runtime en conteneur
    openclaw_git_commit = None
    OPENCLAW_AVAILABLE = False
    OPENCLAW_IMPORT_ERROR = str(exc)


class CodeRequest(BaseModel):
    """Payload de demande de generation de code."""

    prompt: str
    filename: str
    language: str = "python"


class CodeFactoryService:
    """Genere un projet en trois phases BMAD (PM -> Architect -> Developer)."""

    def __init__(self) -> None:
        """Initialise les chemins et la configuration LLM du service."""
        self.output_dir = os.path.join(os.getcwd(), "data", "factory_output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.llm_backend = os.getenv("EVA_BUILDER_LLM_BACKEND", "vllm").strip().lower()
        if self.llm_backend == "vllm":
            self.llm_url = os.getenv(
                "EVA_BUILDER_LLM_URL",
                "http://host.docker.internal:8000/v1/chat/completions",
            )
        else:
            self.llm_url = os.getenv(
                "EVA_BUILDER_LLM_URL",
                "http://host.docker.internal:11434/api/generate",
            )

        self.model = os.getenv("EVA_BUILDER_LLM_MODEL", os.getenv("COUNCIL_MODEL_CODE", "Qwen/Qwen2.5-1.5B-Instruct"))

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        session: aiohttp.ClientSession,
    ) -> str:
        """Envoie une requete au backend LLM et retourne le texte genere."""
        if self.llm_backend == "vllm":
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
            }
            async with session.post(self.llm_url, json=payload, timeout=300) as response:
                if response.status != 200:
                    raise RuntimeError(f"Backend vLLM indisponible (status={response.status})")
                data = await response.json()
                choices = data.get("choices", [])
                if not choices:
                    return ""
                return choices[0].get("message", {}).get("content", "")

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {"model": self.model, "prompt": full_prompt, "stream": False}
        async with session.post(self.llm_url, json=payload, timeout=300) as response:
            if response.status != 200:
                raise RuntimeError(f"Backend Ollama indisponible (status={response.status})")
            data = await response.json()
            return data.get("response", "")

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extrait le plus gros bloc de code markdown, sinon retourne le texte brut."""
        matches = re.findall(r"```(?:[\\w+-]+)?\\s*\\n(.*?)```", text, re.DOTALL)
        if not matches:
            return text.strip()
        return max(matches, key=len).strip()

    async def _safe_git_commit(self, commit_msg: str) -> str:
        """Execute un commit via OpenClaw si le module est disponible."""
        if not OPENCLAW_AVAILABLE or openclaw_git_commit is None:
            logger.warning("OpenClaw indisponible: commit auto ignore.")
            if "OPENCLAW_IMPORT_ERROR" in globals():
                logger.warning("Cause import OpenClaw: %s", OPENCLAW_IMPORT_ERROR)
            return "OpenClaw indisponible: commit automatique ignore."

        try:
            return await openclaw_git_commit(commit_msg)
        except Exception as exc:
            logger.error("Echec du commit OpenClaw: %s", exc)
            return f"Echec commit OpenClaw: {exc}"

    async def generate_code(self, request: CodeRequest) -> dict:
        """Genere les artefacts PRD/DAT/code et tente un commit GitOps."""
        logger.info("Code Factory: demarrage du job '%s'.", request.filename)

        project_name = request.filename.rsplit(".", maxsplit=1)[0] if "." in request.filename else "project"
        project_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)

        try:
            async with aiohttp.ClientSession() as session:
                logger.info("BMAD phase 1: generation PRD.")
                prd_text = await self._call_llm(PM_PROMPT, f"Cahier des charges initial:\n{request.prompt}", session)
                prd_path = os.path.join(project_dir, "prd.md")
                with open(prd_path, "w", encoding="utf-8") as file:
                    file.write(prd_text)

                logger.info("BMAD phase 2: generation DAT.")
                dat_text = await self._call_llm(
                    ARCHITECT_PROMPT,
                    f"Voici le Product Requirement Document (PRD):\n\n{prd_text}",
                    session,
                )
                dat_path = os.path.join(project_dir, "architecture.md")
                with open(dat_path, "w", encoding="utf-8") as file:
                    file.write(dat_text)

                logger.info("BMAD phase 3: generation code.")
                developer_prompt = (
                    f"Voici le PRD:\n{prd_text}\n\n"
                    f"Voici le DAT:\n{dat_text}\n\n"
                    f"Tache: ecris le code source complet ({request.language}) "
                    f"correspondant a ces documents. Nom du fichier attendu: {request.filename}"
                )
                generated_text = await self._call_llm(DEVELOPER_PROMPT, developer_prompt, session)
                code_path = os.path.join(project_dir, request.filename)
                with open(code_path, "w", encoding="utf-8") as file:
                    file.write(self._extract_code(generated_text))

            logger.info("Pipeline BMAD terminee. Dossier genere: %s", project_dir)
            commit_msg = f"Factory: Generated '{project_name}' via BMAD (PRD + DAT + Code)"
            git_result = await self._safe_git_commit(commit_msg)

            return {
                "status": "success",
                "filename": request.filename,
                "project_dir": project_dir,
                "git_status": git_result,
            }
        except Exception as exc:
            logger.error("Erreur Code Factory BMAD: %s", exc)
            return {"status": "error", "message": str(exc)}

