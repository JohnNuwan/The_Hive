"""Service BMAD de generation, validation et tracabilite pour `eva-builder`."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import BaseModel, Field

from eva_builder.cyber_forge import CyberForge
from eva_builder.services.api_catalog import PublicApiCatalogService
from eva_builder.services.bmad_prompts import ARCHITECT_PROMPT, DEVELOPER_PROMPT, PM_PROMPT

logger = logging.getLogger(__name__)

try:
    from openclaw.skills.git_ops import git_commit as openclaw_git_commit

    OPENCLAW_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depend du runtime conteneur
    openclaw_git_commit = None
    OPENCLAW_AVAILABLE = False
    OPENCLAW_IMPORT_ERROR = str(exc)


class CodeRequest(BaseModel):
    """Decrit une demande de generation logicielle.

    Args:
        prompt (str): Besoin fonctionnel a transformer en projet.
        filename (str): Nom du fichier principal a generer.
        language (str): Langage cible du fichier principal.
        auto_validate (bool): Active la validation automatique du code genere.
        use_public_api_catalog (bool): Active l'injection d'APIs publiques
            pertinentes si un catalogue local est disponible.
        api_context_query (str | None): Requete optionnelle pour guider la
            recherche d'APIs a partir d'un besoin plus precis.
        api_context_limit (int): Nombre maximum d'APIs proposees au LLM.
    """

    prompt: str
    filename: str
    language: str = "python"
    auto_validate: bool = Field(
        default=True,
        description="Active la validation automatique quand le langage est supporte.",
    )
    use_public_api_catalog: bool = Field(
        default=True,
        description="Injecte des APIs publiques pertinentes si le catalogue local est disponible.",
    )
    api_context_query: str | None = Field(
        default=None,
        description="Requete specifique pour chercher des APIs utiles au projet.",
    )
    api_context_limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Nombre maximum d'APIs suggerees dans le contexte de generation.",
    )


class CodeFactoryService:
    """Genere un projet BMAD puis valide le resultat si possible."""

    def __init__(
        self,
        forge: CyberForge | None = None,
        api_catalog: PublicApiCatalogService | None = None,
    ) -> None:
        """Initialise les chemins, le LLM et le validateur du service.

        Args:
            forge (CyberForge | None): Instance partagee de CyberForge si
                l'appelant souhaite mutualiser l'historique des validations.
            api_catalog (PublicApiCatalogService | None): Catalogue local
                d'APIs publiques pour enrichir les briefs produit.
        """
        self.output_dir = Path(os.getcwd()) / "data" / "factory_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.forge = forge or CyberForge()
        self.api_catalog = api_catalog or PublicApiCatalogService()

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

        self.model = os.getenv(
            "EVA_BUILDER_LLM_MODEL",
            os.getenv("COUNCIL_MODEL_CODE", "Qwen/Qwen2.5-1.5B-Instruct"),
        )

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        session: aiohttp.ClientSession,
    ) -> str:
        """Interroge le backend LLM configure et retourne le texte genere."""
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
                    raise RuntimeError(
                        f"Backend vLLM indisponible (status={response.status}).",
                    )
                data = await response.json()
                choices = data.get("choices", [])
                if not choices:
                    return ""
                return choices[0].get("message", {}).get("content", "")

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {"model": self.model, "prompt": full_prompt, "stream": False}
        async with session.post(self.llm_url, json=payload, timeout=300) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Backend Ollama indisponible (status={response.status}).",
                )
            data = await response.json()
            return data.get("response", "")

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extrait le plus grand bloc de code Markdown ou retourne le texte brut."""
        matches = re.findall(r"```(?:[\w+-]+)?\s*\n(.*?)```", text, re.DOTALL)
        if not matches:
            return text.strip()
        return max(matches, key=len).strip()

    @staticmethod
    def _resolve_project_name(request: CodeRequest) -> str:
        """Construit un nom de projet a partir du fichier principal demande."""
        stem = Path(request.filename).stem.strip()
        return stem or "project"

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        """Ecrit un fichier texte UTF-8 en creant son dossier parent si besoin."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    async def _safe_git_commit(self, commit_message: str) -> str:
        """Tente un commit GitOps via OpenClaw si disponible."""
        if not OPENCLAW_AVAILABLE or openclaw_git_commit is None:
            logger.warning("OpenClaw indisponible: commit automatique ignore.")
            if "OPENCLAW_IMPORT_ERROR" in globals():
                logger.warning("Cause import OpenClaw: %s", OPENCLAW_IMPORT_ERROR)
            return "OpenClaw indisponible: commit automatique ignore."

        try:
            return await openclaw_git_commit(commit_message)
        except Exception as exc:
            logger.error("Echec du commit OpenClaw: %s", exc)
            return f"Echec commit OpenClaw: {exc}"

    def _build_validation_result(self, request: CodeRequest, code: str) -> dict[str, Any] | None:
        """Valide le code genere si le langage est supporte par CyberForge."""
        if not request.auto_validate:
            return None
        if request.language.strip().lower() != "python":
            return {
                "executed": False,
                "reason": f"Validation auto non supportee pour le langage '{request.language}'.",
            }
        return self.forge.forge_and_test(
            script_name=request.filename,
            code=code,
            context={},
        )

    def _write_report(
        self,
        project_dir: Path,
        request: CodeRequest,
        paths: dict[str, str],
        validation_result: dict[str, Any] | None,
        git_status: str,
        api_suggestions: list[dict[str, Any]],
    ) -> str:
        """Ecrit un rapport JSON de generation dans le dossier du projet."""
        report_path = project_dir / "build_report.json"
        status = "success"
        if validation_result and validation_result.get("executed") is False:
            status = "success"
        elif validation_result and not validation_result.get("success", False):
            status = "warning"

        payload = {
            "status": status,
            "project_dir": str(project_dir),
            "filename": request.filename,
            "language": request.language,
            "files": paths,
            "validation": validation_result,
            "git_status": git_status,
            "api_suggestions": api_suggestions,
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return str(report_path)

    @staticmethod
    def _format_api_context(api_suggestions: list[dict[str, Any]]) -> str:
        """Transforme les suggestions d'APIs en contexte lisible pour le LLM.

        Args:
            api_suggestions (list[dict[str, Any]]): APIs selectionnees.

        Returns:
            str: Bloc texte a injecter dans les prompts.
        """
        if not api_suggestions:
            return ""

        lines = [
            "APIs publiques deja identifiees comme potentiellement utiles:",
        ]
        for entry in api_suggestions:
            lines.append(
                (
                    f"- {entry['name']} | categorie={entry['category']} | auth={entry['auth']} "
                    f"| https={'Yes' if entry['https'] else 'No'} | cors={entry['cors']} "
                    f"| url={entry['url']} | description={entry['description']}"
                )
            )
        lines.append(
            "Consigne: privilegie ces APIs si elles renforcent le MVP, sinon explique pourquoi elles sont ecartees.",
        )
        return "\n".join(lines)

    async def generate_code(self, request: CodeRequest) -> dict[str, Any]:
        """Genere les artefacts BMAD, valide le code et produit un rapport.

        Args:
            request (CodeRequest): Demande de generation.

        Returns:
            dict[str, Any]: Resultat de generation avec chemins et validation.
        """
        logger.info("Code Factory: demarrage du job '%s'.", request.filename)

        project_name = self._resolve_project_name(request)
        project_dir = self.output_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        api_query = (request.api_context_query or request.prompt).strip()
        api_suggestions: list[dict[str, Any]] = []
        if request.use_public_api_catalog and api_query:
            api_suggestions = self.api_catalog.recommend_for_prompt(
                prompt=api_query,
                limit=request.api_context_limit,
            )
            if api_suggestions:
                logger.info(
                    "Code Factory: %s APIs publiques suggerees pour '%s'.",
                    len(api_suggestions),
                    request.filename,
                )
        api_context = self._format_api_context(api_suggestions)

        try:
            async with aiohttp.ClientSession() as session:
                logger.info("BMAD phase 1: generation du PRD.")
                product_prompt = f"Cahier des charges initial:\n{request.prompt}"
                if api_context:
                    product_prompt = f"{product_prompt}\n\n{api_context}"
                prd_text = await self._call_llm(
                    PM_PROMPT,
                    product_prompt,
                    session,
                )
                if not prd_text.strip():
                    raise RuntimeError("Le backend LLM a retourne un PRD vide.")
                prd_path = project_dir / "prd.md"
                self._write_text(prd_path, prd_text)

                logger.info("BMAD phase 2: generation du DAT.")
                architect_prompt = f"Voici le Product Requirement Document (PRD):\n\n{prd_text}"
                if api_context:
                    architect_prompt = f"{architect_prompt}\n\n{api_context}"
                dat_text = await self._call_llm(
                    ARCHITECT_PROMPT,
                    architect_prompt,
                    session,
                )
                if not dat_text.strip():
                    raise RuntimeError("Le backend LLM a retourne un DAT vide.")
                dat_path = project_dir / "architecture.md"
                self._write_text(dat_path, dat_text)

                logger.info("BMAD phase 3: generation du code source.")
                developer_prompt = (
                    f"Voici le PRD:\n{prd_text}\n\n"
                    f"Voici le DAT:\n{dat_text}\n\n"
                    f"{api_context}\n\n"
                    f"Tache: ecris le code source complet ({request.language}) "
                    f"correspondant a ces documents. Nom du fichier attendu: {request.filename}"
                )
                generated_text = await self._call_llm(DEVELOPER_PROMPT, developer_prompt, session)
                extracted_code = self._extract_code(generated_text)
                if not extracted_code.strip():
                    raise RuntimeError("Le backend LLM a retourne un code vide.")
                code_path = project_dir / request.filename
                self._write_text(code_path, extracted_code)

            validation_result = self._build_validation_result(request, extracted_code)
            commit_message = f"Factory: generation '{project_name}' via BMAD"
            git_status = await self._safe_git_commit(commit_message)

            files = {
                "prd": str(prd_path),
                "architecture": str(dat_path),
                "code": str(code_path),
            }
            report_path = self._write_report(
                project_dir=project_dir,
                request=request,
                paths=files,
                validation_result=validation_result,
                git_status=git_status,
                api_suggestions=api_suggestions,
            )
            files["report"] = report_path

            status = "success"
            if validation_result and validation_result.get("executed") is False:
                status = "success"
            elif validation_result and not validation_result.get("success", False):
                status = "warning"

            logger.info("Pipeline BMAD terminee. Dossier genere: %s", project_dir)
            return {
                "status": status,
                "filename": request.filename,
                "project_dir": str(project_dir),
                "files": files,
                "validation": validation_result,
                "git_status": git_status,
                "api_suggestions": api_suggestions,
            }
        except Exception as exc:
            logger.error("Erreur Code Factory BMAD: %s", exc)
            return {"status": "error", "message": str(exc)}

