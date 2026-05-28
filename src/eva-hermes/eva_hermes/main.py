"""Service Hermes pour orchestrer un petit conseil d'experts LLM."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpertDefinition:
    """Decrit un expert expose par Hermes."""

    name: str
    role: str
    backend: str
    host: str
    port: int
    model: str
    system_prompt: str
    description: str


class ChatRequest(BaseModel):
    """Requete de dialogue envoyee a un expert Hermes."""

    message: str = Field(..., min_length=1, max_length=12000)
    expert: str = Field(default="coordinator")
    system_prompt: str = Field(default="")
    context: dict[str, Any] = Field(default_factory=dict)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, ge=64, le=4000)


class MissionRequest(BaseModel):
    """Decrit une mission Hermes strictement consultative.

    Attributes:
        mission_id (str): Identifiant lisible de la mission.
        objective (str): Objectif business a instruire.
        constraints (list[str]): Contraintes non-negociables.
        context (dict[str, Any]): Contexte technique ou marche.
        symbols (list[str]): Sous-univers a prioriser.
        max_tokens (int): Budget de sortie par expert.
    """

    mission_id: str = Field(default="challenge_factory")
    objective: str = Field(..., min_length=12, max_length=4000)
    constraints: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    symbols: list[str] = Field(default_factory=list)
    max_tokens: int = Field(default=1400, ge=128, le=6000)


class ExpertRouter:
    """Resout les routes d'experts Hermes vers vLLM ou Ollama."""

    def __init__(self) -> None:
        """Construit les experts par defaut avec surcharge par variables d'environnement."""
        self.settings = get_settings()
        self._biblio_search_roots = self._build_biblio_search_roots()
        self._experts = {
            expert.name: expert
            for expert in (
                self._build_expert(
                    name="coordinator",
                    default_role="general",
                    default_backend="vllm",
                    description="Coordonne les autres experts et produit une synthese exploitable.",
                    system_prompt=(
                        "Tu es EVA Hermes, coordinatrice du conseil. "
                        "Tu synthétises vite, identifies les risques et proposes un plan d'action net."
                    ),
                ),
                self._build_expert(
                    name="technical",
                    default_role="research",
                    default_backend="vllm",
                    description="Analyse technique, structure de marche et niveaux clefs.",
                    system_prompt=(
                        "Tu es l'experte analyse technique de THE HIVE. "
                        "Tu lis la structure de marche, les regimes et les niveaux decisifs."
                    ),
                ),
                self._build_expert(
                    name="trading",
                    default_role="banker",
                    default_backend="vllm",
                    description="Execution trading, gestion du risque et validation des setups.",
                    system_prompt=(
                        "Tu es l'experte execution et risk management. "
                        "Tu privilegies les setups propres, la monetisation et la protection du capital."
                    ),
                ),
                self._build_expert(
                    name="macro_news",
                    default_role="research",
                    default_backend="vllm",
                    description="Lecture macro, newsflow et evenements susceptibles d'impacter le live.",
                    system_prompt=(
                        "Tu es l'experte macro-news. "
                        "Tu traduis rapidement les evenements et leur impact probable sur les actifs."
                    ),
                ),
                self._build_expert(
                    name="development",
                    default_role="code",
                    default_backend="ollama",
                    description="Copilote developpement pour patchs, diag et architecture.",
                    system_prompt=(
                        "Tu es l'experte developpement de THE HIVE. "
                        "Tu proposes des correctifs concrets, testables et coherents avec la stack."
                    ),
                ),
            )
        }

    def _build_expert(
        self,
        *,
        name: str,
        default_role: str,
        default_backend: str,
        description: str,
        system_prompt: str,
    ) -> ExpertDefinition:
        """Construit un expert a partir des defaults et des surcharges d'environnement.

        Args:
            name (str): Nom public de l'expert.
            default_role (str): Role conseil par defaut.
            default_backend (str): Backend cible par defaut.
            description (str): Description courte de l'expert.
            system_prompt (str): Prompt systeme de base.

        Returns:
            ExpertDefinition: Definition complete de routage.
        """
        env_prefix = f"HERMES_EXPERT_{name.upper()}"
        role = os.getenv(f"{env_prefix}_ROLE", default_role).strip().lower()
        backend = os.getenv(f"{env_prefix}_BACKEND", "").strip().lower() or self._resolve_backend(role, default_backend)
        model = os.getenv(f"{env_prefix}_MODEL", "").strip() or self._resolve_model(role, backend)
        host, port = self._resolve_endpoint(role, backend, env_prefix)
        final_prompt = os.getenv(f"{env_prefix}_SYSTEM_PROMPT", "").strip() or system_prompt
        final_prompt = self._compose_biblio_prompt(name, final_prompt)
        return ExpertDefinition(
            name=name,
            role=role,
            backend=backend,
            host=host,
            port=port,
            model=model,
            system_prompt=final_prompt,
            description=description,
        )

    def _resolve_backend(self, role: str, default_backend: str) -> str:
        """Resout le backend cible pour un role Hermes."""
        role_backend = os.getenv(f"COUNCIL_BACKEND_{role.upper()}", "").strip().lower()
        if role_backend in {"vllm", "ollama", "openrouter"}:
            return role_backend
        fallback = os.getenv("LLM_BACKEND", self.settings.llm_backend).strip().lower()
        if fallback in {"vllm", "ollama", "openrouter"}:
            return fallback if default_backend == "vllm" else default_backend
        return default_backend

    def _resolve_model(self, role: str, backend: str) -> str:
        """Resout le modele pour un role Hermes."""
        role_model = os.getenv(f"COUNCIL_MODEL_{role.upper()}", "").strip()
        if role_model:
            return role_model
        if backend == "ollama":
            return self.settings.ollama_model
        return self.settings.vllm_model

    def _resolve_endpoint(self, role: str, backend: str, env_prefix: str) -> tuple[str, int]:
        """Resout l'endpoint host/port pour un expert Hermes."""
        if backend == "openrouter":
            return "openrouter.ai", 443
        if backend == "ollama":
            default_host = self.settings.ollama_host
            default_port = self.settings.ollama_port
            host = os.getenv(
                f"{env_prefix}_HOST",
                os.getenv(f"COUNCIL_OLLAMA_HOST_{role.upper()}", os.getenv("COUNCIL_OLLAMA_HOST", default_host)),
            ).strip()
            port_raw = os.getenv(
                f"{env_prefix}_PORT",
                os.getenv(f"COUNCIL_OLLAMA_PORT_{role.upper()}", os.getenv("COUNCIL_OLLAMA_PORT", str(default_port))),
            ).strip()
        else:
            default_host = self.settings.vllm_host
            default_port = self.settings.vllm_port
            host = os.getenv(
                f"{env_prefix}_HOST",
                os.getenv(f"COUNCIL_VLLM_HOST_{role.upper()}", os.getenv("COUNCIL_VLLM_HOST", default_host)),
            ).strip()
            port_raw = os.getenv(
                f"{env_prefix}_PORT",
                os.getenv(f"COUNCIL_VLLM_PORT_{role.upper()}", os.getenv("COUNCIL_VLLM_PORT", str(default_port))),
            ).strip()
        try:
            port = int(port_raw)
        except ValueError:
            port = default_port
        return host, port

    def _build_biblio_search_roots(self) -> list[Path]:
        """Construit les repertoires de recherche de la bibliotheque de prompts."""
        roots: list[Path] = []

        def add(path_value: str | Path | None) -> None:
            candidate = Path(path_value or "").expanduser()
            if not str(candidate).strip():
                return
            resolved = candidate.resolve(strict=False)
            if resolved not in roots:
                roots.append(resolved)

        cwd = Path.cwd()
        add(self.settings.prompt_master_templates_dir)
        add(cwd / self.settings.prompt_master_templates_dir)
        add(cwd / "Biblio_IA")
        add(cwd.parent / "Biblio_IA")
        return roots

    def _compose_biblio_prompt(self, expert_name: str, base_prompt: str) -> str:
        """Enrichit un prompt systeme avec des extraits Biblio_IA si disponibles.

        Args:
            expert_name (str): Expert Hermes cible.
            base_prompt (str): Prompt systeme de base.

        Returns:
            str: Prompt enrichi si la bibliotheque est presente.
        """
        snippets = self._load_biblio_snippets(expert_name)
        if not snippets:
            return base_prompt
        return f"{base_prompt}\n\n### CADRE BIBLIO_IA\n{snippets}"

    def _load_biblio_snippets(self, expert_name: str) -> str:
        """Charge les extraits de prompts utiles pour un expert donne."""
        mapping = {
            "coordinator": [
                "Bibliothèque-Prompts/02-Business/Analyse.md",
                "Bibliothèque-Prompts/04-Métiers/Recherche-Veille.md",
            ],
            "technical": [
                "Bibliothèque-Prompts/02-Business/Analyse.md",
                "Bibliothèque-Prompts/04-Métiers/Recherche-Veille.md",
                "Bibliothèque-Prompts/00-Frameworks-Methodes/XML-Tags-Technique.md",
            ],
            "trading": [
                "Bibliothèque-Prompts/02-Business/Analyse.md",
                "Bibliothèque-Prompts/04-Métiers/Comptabilité-Finance.md",
                "Bibliothèque-Prompts/04-Métiers/Recherche-Veille.md",
            ],
            "macro_news": [
                "Bibliothèque-Prompts/02-Business/Analyse.md",
                "Bibliothèque-Prompts/04-Métiers/Recherche-Veille.md",
            ],
            "development": [
                "Bibliothèque-Prompts/01-Technique/Développement.md",
                "Bibliothèque-Prompts/01-Technique/Tech-Lead.md",
            ],
        }
        sections: list[str] = []
        for relative_path in mapping.get(expert_name, []):
            content = self._read_biblio_file(relative_path)
            if not content:
                continue
            title = Path(relative_path).stem
            sections.append(f"## {title}\n{content[:3500].strip()}")
        return "\n\n".join(section for section in sections if section.strip())

    def _read_biblio_file(self, relative_path: str) -> str:
        """Lit un fichier de la bibliotheque de prompts avec repli multi-racines."""
        for root in self._biblio_search_roots:
            candidate = (root / relative_path).resolve(strict=False)
            if candidate.exists() and candidate.is_file():
                try:
                    return candidate.read_text(encoding="utf-8")
                except Exception:
                    continue
        return ""

    def list_experts(self) -> list[ExpertDefinition]:
        """Retourne la liste des experts Hermes."""
        return list(self._experts.values())

    def get_expert(self, name: str) -> ExpertDefinition:
        """Retourne un expert, avec repli sur le coordinateur."""
        key = (name or "coordinator").strip().lower()
        return self._experts.get(key, self._experts["coordinator"])


class HermesService:
    """Client multi-experts Hermes vers vLLM/Ollama."""

    def __init__(self) -> None:
        """Initialise le routeur et le client HTTP."""
        self.router = ExpertRouter()
        self.client = httpx.AsyncClient(timeout=120.0)
        self._mission_output_dir = Path(
            os.getenv("HERMES_MISSION_OUTPUT_DIR", "data/hermes/missions")
        ).resolve(strict=False)
        self._mission_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Hermes initialise avec %s experts.", len(self.router.list_experts()))

    async def close(self) -> None:
        """Ferme les ressources reseau Hermes."""
        await self.client.aclose()

    async def ask(self, request: ChatRequest) -> dict[str, Any]:
        """Execute une requete sur l'expert cible.

        Args:
            request (ChatRequest): Requete utilisateur a router.

        Returns:
            dict[str, Any]: Reponse structuree d'Hermes.

        Raises:
            HTTPException: Si le backend cible ne repond pas correctement.
        """
        expert = self.router.get_expert(request.expert)
        system_prompt = expert.system_prompt
        if request.system_prompt.strip():
            system_prompt = f"{system_prompt}\n\n{request.system_prompt.strip()}"

        if request.context:
            context_lines = [f"- {key}: {value}" for key, value in request.context.items()]
            system_prompt = f"{system_prompt}\n\nContexte operatoire:\n" + "\n".join(context_lines)

        if expert.backend == "ollama":
            reply = await self._call_ollama(expert, request, system_prompt)
        elif expert.backend == "openrouter":
            reply = await self._call_openrouter(expert, request, system_prompt)
        else:
            reply = await self._call_vllm(expert, request, system_prompt)

        return {
            "status": "ok",
            "expert": asdict(expert),
            "message": reply,
        }

    async def run_mission(self, request: MissionRequest) -> dict[str, Any]:
        """Execute une mission Hermes consultative sans action destructive.

        Args:
            request (MissionRequest): Mission a instruire.

        Returns:
            dict[str, Any]: Rapport structure contenant les contributions.
        """
        mission_context = {
            "mode": "research_only",
            "promotion_automatique": False,
            "deployment_automatique": False,
            "execution_live": False,
            "symbols": request.symbols,
            **request.context,
        }
        experts_sequence = ("technical", "trading", "macro_news", "development")
        expert_outputs: dict[str, str] = {}

        for expert_name in experts_sequence:
            expert_request = ChatRequest(
                expert=expert_name,
                message=self._build_mission_prompt(request, expert_name),
                context=mission_context,
                temperature=0.15,
                max_tokens=request.max_tokens,
            )
            expert_reply = await self.ask(expert_request)
            expert_outputs[expert_name] = str(expert_reply.get("message") or "").strip()

        coordinator_request = ChatRequest(
            expert="coordinator",
            message=self._build_coordinator_prompt(request, expert_outputs),
            context=mission_context,
            temperature=0.1,
            max_tokens=request.max_tokens,
        )
        coordinator_reply = await self.ask(coordinator_request)
        report = {
            "status": "ok",
            "mission_id": request.mission_id,
            "objective": request.objective,
            "constraints": list(request.constraints),
            "symbols": list(request.symbols),
            "context": mission_context,
            "experts": expert_outputs,
            "synthesis": str(coordinator_reply.get("message") or "").strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "research_only",
            "promotion_automatique": False,
            "deployment_automatique": False,
        }
        self._write_mission_report(request.mission_id, report)
        return report

    def _build_mission_prompt(self, request: MissionRequest, expert_name: str) -> str:
        """Construit un prompt mission borne pour un expert donne."""
        constraints = "\n".join(f"- {item}" for item in request.constraints) or "- Aucune"
        symbols = ", ".join(request.symbols) if request.symbols else "Univers libre"
        role_guidance = {
            "technical": (
                "Propose des algorithmes et filtres de structure de marche adaptes "
                "aux comptes challenge FTMO/FTUK. Priorise les regimes, la seance, "
                "les entrees, les sorties et les conditions d'invalidation."
            ),
            "trading": (
                "Propose des mecaniques de monétisation et de gestion du risque "
                "permettant d'atteindre un challenge en moins de 7 jours sans violer "
                "les drawdowns. Reste concret sur split, runner, stop et exposition."
            ),
            "macro_news": (
                "Identifie les contraintes macro, news et spreads qui rendent certains "
                "algorithmes invalides pour FTMO/FTUK sur une fenetre de 7 jours."
            ),
            "development": (
                "Traduis les idees en changements implementables dans THE HIVE, avec "
                "ordre de priorite, fichiers cibles et risques techniques."
            ),
        }
        guidance = role_guidance.get(expert_name, "Analyse la mission avec rigueur.")
        return (
            f"Mission Hermes: {request.objective}\n\n"
            f"Role attendu pour {expert_name}: {guidance}\n\n"
            f"Contraintes:\n{constraints}\n\n"
            f"Symboles prioritaires: {symbols}\n\n"
            "Regles strictes:\n"
            "- Aucun deploiement automatique.\n"
            "- Aucune promotion live automatique.\n"
            "- Aucun ordre de trading reel.\n"
            "- Reponse en plan concret, hypotheses explicites, et tests proposes.\n"
        )

    def _build_coordinator_prompt(
        self,
        request: MissionRequest,
        expert_outputs: dict[str, str],
    ) -> str:
        """Construit le prompt de synthese finale du coordinateur."""
        sections = []
        for expert_name, content in expert_outputs.items():
            sections.append(f"### {expert_name}\n{content}")
        return (
            f"Objectif mission: {request.objective}\n\n"
            "Tu dois synthétiser les contributions ci-dessous en plan d'action "
            "priorisé pour faire émerger des algorithmes de trading capables de "
            "passer FTMO/FTUK en moins de 7 jours.\n\n"
            "Exigences de sortie:\n"
            "- 5 hypotheses maximum.\n"
            "- 5 patches prioritaires maximum.\n"
            "- 3 protocoles de test maximum.\n"
            "- une section finale 'Ne pas faire'.\n\n"
            + "\n\n".join(sections)
        )

    def _write_mission_report(self, mission_id: str, report: dict[str, Any]) -> None:
        """Persiste un rapport de mission Hermes sur disque."""
        safe_mission_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in mission_id.strip().lower()
        ) or "mission"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = self._mission_output_dir / f"{timestamp}_{safe_mission_id}.json"
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Rapport Hermes ecrit: %s", output_path)

    async def _call_vllm(self, expert: ExpertDefinition, request: ChatRequest, system_prompt: str) -> str:
        """Interroge un endpoint vLLM compatible OpenAI."""
        response = await self.client.post(
            f"http://{expert.host}:{expert.port}/v1/chat/completions",
            json={
                "model": expert.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.message},
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Backend vLLM indisponible pour l'expert {expert.name}: HTTP {response.status_code}",
            )
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"]).strip()

    async def _call_ollama(self, expert: ExpertDefinition, request: ChatRequest, system_prompt: str) -> str:
        """Interroge un endpoint Ollama."""
        prompt = f"System: {system_prompt}\nUser: {request.message}\nAssistant:"
        response = await self.client.post(
            f"http://{expert.host}:{expert.port}/api/generate",
            json={
                "model": expert.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            },
        )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Backend Ollama indisponible pour l'expert {expert.name}: HTTP {response.status_code}",
            )
        payload = response.json()
        return str(payload.get("response", "")).strip()

    async def _call_openrouter(self, expert: ExpertDefinition, request: ChatRequest, system_prompt: str) -> str:
        """Interroge l'API OpenRouter."""
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            logger.warning("OPENROUTER_API_KEY non configurée dans l'environnement, repli vers vLLM")
            return await self._call_vllm(expert, request, system_prompt)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://thehive.dev",
            "X-Title": "The Hive Hermes",
            "Content-Type": "application/json"
        }
        
        # Modèle par défaut si non spécifié (Llama 3.1 Nemotron 70B Free de Nvidia)
        model_name = expert.model or "nvidia/llama-3.1-nemotron-70b-instruct:free"
        
        response = await self.client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.message},
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"OpenRouter indisponible pour l'expert {expert.name}: HTTP {response.status_code} - {response.text}",
            )
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"]).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise Hermes et ferme le client HTTP a l'arret."""
    app.state.hermes = HermesService()
    logger.info("EVA Hermes demarre.")
    yield
    await app.state.hermes.close()
    logger.info("EVA Hermes arrete.")


app = FastAPI(
    title="EVA Hermes API",
    description="Conseil d'experts LLM connecte a vLLM et Ollama.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Retourne l'etat minimal d'Hermes."""
    hermes: HermesService = app.state.hermes
    return {
        "status": "online",
        "service": "hermes",
        "experts": [expert.name for expert in hermes.router.list_experts()],
    }


@app.get("/experts")
async def list_experts() -> dict[str, Any]:
    """Retourne la liste des experts exposes."""
    hermes: HermesService = app.state.hermes
    return {
        "experts": [asdict(expert) for expert in hermes.router.list_experts()],
        "count": len(hermes.router.list_experts()),
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    """Execute une requete utilisateur via l'expert cible."""
    hermes: HermesService = app.state.hermes
    return await hermes.ask(request)


@app.post("/missions/challenge")
async def run_challenge_mission(request: MissionRequest) -> dict[str, Any]:
    """Lance une mission challenge FTMO/FTUK strictement consultative."""
    hermes: HermesService = app.state.hermes
    return await hermes.run_mission(request)
