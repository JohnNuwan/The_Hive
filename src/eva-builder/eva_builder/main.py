"""API de `eva-builder` pour generation, validation et maintenance de code."""

from __future__ import annotations

import asyncio
import ast
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared.redis_client import get_redis_client, init_redis

from eva_builder.cyber_forge import CyberForge
from eva_builder.services.api_catalog import PublicApiCatalogService
from eva_builder.services.deployment import DeploymentService
from eva_builder.services.factory import CodeFactoryService, CodeRequest
from eva_builder.services.librarian import LibrarianService
from eva_builder.services.mutation import MutationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefactorRequest(BaseModel):
    """Decrit un fichier a analyser par le Builder.

    Args:
        file_path (str): Chemin du fichier a analyser.
        analysis_type (str): Type d'analyse demande.
    """

    file_path: str = Field(..., description="Chemin du fichier a analyser")
    analysis_type: str = Field(
        default="quality",
        description="Type d'analyse: quality, complexity, security, performance",
    )


class DeployRequest(BaseModel):
    """Decrit une demande de deploiement.

    Args:
        service (str): Nom du service cible.
        target (str): Environnement cible.
        force_rebuild (bool): Active un rebuild force.
        dry_run (bool): Simule le deploiement sans execution reelle.
        compose_file (str | None): Fichier compose force si necessaire.
    """

    service: str = Field(..., description="Nom du service a deployer")
    target: str = Field(default="proxmox", description="Cible: proxmox ou local")
    force_rebuild: bool = False
    dry_run: bool = True
    compose_file: str | None = Field(
        default=None,
        description="Fichier docker compose force si besoin.",
    )


class ForgeRequest(BaseModel):
    """Decrit une execution securisee dans CyberForge.

    Args:
        script_name (str): Nom logique du script.
        code (str): Code source a executer.
        context (dict[str, Any]): Contexte d'execution injecte.
    """

    script_name: str = Field(..., description="Nom logique du script")
    code: str = Field(..., description="Code source a executer")
    context: dict[str, Any] = Field(default_factory=dict, description="Contexte simple injecte")


class MutationRequest(BaseModel):
    """Decrit une demande de mutation/evolution.

    Args:
        change_summary (str): Resume humain de la mutation.
        dry_run (bool): Simule la commande sans execution reelle.
    """

    change_summary: str = Field(..., description="Resume de la mutation a faire evoluer")
    dry_run: bool = Field(
        default=True,
        description="Simule la commande sans lancer le runner par defaut.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise les services du Builder et son heartbeat."""
    logger.info("Demarrage The Builder.")

    try:
        await init_redis()
        logger.info("Redis connecte pour The Builder.")
    except Exception as exc:  # pragma: no cover - depend de l'environnement
        logger.warning("Redis indisponible au demarrage du Builder: %s", exc)

    app.state.forge = CyberForge()
    app.state.api_catalog = PublicApiCatalogService()
    app.state.librarian = LibrarianService()
    app.state.mutation = MutationService()
    app.state.deployment = DeploymentService()
    app.state.factory = CodeFactoryService(
        forge=app.state.forge,
        api_catalog=app.state.api_catalog,
    )
    app.state.build_history: deque[dict[str, Any]] = deque(maxlen=200)
    app.state.deployment_log: deque[dict[str, Any]] = deque(maxlen=100)
    app.state.pipelines: list[dict[str, Any]] = []

    heartbeat_task = asyncio.create_task(hard_heartbeat(app))
    logger.info("The Builder est pret.")

    try:
        yield
    finally:
        heartbeat_task.cancel()
        logger.info("Arret The Builder.")


async def hard_heartbeat(app: FastAPI) -> None:
    """Publie periodiquement la presence du Builder dans Redis."""
    try:
        redis = get_redis_client()
    except Exception:
        redis = None

    while True:
        try:
            if redis:
                payload = {
                    "status": "online",
                    "ts": datetime.now().timestamp(),
                    "expert": "builder",
                    "active_pipelines": len(app.state.pipelines),
                }
                await redis.cache_set("eva.builder.status", payload, ttl_seconds=10)
        except Exception:  # pragma: no cover - heartbeat tolerant aux pannes
            logger.debug("Heartbeat Builder ignore suite a une erreur Redis.")
        await asyncio.sleep(2.0)


app = FastAPI(
    title="The Builder API",
    description="Agent DevOps, maintenance et usine logicielle de THE HIVE",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Systeme"])
async def health() -> dict[str, Any]:
    """Retourne l'etat global du service Builder."""
    return {
        "status": "ok",
        "service": "builder",
        "active_pipelines": len(app.state.pipelines),
        "builds_completed": len(app.state.build_history),
        "forge_runs": len(app.state.forge.get_forge_history()),
        "public_api_entries": app.state.api_catalog.get_stats()["total_entries"],
        "mutation_enabled": app.state.mutation.enabled,
        "deploy_enabled": app.state.deployment.enabled,
    }


@app.post("/maintenance/docgen", tags=["Maintenance"])
async def generate_docs() -> dict[str, Any]:
    """Regene la documentation minimale des modules du depot."""
    librarian: LibrarianService = app.state.librarian
    processed_count = await librarian.scan_and_generate()
    app.state.build_history.append(
        {
            "action": "docgen",
            "service": "librarian",
            "status": "success",
            "details": f"Fichiers documentes: {processed_count}",
            "timestamp": datetime.now().isoformat(),
        }
    )
    return {"status": "success", "files_processed": processed_count}


@app.get("/maintenance/logs/analyze", tags=["Maintenance"])
async def analyze_errors() -> dict[str, Any]:
    """Retourne un premier diagnostic d'etat du Builder."""
    return {
        "status": "info",
        "message": "Analyse automatique minimale disponible. Aucun connecteur de logs centralises n'est encore branche.",
        "last_check": datetime.now().isoformat(),
    }


@app.post("/catalog/public-apis/sync", tags=["Catalog"])
async def sync_public_api_catalog() -> dict[str, Any]:
    """Synchronise le catalogue d'APIs publiques utilise par Builder."""
    api_catalog: PublicApiCatalogService = app.state.api_catalog
    result = await api_catalog.sync_catalog()
    app.state.build_history.append(
        {
            "action": "catalog_sync",
            "service": "public_api_catalog",
            "status": result.get("status", "unknown"),
            "details": f"entrees={result.get('total_entries', 0)}",
            "timestamp": datetime.now().isoformat(),
        }
    )
    return result


@app.get("/catalog/public-apis/search", tags=["Catalog"])
async def search_public_api_catalog(
    query: str = "",
    category: str | None = None,
    auth: str | None = None,
    https_only: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    """Recherche des APIs publiques dans le catalogue local."""
    api_catalog: PublicApiCatalogService = app.state.api_catalog
    results = api_catalog.search_entries(
        query=query,
        category=category,
        auth=auth,
        https_only=https_only,
        limit=limit,
    )
    return {"results": results, "total": len(results)}


@app.post("/factory/build", tags=["Factory"])
async def build_software(request: CodeRequest) -> dict[str, Any]:
    """Genere un projet BMAD puis le valide si possible."""
    factory: CodeFactoryService = app.state.factory
    result = await factory.generate_code(request)
    app.state.build_history.append(
        {
            "action": "code_generation",
            "service": "factory",
            "status": result.get("status", "unknown"),
            "details": f"{request.filename} | langage={request.language}",
            "timestamp": datetime.now().isoformat(),
        }
    )
    return result


@app.post("/factory/forge", tags=["Factory"])
async def forge_code(request: ForgeRequest) -> dict[str, Any]:
    """Execute un script de maniere securisee dans CyberForge."""
    forge: CyberForge = app.state.forge
    result = forge.forge_and_test(
        script_name=request.script_name,
        code=request.code,
        context=request.context,
    )
    app.state.build_history.append(
        {
            "action": "forge_execution",
            "service": "cyber_forge",
            "status": "success" if result["success"] else "failed",
            "details": request.script_name,
            "timestamp": datetime.now().isoformat(),
        }
    )
    return result


@app.post("/mutation/trigger", tags=["Factory"])
async def trigger_mutation(request: MutationRequest) -> dict[str, Any]:
    """Declenche ou simule le pipeline de mutation du Builder."""
    mutation: MutationService = app.state.mutation
    result = await mutation.trigger_evolution(
        change_summary=request.change_summary,
        dry_run=request.dry_run,
    )
    app.state.build_history.append(
        {
            "action": "mutation_trigger",
            "service": "mutation",
            "status": result.get("status", "unknown"),
            "details": request.change_summary,
            "timestamp": datetime.now().isoformat(),
        }
    )
    return result


@app.get("/factory/forge/history", tags=["Factory"])
async def forge_history() -> dict[str, Any]:
    """Retourne l'historique des executions CyberForge."""
    forge: CyberForge = app.state.forge
    history = forge.get_forge_history()
    return {"history": history, "total": len(history)}


@app.post("/refactor/analyze", tags=["Qualite"])
async def analyze_code(request: RefactorRequest) -> dict[str, Any]:
    """Analyse la structure d'un fichier source et retourne des suggestions."""
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable: {request.file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lecture impossible du fichier: {exc}") from exc

    lines = content.splitlines()
    num_lines = len(lines)
    num_functions = sum(
        1
        for line in lines
        if line.strip().startswith("def ") or line.strip().startswith("async def ")
    )
    num_classes = sum(1 for line in lines if line.strip().startswith("class "))
    num_imports = sum(1 for line in lines if line.strip().startswith(("import ", "from ")))
    num_comments = sum(1 for line in lines if line.strip().startswith("#"))
    num_docstrings = content.count('"""') // 2

    syntax_ok = True
    syntax_error = None
    try:
        ast.parse(content)
    except SyntaxError as exc:
        syntax_ok = False
        syntax_error = str(exc)

    comment_ratio = (num_comments + num_docstrings) / max(num_lines, 1)
    quality_score = min(
        10.0,
        round(comment_ratio * 30 + (2 if num_lines < 500 else 0) + (3 if num_docstrings > 0 else 0), 1),
    )

    suggestions: list[str] = []
    if not syntax_ok:
        suggestions.append("Corriger d'abord l'erreur de syntaxe detectee.")
    if num_lines > 500:
        suggestions.append("Decouper le fichier en modules plus petits.")
    if comment_ratio < 0.05:
        suggestions.append("Ajouter plus de docstrings ou de commentaires metier.")
    if num_functions > 20:
        suggestions.append("Regrouper les comportements dans des services ou classes dedies.")

    return {
        "file": str(file_path),
        "analysis_type": request.analysis_type,
        "metrics": {
            "total_lines": num_lines,
            "functions": num_functions,
            "classes": num_classes,
            "imports": num_imports,
            "comments": num_comments,
            "docstrings": num_docstrings,
            "syntax_ok": syntax_ok,
            "syntax_error": syntax_error,
        },
        "quality_score": quality_score,
        "suggestions": suggestions,
    }


@app.get("/pipeline/status", tags=["CI/CD"])
async def pipeline_status() -> dict[str, Any]:
    """Retourne l'etat courant des pipelines suivis par Builder."""
    pipelines = app.state.pipelines
    return {
        "pipelines": pipelines,
        "total": len(pipelines),
        "running": sum(1 for pipeline in pipelines if pipeline.get("status") == "running"),
    }


@app.post("/deploy", tags=["CI/CD"])
async def trigger_deploy(request: DeployRequest) -> dict[str, Any]:
    """Prepare ou execute un deploiement pilote par Builder."""
    deployment: DeploymentService = app.state.deployment
    result = await deployment.deploy(
        service=request.service,
        target=request.target,
        force_rebuild=request.force_rebuild,
        dry_run=request.dry_run,
        compose_file=request.compose_file,
    )
    deploy_entry = {
        "id": f"DEP-{uuid4().hex[:8].upper()}",
        "service": request.service,
        "target": request.target,
        "force_rebuild": request.force_rebuild,
        "dry_run": request.dry_run,
        "compose_file": request.compose_file,
        "status": result.get("status", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "deployment": result.get("deployment"),
    }
    app.state.deployment_log.append(deploy_entry)
    logger.info(
        "Deploiement traite: service=%s cible=%s statut=%s",
        request.service,
        request.target,
        result.get("status", "unknown"),
    )
    return {"status": result.get("status", "unknown"), "deployment": deploy_entry, "result": result}


@app.get("/deploy/history", tags=["CI/CD"])
async def deployment_history() -> dict[str, Any]:
    """Retourne l'historique des demandes de deploiement."""
    deployments = list(app.state.deployment_log)
    return {"deployments": deployments, "total": len(deployments)}


@app.get("/build/history", tags=["Factory"])
async def build_history() -> dict[str, Any]:
    """Retourne l'historique des actions realisees par Builder."""
    history = list(app.state.build_history)
    return {"history": history, "total": len(history)}

