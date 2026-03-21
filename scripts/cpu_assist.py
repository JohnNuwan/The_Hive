#!/usr/bin/env python3
"""
Orchestre des charges CPU sures pour les usines non trading.

Cette CLI ne touche jamais aux artefacts critiques du training market.
Elle cible uniquement les services independants du run actif:
- connaissance (`researcher`, `core`);
- securite et observabilite (`shadow`, `sentinel`, `substrate`);
- usines business (`builder`, `muse`, `accountant`, `compliance`, `rwa`);
- consolidation lecture seule (`core`).

Par defaut, la commande fonctionne en dry-run pour afficher le plan
d'execution sans appeler les endpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cpu_assist")


DEFAULT_PORTS = {
    "core": 8080,
    "sentinel": 8200,
    "compliance": 8300,
    "substrate": 8400,
    "accountant": 8500,
    "lab": 8600,
    "rwa": 8700,
    "shadow": 8900,
    "builder": 9000,
    "muse": 9100,
    "researcher": 9300,
}

GROUP_ORDER = ("knowledge", "ops", "business", "core-readonly")


@dataclass(frozen=True, slots=True)
class JobDefinition:
    """Decrit un job CPU adressant un service non trading."""

    name: str
    group: str
    service: str
    method: str
    path: str
    description: str
    query: dict[str, Any] = field(default_factory=dict)
    body: dict[str, Any] | None = None
    active: bool = False
    optional: bool = False
    timeout_seconds: float | None = None


@dataclass(slots=True)
class JobResult:
    """Represente le resultat d'execution d'un job."""

    name: str
    group: str
    service: str
    method: str
    url: str
    active: bool
    optional: bool
    status: str
    status_code: int | None = None
    duration_seconds: float = 0.0
    detail: str = ""
    response_preview: str = ""


def _utc_now_iso() -> str:
    """Retourne la date UTC au format ISO 8601."""

    return datetime.now(timezone.utc).isoformat()


def build_service_urls(host: str) -> dict[str, str]:
    """Construit les URLs de base des services cibles."""

    urls: dict[str, str] = {}
    for service, port in DEFAULT_PORTS.items():
        env_key = f"CPU_ASSIST_URL_{service.upper().replace('-', '_')}"
        urls[service] = parse.urljoin(f"http://{host}:{port}", "/")
        if env_key in os.environ:
            urls[service] = os.environ[env_key].rstrip("/") + "/"
    return urls


def build_job_catalog() -> list[JobDefinition]:
    """Retourne le catalogue des charges CPU sures."""

    return [
        JobDefinition("researcher.health", "knowledge", "researcher", "GET", "/health", "Verifie la disponibilite du service de connaissance."),
        JobDefinition(
            "researcher.sync_sources",
            "knowledge",
            "researcher",
            "POST",
            "/ingest/sources/sync",
            "Declenche une synchronisation arXiv + actualites vers la file de revue.",
            body={"include_arxiv": True, "include_news": True, "max_items_per_source": 10, "trigger": "cpu_assist"},
            active=True,
            optional=True,
        ),
        JobDefinition("researcher.ingest_status", "knowledge", "researcher", "GET", "/ingest/status", "Recupere l'etat de la file de revue et des deduplications.", query={"tail": 20}, optional=True),
        JobDefinition("researcher.review_pending", "knowledge", "researcher", "GET", "/ingest/review", "Liste les candidats de connaissance en attente de revue.", query={"status": "pending", "limit": 25}, optional=True),
        JobDefinition("researcher.approved", "knowledge", "researcher", "GET", "/ingest/approved", "Liste les connaissances deja validees.", query={"limit": 25}, optional=True),
        JobDefinition("researcher.sources", "knowledge", "researcher", "GET", "/ingest/sources", "Recupere le statut des sources arXiv et actualites.", optional=True),
        JobDefinition("core.memory_fragments", "knowledge", "core", "GET", "/memory/fragments", "Lit les fragments memoire disponibles.", query={"limit": 25}),
        JobDefinition("core.memory_graph", "knowledge", "core", "GET", "/memory/graph", "Lit le graphe memoire consolide."),
        JobDefinition("shadow.health", "ops", "shadow", "GET", "/health", "Verifie la disponibilite du service OSINT."),
        JobDefinition("shadow.alerts", "ops", "shadow", "GET", "/alerts", "Recupere les alertes de veille.", query={"limit": 25}),
        JobDefinition("shadow.monitors", "ops", "shadow", "GET", "/monitor", "Liste les cibles de veille actives."),
        JobDefinition("shadow.personas", "ops", "shadow", "GET", "/profiles", "Liste les personas disponibles."),
        JobDefinition("shadow.threat_history", "ops", "shadow", "GET", "/threats/history", "Recupere l'historique de threat intel."),
        JobDefinition("sentinel.health", "ops", "sentinel", "GET", "/health", "Verifie la disponibilite du service de securite."),
        JobDefinition("sentinel.metrics", "ops", "sentinel", "GET", "/metrics", "Lit les metriques de securite et monitoring."),
        JobDefinition("sentinel.alerts", "ops", "sentinel", "GET", "/alerts", "Recupere les alertes de securite.", query={"limit": 25}),
        JobDefinition("sentinel.audit_logs", "ops", "sentinel", "GET", "/audit/logs", "Lit les logs d'audit recents.", query={"limit": 25}),
        JobDefinition("sentinel.compliance_check", "ops", "sentinel", "GET", "/compliance/check", "Lance une lecture de controle compliance."),
        JobDefinition("sentinel.security_scan", "ops", "sentinel", "POST", "/security/scan", "Declenche un scan de securite de routine.", active=True),
        JobDefinition("sentinel.integrity_check", "ops", "sentinel", "POST", "/integrity/check", "Declenche une verification d'integrite par defaut.", body={"files": []}, active=True),
        JobDefinition("substrate.health", "ops", "substrate", "GET", "/health", "Verifie la disponibilite du service energie."),
        JobDefinition("substrate.metrics", "ops", "substrate", "GET", "/metrics", "Recupere les metriques systeme et energie."),
        JobDefinition("substrate.mode", "ops", "substrate", "GET", "/mode", "Lit le mode circadien actif."),
        JobDefinition("substrate.alerts", "ops", "substrate", "GET", "/alerts", "Liste les alertes energie.", query={"limit": 25}),
        JobDefinition("substrate.thresholds", "ops", "substrate", "GET", "/thresholds", "Liste les seuils de supervision."),
        JobDefinition("substrate.metrics_history", "ops", "substrate", "GET", "/metrics/history", "Lit l'historique recent des metriques.", query={"limit": 60}),
        JobDefinition("builder.health", "business", "builder", "GET", "/health", "Verifie la disponibilite du Builder."),
        JobDefinition("builder.docgen", "business", "builder", "POST", "/maintenance/docgen", "Genere la documentation technique du Builder.", active=True),
        JobDefinition("builder.log_analysis", "business", "builder", "GET", "/maintenance/logs/analyze", "Analyse les erreurs recentes du Builder.", active=True),
        JobDefinition("builder.catalog_sync", "business", "builder", "POST", "/catalog/public-apis/sync", "Met a jour le catalogue des APIs publiques.", active=True),
        JobDefinition("builder.pipeline_status", "business", "builder", "GET", "/pipeline/status", "Recupere l'etat du pipeline Builder."),
        JobDefinition("builder.build_history", "business", "builder", "GET", "/build/history", "Lit l'historique de build du Builder."),
        JobDefinition("builder.deploy_history", "business", "builder", "GET", "/deploy/history", "Lit l'historique de deploiement du Builder."),
        JobDefinition("muse.health", "business", "muse", "GET", "/health", "Verifie la disponibilite de Muse."),
        JobDefinition("muse.stats", "business", "muse", "GET", "/stats", "Recupere les statistiques de Muse."),
        JobDefinition("muse.niches", "business", "muse", "GET", "/niches", "Liste les niches detectees par Muse."),
        JobDefinition("muse.niche_scores", "business", "muse", "GET", "/niches/scores", "Recupere les scores de niches de Muse.", optional=True, timeout_seconds=30.0),
        JobDefinition("muse.templates", "business", "muse", "GET", "/templates", "Liste les templates textuels disponibles."),
        JobDefinition("accountant.health", "business", "accountant", "GET", "/health", "Verifie la disponibilite d'Accountant."),
        JobDefinition("accountant.report", "business", "accountant", "GET", "/report", "Recupere le bilan financier consolide."),
        JobDefinition("accountant.dashboard", "business", "accountant", "GET", "/dashboard", "Recupere le tableau de bord financier."),
        JobDefinition("accountant.expenses", "business", "accountant", "GET", "/expenses", "Liste les depenses enregistrees.", query={"limit": 50}),
        JobDefinition("accountant.export", "business", "accountant", "GET", "/export", "Exporte les donnees comptables en JSON.", query={"format": "json"}),
        JobDefinition("compliance.health", "business", "compliance", "GET", "/health", "Verifie la disponibilite de Compliance."),
        JobDefinition("compliance.ledger", "business", "compliance", "GET", "/ledger", "Recupere le ledger fiscal."),
        JobDefinition("compliance.identity", "business", "compliance", "GET", "/identity", "Recupere l'identite juridique active."),
        JobDefinition("compliance.history", "business", "compliance", "GET", "/history", "Recupere l'historique des provisions.", query={"limit": 50}),
        JobDefinition("compliance.urssaf_report", "business", "compliance", "GET", "/report/urssaf", "Recupere le rapport URSSAF courant."),
        JobDefinition("compliance.alerts", "business", "compliance", "GET", "/alerts", "Recupere les alertes compliance.", query={"limit": 50}),
        JobDefinition("rwa.health", "business", "rwa", "GET", "/health", "Verifie la disponibilite du portefeuille RWA."),
        JobDefinition("rwa.portfolio", "business", "rwa", "GET", "/portfolio", "Recupere l'etat du portefeuille RWA."),
        JobDefinition("rwa.strategy", "business", "rwa", "GET", "/strategy", "Recupere la strategie souveraine active."),
        JobDefinition("rwa.recommendations", "business", "rwa", "GET", "/strategy/recommendations", "Recupere les recommandations d'investissement."),
        JobDefinition("rwa.telemetry", "business", "rwa", "GET", "/iot/telemetry", "Recupere la telemetrie IoT recente."),
        JobDefinition("rwa.energy_history", "business", "rwa", "GET", "/iot/energy/history", "Recupere l'historique energie sur 7 jours.", query={"days": 7}),
        JobDefinition("core.health", "core-readonly", "core", "GET", "/health", "Verifie la disponibilite de Core."),
        JobDefinition("core.agents_status", "core-readonly", "core", "GET", "/agents/status", "Recupere le statut agrege des agents."),
        JobDefinition("core.intelligence_status", "core-readonly", "core", "GET", "/intelligence/status", "Recupere l'etat de l'intelligence et des modes actifs."),
        JobDefinition("core.autonomy_context", "core-readonly", "core", "GET", "/intelligence/autonomy/context", "Recupere le contexte d'autonomie et les blocages."),
        JobDefinition("core.system_status", "core-readonly", "core", "GET", "/system/status", "Recupere l'etat systeme consolide."),
        JobDefinition("core.docker_containers", "core-readonly", "core", "GET", "/docker/containers", "Liste les conteneurs vus par Core."),
        JobDefinition("core.telemetry", "core-readonly", "core", "GET", "/telemetry", "Recupere la telemetrie legere du systeme."),
        JobDefinition("core.circuit_breaker", "core-readonly", "core", "GET", "/circuit-breaker/status", "Recupere le statut du coupe-circuit global."),
    ]


def _build_url(base_url: str, job: JobDefinition) -> str:
    """Construit l'URL finale d'un job."""

    url = base_url.rstrip("/") + job.path
    if job.query:
        url = f"{url}?{parse.urlencode(job.query, doseq=True)}"
    return url


def _preview_payload(payload: Any) -> str:
    """Construit un apercu compact d'une reponse JSON ou texte."""

    if payload is None:
        return ""
    try:
        if isinstance(payload, (dict, list)):
            text = json.dumps(payload, ensure_ascii=True)
        else:
            text = str(payload)
    except Exception:
        text = str(payload)
    text = " ".join(text.split())
    if len(text) <= 220:
        return text
    return text[:217].rstrip() + "..."


def _fetch_training_snapshot(service_urls: dict[str, str], timeout: float) -> dict[str, Any] | None:
    """Recupere un snapshot du training actif pour controle de non-regression."""

    url = service_urls["lab"].rstrip("/") + "/training/status"
    req = request.Request(url=url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Training status indisponible pour controle CPU: %s", exc)
        return None

    run = payload.get("run") or {}
    dependencies = payload.get("dependencies") or {}
    trainer = dependencies.get("trainer") or {}
    return {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "step_label": run.get("step_label"),
        "trainer_state": trainer.get("state"),
        "trainer_container": trainer.get("container"),
    }


def _call_job(job: JobDefinition, service_urls: dict[str, str], timeout: float) -> JobResult:
    """Execute un job HTTP et retourne un resultat structure."""

    base_url = service_urls[job.service]
    url = _build_url(base_url, job)
    timeout_seconds = job.timeout_seconds if job.timeout_seconds is not None else timeout
    headers = {"Accept": "application/json"}
    data: bytes | None = None
    if job.body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(job.body).encode("utf-8")

    req = request.Request(url=url, method=job.method, headers=headers, data=data)
    started_at = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            duration = time.perf_counter() - started_at
            try:
                payload = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                payload = raw
            return JobResult(
                name=job.name,
                group=job.group,
                service=job.service,
                method=job.method,
                url=url,
                active=job.active,
                optional=job.optional,
                status="ok",
                status_code=response.status,
                duration_seconds=duration,
                detail="Execution terminee.",
                response_preview=_preview_payload(payload),
            )
    except error.HTTPError as exc:
        duration = time.perf_counter() - started_at
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = str(exc)
        status = "http_error"
        detail = f"Erreur HTTP {exc.code}"
        if job.optional:
            if exc.code in {404, 405, 501}:
                status = "unavailable"
                detail = f"Endpoint optionnel indisponible (HTTP {exc.code})"
            elif exc.code in {408, 409, 423, 429, 500, 502, 503, 504}:
                status = "degraded"
                detail = f"Endpoint optionnel en mode degrade (HTTP {exc.code})"
        return JobResult(
            name=job.name,
            group=job.group,
            service=job.service,
            method=job.method,
            url=url,
            active=job.active,
            optional=job.optional,
            status=status,
            status_code=exc.code,
            duration_seconds=duration,
            detail=detail,
            response_preview=_preview_payload(body),
        )
    except Exception as exc:
        duration = time.perf_counter() - started_at
        status = "error"
        detail = str(exc)
        if job.optional:
            message = str(exc).lower()
            if isinstance(exc, (TimeoutError, socket.timeout)):
                status = "degraded"
                detail = "Timeout sur un endpoint optionnel."
            elif isinstance(exc, error.URLError):
                reason_text = str(exc.reason).lower() if getattr(exc, "reason", None) is not None else message
                if "timed out" in reason_text or "temps" in reason_text:
                    status = "degraded"
                    detail = "Timeout sur un endpoint optionnel."
                else:
                    status = "unavailable"
                    detail = "Endpoint optionnel indisponible."
            else:
                status = "degraded"
                detail = "Endpoint optionnel en erreur."
        return JobResult(
            name=job.name,
            group=job.group,
            service=job.service,
            method=job.method,
            url=url,
            active=job.active,
            optional=job.optional,
            status=status,
            duration_seconds=duration,
            detail=detail,
            response_preview=_preview_payload(exc),
        )


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    """Ecrit un rapport JSON hors des artefacts de training."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _select_jobs(catalog: list[JobDefinition], groups: list[str] | None, names: list[str] | None, include_active: bool) -> list[JobDefinition]:
    """Selectionne les jobs demandes par groupe ou par nom."""

    jobs = catalog
    if groups:
        wanted_groups = {group.strip() for group in groups if group.strip()}
        jobs = [job for job in jobs if job.group in wanted_groups]
    if names:
        wanted_names = {name.strip() for name in names if name.strip()}
        jobs = [job for job in jobs if job.name in wanted_names]
    if not include_active:
        jobs = [job for job in jobs if not job.active]
    return jobs


def _group_jobs(jobs: list[JobDefinition]) -> dict[str, list[JobDefinition]]:
    """Regroupe les jobs par groupe fonctionnel."""

    grouped: dict[str, list[JobDefinition]] = {group: [] for group in GROUP_ORDER}
    for job in jobs:
        grouped.setdefault(job.group, []).append(job)
    return {group: grouped[group] for group in GROUP_ORDER if grouped.get(group)}


def list_jobs(args: argparse.Namespace) -> int:
    """Affiche le catalogue des charges CPU disponibles."""

    catalog = build_job_catalog()
    jobs = _select_jobs(catalog, args.group, args.job, include_active=True)
    if not jobs:
        logger.error("Aucun job ne correspond aux filtres demandes.")
        return 1

    for group, items in _group_jobs(jobs).items():
        print(f"[{group}]")
        for job in items:
            nature = "actif" if job.active else "lecture"
            optional = "optionnel" if job.optional else "requis"
            print(f"- {job.name:<28} {job.method:<4} {job.service:<11} {nature:<7} {optional:<9} {job.description}")
        print()
    return 0


def run_jobs(args: argparse.Namespace) -> int:
    """Execute ou simule les jobs CPU selectionnes."""

    service_urls = build_service_urls(args.host)
    catalog = build_job_catalog()
    jobs = _select_jobs(catalog, args.group, args.job, include_active=args.include_active)
    if not jobs:
        logger.error("Aucun job selectionne. Utilise 'list' pour afficher le catalogue.")
        return 1

    training_before = _fetch_training_snapshot(service_urls, args.timeout)
    if training_before:
        logger.info(
            "Controle training avant run CPU: run_id=%s | trainer=%s | etape=%s",
            training_before.get("run_id"),
            training_before.get("trainer_state"),
            training_before.get("step_label"),
        )

    grouped = _group_jobs(jobs)
    logger.info(
        "Preparation du run CPU: %s jobs | execute=%s | include_active=%s | host=%s",
        len(jobs),
        args.execute,
        args.include_active,
        args.host,
    )

    if not args.execute:
        for group, items in grouped.items():
            logger.info("Dry-run groupe=%s | jobs=%s", group, len(items))
            for job in items:
                logger.info(
                    "  - %s | %s %s%s | actif=%s | optionnel=%s",
                    job.name,
                    job.method,
                    service_urls[job.service].rstrip("/"),
                    job.path,
                    job.active,
                    job.optional,
                )
        return 0

    max_workers = max(1, min(args.parallelism, len(jobs)))
    logger.info("Execution CPU avec parallelisme=%s", max_workers)
    results: list[JobResult] = []
    started_at = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_call_job, job, service_urls, args.timeout): job for job in jobs}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            logger.info(
                "%s | %s | code=%s | %.2fs | %s",
                result.name,
                result.status,
                result.status_code if result.status_code is not None else "-",
                result.duration_seconds,
                result.detail,
            )

    total_duration = time.perf_counter() - started_at
    training_after = _fetch_training_snapshot(service_urls, args.timeout)

    training_consistent = True
    training_detail = "Controle training indisponible."
    if training_before and training_after:
        same_run_id = training_before.get("run_id") == training_after.get("run_id")
        trainer_running = training_after.get("trainer_state") == "running"
        training_consistent = same_run_id and trainer_running
        training_detail = (
            f"avant={training_before.get('run_id')} | "
            f"apres={training_after.get('run_id')} | "
            f"trainer={training_after.get('trainer_state')}"
        )

    summary = {
        "generated_at": _utc_now_iso(),
        "host": args.host,
        "execute": args.execute,
        "include_active": args.include_active,
        "parallelism": max_workers,
        "total_jobs": len(results),
        "groups": sorted({result.group for result in results}),
        "duration_seconds": round(total_duration, 3),
        "training_before": training_before,
        "training_after": training_after,
        "training_consistent": training_consistent,
        "results": [asdict(result) for result in sorted(results, key=lambda item: (item.group, item.name))],
    }
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    summary["status_counts"] = status_counts

    if args.report:
        _write_report(Path(args.report), summary)
        logger.info("Rapport CPU ecrit dans %s", args.report)

    ok_count = sum(1 for result in results if result.status == "ok")
    tolerated_count = sum(1 for result in results if result.status in {"unavailable", "degraded"})
    blocking_failures = [result for result in results if result.status != "ok" and not result.optional]
    logger.info(
        "Run CPU termine: %s/%s jobs OK | toleres=%s | echecs bloquants=%s | duree=%.2fs | training=%s",
        ok_count,
        len(results),
        tolerated_count,
        len(blocking_failures),
        total_duration,
        training_detail,
    )

    if args.strict_training_check and training_before and training_after and not training_consistent:
        logger.error("Le controle training detecte une derive: %s", training_detail)
        return 2

    if blocking_failures:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI."""

    parser = argparse.ArgumentParser(description="Orchestre des charges CPU sures pour les usines non trading.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_list = subparsers.add_parser("list", help="Liste les jobs CPU disponibles.")
    parser_list.add_argument("--group", action="append", help="Filtre par groupe (knowledge, ops, business, core-readonly).")
    parser_list.add_argument("--job", action="append", help="Filtre par nom de job exact.")

    parser_run = subparsers.add_parser("run", help="Execute ou simule les jobs CPU.")
    parser_run.add_argument("--group", action="append", help="Selectionne un ou plusieurs groupes.")
    parser_run.add_argument("--job", action="append", help="Selectionne un ou plusieurs jobs exacts.")
    parser_run.add_argument("--host", default="127.0.0.1", help="Hote cible expose par docker compose.")
    parser_run.add_argument("--timeout", type=float, default=15.0, help="Timeout HTTP par job en secondes.")
    parser_run.add_argument("--parallelism", type=int, default=2, help="Nombre maximal de jobs lances en parallele.")
    parser_run.add_argument("--execute", action="store_true", help="Execute vraiment les jobs. Sans ce flag, la commande reste en dry-run.")
    parser_run.add_argument("--include-active", action="store_true", help="Inclut les jobs actifs (sync, scans, maintenance) en plus de la lecture seule.")
    parser_run.add_argument("--strict-training-check", action="store_true", help="Echoue si le run_id training change ou si le trainer n'est plus running.")
    parser_run.add_argument("--report", default="", help="Chemin optionnel d'un rapport JSON hors des artefacts trading.")

    return parser.parse_args()


def main() -> int:
    """Point d'entree principal de la CLI."""

    args = parse_args()
    if args.command == "list":
        return list_jobs(args)
    if args.command == "run":
        return run_jobs(args)
    logger.error("Commande inconnue: %s", args.command)
    return 1


if __name__ == "__main__":
    sys.exit(main())
