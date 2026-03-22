#!/usr/bin/env python3
"""Ordonnanceur CPU leger pour les lanes data, research et memory."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import time
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]


def _ensure_local_import_paths() -> None:
    """Ajoute les chemins source necessaires au scheduler local."""

    for path in (
        ROOT_DIR / "scripts",
        ROOT_DIR / "src" / "shared",
        ROOT_DIR / "src" / "eva-lab",
    ):
        normalized = str(path)
        if normalized not in sys.path:
            sys.path.insert(0, normalized)


_ensure_local_import_paths()

from cpu_assist import (
    build_job_catalog,
    build_service_urls,
    _call_job,
    _fetch_training_snapshot,
    _select_jobs,
)

STATE_DIR = ROOT_DIR / "data" / "checkpoints" / "cpu_scheduler"
STATE_PATH = STATE_DIR / "state.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cpu_scheduler")


@dataclass(slots=True)
class LaneReport:
    """Represente le resultat d'une lane d'execution CPU."""

    lane: str
    started_at: str
    finished_at: str
    status: str
    host: str
    jobs: list[dict[str, Any]]
    summary: dict[str, Any]


LANE_JOB_NAMES = {
    "lane_research": [
        "researcher.health",
        "researcher.sync_sources",
        "researcher.ingest_status",
        "shadow.health",
        "shadow.alerts",
    ],
    "lane_memory": [
        "core.memory_fragments",
        "core.memory_graph",
        "core.autonomy_context",
        "core.intelligence_status",
    ],
}


def _utc_now_iso() -> str:
    """Retourne l'heure UTC courante au format ISO."""

    return datetime.now(timezone.utc).isoformat()


def _load_state() -> dict[str, Any]:
    """Charge l'etat persiste du scheduler."""

    if not STATE_PATH.exists():
        return {"lanes": {}, "updated_at": None}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state: dict[str, Any]) -> None:
    """Ecrit l'etat du scheduler sur disque."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def _record_cpu_job_history(payload: dict[str, Any]) -> None:
    """Persiste un job CPU dans TimeDB si la couche est disponible."""

    _ensure_local_import_paths()
    try:
        from eva_lab.timescale_store import record_cpu_job_history

        record_cpu_job_history(payload)
    except Exception as exc:
        logger.debug("Persistence CPU job ignoree: %s", exc)


def _resolve_history_dir() -> Path:
    """Retourne le dossier d'historiques cible pour le scheduler CPU."""

    configured = os.getenv("TRAINING_DATA_DIR")
    if configured:
        return Path(configured)
    return ROOT_DIR / "data" / "history"


def _resolve_active_families() -> list[str]:
    """Retourne les familles TimeDB actives a verifier par le scheduler."""

    raw_value = str(os.getenv("TRAINING_TIMESCALE_ACTIVE_FAMILIES", "metals")).strip()
    families = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    return families or ["metals"]


def _run_data_lane() -> list[dict[str, Any]]:
    """Execute les jobs internes de la lane data."""

    _ensure_local_import_paths()

    from eva_lab.timescale_store import ensure_timescale_ready
    from eva_lab.training_utils import build_dataset_coverage, get_model_family_symbols

    results: list[dict[str, Any]] = []
    history_dir = _resolve_history_dir()
    active_families = _resolve_active_families()
    schema_ok = ensure_timescale_ready()
    results.append(
        {
            "job_name": "timescaledb.ensure_schema",
            "status": "ok" if schema_ok else "degraded",
            "payload": {"schema_ready": schema_ok},
        }
    )
    results.append(
        {
            "job_name": "timescaledb.coverage_scope",
            "status": "ok",
            "payload": {
                "active_families": active_families,
                "history_dir_used": str(history_dir),
            },
        }
    )

    for family in active_families:
        coverage = build_dataset_coverage(
            symbols=get_model_family_symbols(family),
            timeframe="M5",
            data_dir=history_dir,
        )
        coverage_payload = dict(coverage)
        coverage_payload["history_dir_used"] = str(history_dir)
        coverage_payload["family"] = family
        timescale_status = dict(coverage.get("timescaledb") or {})
        effective_source = str(coverage.get("effective_source") or "")
        effective_ratio = float(coverage.get("coverage_ratio") or 0.0)
        coverage_ok = (
            schema_ok
            and effective_source == "timescaledb"
            and bool(timescale_status.get("ready"))
            and effective_ratio >= 1.0
        )
        results.append(
            {
                "job_name": f"timescaledb.coverage_{family}",
                "status": "ok" if coverage_ok else "degraded",
                "payload": coverage_payload,
            }
        )
    return results


def _run_http_lane(lane: str, host: str, timeout: float) -> list[dict[str, Any]]:
    """Execute une lane basee sur les jobs HTTP deja catalogues."""

    service_urls = build_service_urls(host)
    catalog = build_job_catalog()
    jobs = _select_jobs(catalog, groups=None, names=LANE_JOB_NAMES.get(lane, []), include_active=True)
    results: list[dict[str, Any]] = []
    for job in jobs:
        result = _call_job(job, service_urls, timeout)
        results.append(asdict(result))
    return results


def run_lane(lane: str, host: str, timeout: float) -> LaneReport:
    """Execute une lane et retourne un rapport structure."""

    started_at = _utc_now_iso()
    if lane == "lane_data":
        jobs = _run_data_lane()
    else:
        jobs = _run_http_lane(lane, host, timeout)
    finished_at = _utc_now_iso()
    statuses = [str(item.get("status") or "unknown") for item in jobs]
    status = "ok" if statuses and all(item == "ok" for item in statuses) else "degraded"
    summary = {
        "jobs_total": len(jobs),
        "jobs_ok": sum(1 for item in jobs if str(item.get("status")) == "ok"),
        "jobs_degraded": sum(1 for item in jobs if str(item.get("status")) == "degraded"),
        "jobs_unavailable": sum(1 for item in jobs if str(item.get("status")) == "unavailable"),
    }
    return LaneReport(
        lane=lane,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        host=host,
        jobs=jobs,
        summary=summary,
    )


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI du scheduler CPU."""

    parser = argparse.ArgumentParser(description="Ordonnanceur CPU pour les lanes data/research/memory.")
    parser.add_argument(
        "--lane",
        action="append",
        choices=["lane_data", "lane_research", "lane_memory"],
        help="Lane a executer. Peut etre repete.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Hote expose pour les services HTTP.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout HTTP par job.")
    parser.add_argument("--loop", action="store_true", help="Execute le scheduler en boucle continue.")
    parser.add_argument("--interval", type=float, default=300.0, help="Intervalle entre deux boucles, en secondes.")
    return parser.parse_args()


def _run_once(args: argparse.Namespace) -> int:
    """Execute un cycle complet du scheduler CPU."""

    lanes = args.lane or ["lane_data", "lane_research", "lane_memory"]
    service_urls = build_service_urls(args.host)
    training_before = _fetch_training_snapshot(service_urls, args.timeout)
    state = _load_state()
    state.setdefault("lanes", {})

    for lane in lanes:
        job_id = f"cpu-{uuid4().hex[:12]}"
        report = run_lane(lane, args.host, args.timeout)
        state["lanes"][lane] = asdict(report)
        state["updated_at"] = _utc_now_iso()
        _save_state(state)
        _record_cpu_job_history(
            {
                "job_id": job_id,
                "lane": lane,
                "job_name": lane,
                "status": report.status,
                "host": socket.gethostname(),
                "payload": asdict(report),
                "started_at": report.started_at,
                "finished_at": report.finished_at,
            }
        )
        logger.info(
            "Lane %s terminee: statut=%s | ok=%s/%s",
            lane,
            report.status,
            report.summary.get("jobs_ok", 0),
            report.summary.get("jobs_total", 0),
        )

    training_after = _fetch_training_snapshot(service_urls, args.timeout)
    if training_before and training_after:
        same_run = training_before.get("run_id") == training_after.get("run_id")
        trainer_running = training_after.get("trainer_state") == "running"
        if not same_run or not trainer_running:
            logger.error(
                "Le scheduler CPU a detecte une derive training: avant=%s apres=%s trainer=%s",
                training_before.get("run_id"),
                training_after.get("run_id"),
                training_after.get("trainer_state"),
            )
            return 2
    return 0


def main() -> int:
    """Point d'entree principal du scheduler CPU."""

    args = parse_args()
    if not args.loop:
        return _run_once(args)

    logger.info(
        "Demarrage du scheduler CPU en boucle: intervalle=%ss lanes=%s",
        args.interval,
        ",".join(args.lane or ["lane_data", "lane_research", "lane_memory"]),
    )
    while True:
        exit_code = _run_once(args)
        if exit_code != 0:
            return exit_code
        time.sleep(max(args.interval, 5.0))


if __name__ == "__main__":
    raise SystemExit(main())
