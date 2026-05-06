"""Force un champion live manuel et verrouille son manifeste."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.training_status import load_terminal_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.force_live_champion")


def _read_flag(name: str, default: bool = True) -> bool:
    """Interprete une variable d'environnement booleenne."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _load_optional_summary(summary_path: str | None) -> dict[str, Any]:
    """Charge un resume optionnel si le chemin est fourni."""

    if not str(summary_path or "").strip():
        return {}
    payload = load_terminal_summary(path=summary_path)
    return dict(payload or {})


def _build_manual_promotion_gate(summary: dict[str, Any], reason: str) -> dict[str, Any]:
    """Construit un verdict de gate manuel a partir d'un resume existant."""

    existing_gate = dict(summary.get("promotion_gate") or {})
    metrics = dict(existing_gate.get("metrics") or summary.get("metrics") or {})
    thresholds = dict(existing_gate.get("thresholds") or {})
    return {
        "allowed": True,
        "status": "promoted",
        "reason": reason,
        "gate_profile": str(existing_gate.get("gate_profile") or "manual_override"),
        "failure_mode": "manual_override",
        "checks": {"manual_override": True},
        "thresholds": thresholds,
        "metrics": metrics,
    }


def force_live_champion() -> dict[str, Any]:
    """Force un checkpoint donne comme champion live verrouille."""

    promoter = ChampionPromoter()
    engine = promoter.normalize_engine_name(os.getenv("HIVE_FORCE_LIVE_ENGINE", "muzero"))
    horizon = str(os.getenv("HIVE_FORCE_LIVE_HORIZON", "scalp") or "scalp").strip().lower()
    source_path = Path(
        str(os.getenv("HIVE_FORCE_LIVE_CHAMPION_PATH", "")).strip()
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint champion introuvable: {source_path}")

    challenger_id = (
        str(os.getenv("HIVE_FORCE_LIVE_CHAMPION_ID", "")).strip()
        or source_path.stem
    )
    summary_path = str(os.getenv("HIVE_FORCE_LIVE_SUMMARY_PATH", "")).strip() or None
    reason = str(
        os.getenv("HIVE_FORCE_LIVE_REASON", "manual_override_forced_live")
    ).strip() or "manual_override_forced_live"
    manual_live_lock = _read_flag("HIVE_FORCE_LIVE_LOCK", True)

    summary = _load_optional_summary(summary_path)
    compatibility = promoter.inspect_checkpoint_compatibility(
        source_path,
        horizon=horizon,
        engine=engine,
    )
    if not compatibility.get("allowed", False):
        raise RuntimeError(
            f"Checkpoint incompatible pour le live: {compatibility.get('reason')}"
        )

    champion_path = promoter.get_champion_path(horizon, engine=engine)
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, champion_path)

    promotion_gate = _build_manual_promotion_gate(summary, reason)
    battle_report = dict(summary.get("battle_report") or {})
    training_metrics = dict(summary.get("training_metrics") or summary.get("metrics") or {})
    promotion_result = {
        "status": "promoted",
        "reason": reason,
        "engine": engine,
        "horizon": horizon,
        "source_path": str(source_path),
        "champion_paths": [str(champion_path)],
        "promotion_gate": promotion_gate,
        "artifact_compatibility": compatibility,
        "checkpoint_schema_version": compatibility.get("schema_version"),
        "resume_source": "manual_override",
        "lineage": dict(summary.get("lineage") or {}),
        "live_comparison": dict(summary.get("live_comparison") or {}),
    }
    manifest = promoter.persist_challenger_manifest(
        engine=engine,
        horizon=horizon,
        status="promoted",
        challenger_id=challenger_id,
        challenger_path=str(source_path),
        latest_checkpoint=str(source_path),
        battle_report=battle_report or None,
        training_metrics=training_metrics,
        promotion_gate=promotion_gate,
        promotion_result=promotion_result,
        artifact_compatibility=compatibility,
        checkpoint_schema_version=compatibility.get("schema_version"),
        resume_source="manual_override",
        lineage=dict(summary.get("lineage") or {}),
        promotion_metadata={
            "manual_live_lock": manual_live_lock,
            "manual_override_reason": reason,
            "manual_override_source_checkpoint": str(source_path),
            "manual_override_source_summary": summary_path,
            "manual_override_summary": {
                "checkpoint": str(source_path),
                "challenger_id": challenger_id,
                "reason": reason,
            },
        },
    )
    logger.info(
        "Champion live force pour %s/%s: %s -> %s",
        engine,
        horizon,
        challenger_id,
        champion_path,
    )
    return {
        "engine": engine,
        "horizon": horizon,
        "challenger_id": challenger_id,
        "source_path": str(source_path),
        "champion_path": str(champion_path),
        "manifest_path": str(promoter.get_manifest_path(horizon, engine=engine)),
        "manual_live_lock": manual_live_lock,
        "promotion_gate": promotion_gate,
        "manifest_status": manifest.get("status"),
    }


def main() -> dict[str, Any]:
    """Execute le forçage et affiche un resume JSON."""

    result = force_live_champion()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
