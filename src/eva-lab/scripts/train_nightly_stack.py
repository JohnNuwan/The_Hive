"""Orchestre la sequence nocturne complete des entrainements trading."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.nightly_training")

WORKDIR = Path(__file__).resolve().parents[1]
SUMMARY_PATH = WORKDIR / "data" / "checkpoints" / "nightly_training_summary.json"


def persist_summary(summary: dict[str, object]) -> None:
    """Ecrit le resume courant sur disque pour garder une trace meme en cas d'echec."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def append_step(
    summary: dict[str, object],
    name: str,
    status: str,
    error: str | None = None,
) -> None:
    """Ajoute le resultat d'une etape dans le resume JSON."""
    step: dict[str, object] = {"name": name, "status": status}
    if error:
        step["error"] = error
    summary.setdefault("steps", []).append(step)
    persist_summary(summary)


def run_step(name: str, command: list[str], extra_env: dict[str, str] | None = None) -> None:
    """Execute une etape d'entrainement dans un processus isole."""
    env = os.environ.copy()
    pythonpath_entries = [str(WORKDIR), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([entry for entry in pythonpath_entries if entry])
    if extra_env:
        env.update(extra_env)

    logger.info("Debut etape %s: %s", name, command)
    result = subprocess.run(command, cwd=WORKDIR, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Echec de l'etape {name} (code {result.returncode}).")
    logger.info("Etape %s terminee avec succes.", name)


def main() -> dict[str, object]:
    """Lance GNN, MuZero multi-horizon puis Dreamer offline."""
    summary: dict[str, object] = {
        "started_at": datetime.now().isoformat(),
        "workdir": str(WORKDIR),
        "steps": [],
        "status": "running",
    }
    persist_summary(summary)

    run_gnn = os.getenv("RUN_TRAIN_GNN", "1") == "1"
    run_muzero = os.getenv("RUN_TRAIN_MUZERO", "1") == "1"
    run_dreamer = os.getenv("RUN_TRAIN_DREAMER", "1") == "1"

    try:
        if run_gnn:
            run_step("gnn", [sys.executable, "scripts/train_gnn.py"])
            append_step(summary, "gnn", "ok")

        if run_muzero:
            horizons = [
                item.strip()
                for item in os.getenv("MUZERO_HORIZONS", "scalp,intraday,swing").split(",")
                if item.strip()
            ]
            for horizon in horizons:
                step_name = f"muzero_{horizon}"
                run_step(
                    step_name,
                    [sys.executable, "scripts/train_global_models.py"],
                    extra_env={"MUZERO_HORIZON": horizon},
                )
                append_step(summary, step_name, "ok")

        if run_dreamer:
            run_step(
                "dreamer_offline",
                [sys.executable, "-m", "eva_lab.muzero.offline_trainer"],
                extra_env={"DREAMER_EPOCHS": os.getenv("DREAMER_EPOCHS", "1500")},
            )
            append_step(summary, "dreamer_offline", "ok")

        summary["status"] = "ok"
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        logger.info("Resume nocturne ecrit dans %s", SUMMARY_PATH)
        return summary
    except Exception as exc:
        logger.exception("Sequence nocturne en echec: %s", exc)
        summary["status"] = "error"
        summary["error"] = str(exc)
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        raise


if __name__ == "__main__":
    report = main()
    logger.info("Sequence nocturne terminee: %s", report)
