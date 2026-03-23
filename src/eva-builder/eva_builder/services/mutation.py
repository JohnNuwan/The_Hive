"""Service de mutation pour declencher l'evolution de `eva-builder`."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MutationService:
    """Declenche le pipeline d'evolution quand il est explicitement autorise."""

    def __init__(
        self,
        root_dir: str | None = None,
        python_executable: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Initialise les chemins et garde-fous du service.

        Args:
            root_dir (str | None): Racine du depot THE HIVE.
            python_executable (str | None): Interpreteur a utiliser.
            enabled (bool | None): Active l'execution reelle du runner.
        """
        candidate_root = Path(root_dir or "/app/the_hive")
        self.root_dir = candidate_root if candidate_root.exists() else Path.cwd()
        self.runner_path = self.root_dir / "scripts" / "evolution_runner.py"
        self.python_executable = python_executable or sys.executable
        if enabled is None:
            enabled = os.getenv("EVA_BUILDER_MUTATION_ENABLED", "false").strip().lower() == "true"
        self.enabled = enabled

    async def trigger_evolution(
        self,
        change_summary: str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Declenche le runner d'evolution si le service est active.

        Args:
            change_summary (str): Resume de la mutation a faire evoluer.
            dry_run (bool): Retourne la commande sans execution reelle.

        Returns:
            dict[str, Any]: Resultat d'execution ou diagnostic de blocage.
        """
        logger.info("MutationService: demande recue pour '%s'.", change_summary)

        if not self.runner_path.exists():
            error = f"Runner d'evolution introuvable: {self.runner_path}"
            logger.error(error)
            return {"status": "error", "message": error}

        command = [self.python_executable, str(self.runner_path)]
        if dry_run:
            return {
                "status": "dry_run",
                "command": command,
                "runner_path": str(self.runner_path),
                "summary": change_summary,
            }

        if not self.enabled:
            logger.warning("MutationService desactive par configuration.")
            return {
                "status": "disabled",
                "message": "Le pipeline de mutation est desactive par configuration.",
                "runner_path": str(self.runner_path),
                "summary": change_summary,
            }

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.root_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            if process.returncode == 0:
                logger.info("MutationService: evolution executee avec succes.")
                return {
                    "status": "success",
                    "summary": change_summary,
                    "output": stdout_text,
                    "error": stderr_text,
                    "returncode": process.returncode,
                }

            logger.error("MutationService: echec du runner d'evolution: %s", stderr_text)
            return {
                "status": "failed",
                "summary": change_summary,
                "output": stdout_text,
                "error": stderr_text,
                "returncode": process.returncode,
            }
        except Exception as exc:
            logger.error("MutationService: exception pendant l'evolution: %s", exc)
            return {"status": "error", "message": str(exc), "summary": change_summary}
