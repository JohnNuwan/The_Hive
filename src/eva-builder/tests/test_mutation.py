"""Tests du service de mutation pour `eva-builder`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from eva_builder.services.mutation import MutationService


def test_trigger_evolution_retourne_dry_run(tmp_path: Path) -> None:
    """Verifie qu'un dry-run retourne la commande sans execution reelle."""
    runner = tmp_path / "scripts" / "evolution_runner.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("print('ok')", encoding="utf-8")

    service = MutationService(root_dir=str(tmp_path), python_executable="python-test", enabled=True)
    result = asyncio.run(service.trigger_evolution("ajout d'un garde-fou", dry_run=True))

    assert result["status"] == "dry_run"
    assert result["command"] == ["python-test", str(runner)]


def test_trigger_evolution_refuse_si_desactive(tmp_path: Path) -> None:
    """Verifie que l'execution reelle reste bloquee tant que le service est desactive."""
    runner = tmp_path / "scripts" / "evolution_runner.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("print('ok')", encoding="utf-8")

    service = MutationService(root_dir=str(tmp_path), enabled=False)
    result = asyncio.run(service.trigger_evolution("mutation critique", dry_run=False))

    assert result["status"] == "disabled"
    assert "desactive" in result["message"]


def test_trigger_evolution_execute_le_runner_si_active(tmp_path: Path, monkeypatch) -> None:
    """Verifie qu'un runner actif remonte correctement stdout, stderr et code retour."""
    runner = tmp_path / "scripts" / "evolution_runner.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("print('ok')", encoding="utf-8")

    class FakeProcess:
        """Simule un processus asyncio pour le test."""

        def __init__(self) -> None:
            self.returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"mutation ok", b"")

    async def fake_create_subprocess_exec(*command, **kwargs):
        """Simule le lancement du runner sans sous-processus reel."""
        del kwargs
        assert command[1] == str(runner)
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    service = MutationService(root_dir=str(tmp_path), python_executable="python-test", enabled=True)
    result = asyncio.run(service.trigger_evolution("mutation critique", dry_run=False))

    assert result["status"] == "success"
    assert result["output"] == "mutation ok"
    assert result["returncode"] == 0
