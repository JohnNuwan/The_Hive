"""Tests du service de deploiement pour `eva-builder`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from eva_builder.services.deployment import DeploymentService


def test_deploy_dry_run_retourne_un_plan_swarm(tmp_path: Path) -> None:
    """Verifie qu'un dry-run swarm retourne le plan Docker attendu."""
    service = DeploymentService(root_dir=str(tmp_path), enabled=True)

    result = asyncio.run(
        service.deploy(
            service="swarm",
            target="proxmox",
            force_rebuild=True,
            dry_run=True,
        )
    )

    assert result["status"] == "dry_run"
    commands = result["deployment"]["commands"]
    assert commands[0] == "docker compose -f docker-compose.yml build --no-cache"
    assert commands[1] == "docker compose -f docker-compose.yml up -d --remove-orphans"


def test_deploy_refuse_execution_si_desactive(tmp_path: Path) -> None:
    """Verifie que l'execution reelle reste bloquee tant que le service est desactive."""
    service = DeploymentService(root_dir=str(tmp_path), enabled=False)

    result = asyncio.run(
        service.deploy(
            service="builder",
            target="local",
            force_rebuild=False,
            dry_run=False,
        )
    )

    assert result["status"] == "disabled"
    assert "desactive" in result["message"]


def test_deploy_execute_localement_avec_processus_simule(tmp_path: Path, monkeypatch) -> None:
    """Verifie qu'un deploiement local agrege correctement la sortie des commandes."""

    class FakeProcess:
        """Simule un sous-processus asyncio local."""

        def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
            self._stdout = stdout
            self._stderr = stderr
            self.returncode = returncode

        async def communicate(self) -> tuple[bytes, bytes]:
            return (
                self._stdout.encode("utf-8"),
                self._stderr.encode("utf-8"),
            )

    executed_commands: list[str] = []

    async def fake_create_subprocess_shell(command: str, cwd: str, stdout, stderr):
        """Simule l'execution locale sans Docker reel."""
        del cwd, stdout, stderr
        executed_commands.append(command)
        if command.endswith("ps builder"):
            return FakeProcess("builder up", "", 0)
        return FakeProcess(f"ok:{command}", "", 0)

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create_subprocess_shell)

    service = DeploymentService(root_dir=str(tmp_path), enabled=True)
    result = asyncio.run(
        service.deploy(
            service="builder",
            target="local",
            force_rebuild=True,
            dry_run=False,
        )
    )

    assert result["status"] == "success"
    assert len(executed_commands) == 3
    assert executed_commands[0] == "docker compose -f docker-compose.yml build --no-cache builder"
    assert result["execution"]["returncode"] == 0
    assert "builder up" in result["execution"]["stdout"]
