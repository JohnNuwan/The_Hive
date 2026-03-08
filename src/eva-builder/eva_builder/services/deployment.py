"""Service de deploiement pilote par `eva-builder`."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

try:
    import paramiko
except Exception:  # pragma: no cover - optionnel au chargement
    paramiko = None

logger = logging.getLogger(__name__)


class DeploymentService:
    """Prepare et execute des deploiements locaux ou Proxmox."""

    DEFAULT_REMOTE_DIR = "/home/aza/The_Hive"

    def __init__(
        self,
        root_dir: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Initialise les chemins et l'etat d'activation.

        Args:
            root_dir (str | None): Racine locale du depot.
            enabled (bool | None): Active l'execution reelle des deploiements.
        """
        self.root_dir = Path(root_dir or Path.cwd())
        if enabled is None:
            enabled = os.getenv("EVA_BUILDER_DEPLOY_ENABLED", "false").strip().lower() == "true"
        self.enabled = enabled
        self.remote_host = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
        self.remote_user = os.getenv("HIVE_SSH_USER", "aza")
        self.remote_password = os.getenv("HIVE_SSH_PASSWORD")
        self.remote_sudo_password = os.getenv("HIVE_SUDO_PASSWORD") or self.remote_password or ""
        self.remote_dir = os.getenv("HIVE_REMOTE_DIR", self.DEFAULT_REMOTE_DIR)

    async def deploy(
        self,
        service: str,
        target: str = "proxmox",
        force_rebuild: bool = False,
        dry_run: bool = True,
        compose_file: str | None = None,
    ) -> dict[str, Any]:
        """Prepare ou execute un deploiement.

        Args:
            service (str): Service logique a deployer.
            target (str): Cible de deploiement (`proxmox` ou `local`).
            force_rebuild (bool): Force un rebuild des images.
            dry_run (bool): Retourne le plan sans execution.
            compose_file (str | None): Fichier compose a utiliser si necessaire.

        Returns:
            dict[str, Any]: Resume du deploiement ou de son plan.
        """
        plan = self._build_plan(
            service=service,
            target=target,
            force_rebuild=force_rebuild,
            compose_file=compose_file,
        )
        plan["dry_run"] = dry_run

        if dry_run:
            return {"status": "dry_run", "deployment": plan}

        if not self.enabled:
            return {
                "status": "disabled",
                "message": "Le deploiement Builder est desactive par configuration.",
                "deployment": plan,
            }

        if target == "local":
            execution = await self._run_local(plan["commands"])
        elif target == "proxmox":
            execution = await self._run_remote(plan["commands"])
        else:
            raise ValueError(f"Cible de deploiement non supportee: {target}")

        return {
            "status": "success" if execution["returncode"] == 0 else "failed",
            "deployment": plan,
            "execution": execution,
        }

    def _build_plan(
        self,
        service: str,
        target: str,
        force_rebuild: bool,
        compose_file: str | None,
    ) -> dict[str, Any]:
        """Construit un plan de deploiement deterministe.

        Args:
            service (str): Service logique a deployer.
            target (str): Cible de deploiement.
            force_rebuild (bool): Force un rebuild.
            compose_file (str | None): Fichier compose optionnel.

        Returns:
            dict[str, Any]: Plan de commande a executer.
        """
        normalized_service = service.strip().lower()
        normalized_target = target.strip().lower()
        compose = compose_file or self._resolve_compose_file(normalized_service)
        commands = self._build_commands(
            service=normalized_service,
            compose_file=compose,
            force_rebuild=force_rebuild,
        )

        return {
            "service": normalized_service,
            "target": normalized_target,
            "compose_file": compose,
            "force_rebuild": force_rebuild,
            "commands": commands,
            "remote_dir": self.remote_dir if normalized_target == "proxmox" else None,
        }

    @staticmethod
    def _resolve_compose_file(service: str) -> str:
        """Associe un service logique a son fichier compose.

        Args:
            service (str): Service logique cible.

        Returns:
            str: Fichier compose a utiliser.
        """
        if service == "audiocraft":
            return "docker-compose.audiocraft.yml"
        if service == "comfyui":
            return "docker-compose.comfyui.yml"
        return "docker-compose.yml"

    @staticmethod
    def _build_commands(
        service: str,
        compose_file: str,
        force_rebuild: bool,
    ) -> list[str]:
        """Construit les commandes Docker Compose a executer.

        Args:
            service (str): Service logique cible.
            compose_file (str): Fichier compose a utiliser.
            force_rebuild (bool): Force un rebuild si vrai.

        Returns:
            list[str]: Commandes shell ordonnees.
        """
        compose_base = f"docker compose -f {compose_file}"
        commands: list[str] = []

        if service == "swarm":
            if force_rebuild:
                commands.append(f"{compose_base} build --no-cache")
            else:
                commands.append(f"{compose_base} build")
            commands.append(f"{compose_base} up -d --remove-orphans")
            commands.append(f"{compose_base} ps")
            return commands

        service_groups = {
            "muse-nexus": ["muse", "nexus"],
            "core-stack": ["core", "lab", "nexus"],
        }
        services = service_groups.get(service, [service])
        service_args = " ".join(services)

        build_cmd = f"{compose_base} build {'--no-cache ' if force_rebuild else ''}{service_args}".strip()
        commands.append(build_cmd)
        commands.append(f"{compose_base} up -d --no-deps {service_args}")
        commands.append(f"{compose_base} ps {service_args}")
        return commands

    async def _run_local(self, commands: list[str]) -> dict[str, Any]:
        """Execute les commandes localement.

        Args:
            commands (list[str]): Commandes a lancer.

        Returns:
            dict[str, Any]: Resultat agrege.
        """
        output_chunks: list[str] = []
        error_chunks: list[str] = []
        returncode = 0

        for command in commands:
            logger.info("Deploiement local: %s", command)
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.root_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            output_chunks.append(stdout.decode("utf-8", errors="replace"))
            error_chunks.append(stderr.decode("utf-8", errors="replace"))
            returncode = process.returncode
            if returncode != 0:
                break

        return {
            "mode": "local",
            "stdout": "\n".join(chunk for chunk in output_chunks if chunk),
            "stderr": "\n".join(chunk for chunk in error_chunks if chunk),
            "returncode": returncode,
        }

    async def _run_remote(self, commands: list[str]) -> dict[str, Any]:
        """Execute les commandes sur Proxmox via SSH.

        Args:
            commands (list[str]): Commandes shell a executer a distance.

        Returns:
            dict[str, Any]: Resultat agrege.
        """
        if paramiko is None:
            raise RuntimeError("Paramiko est requis pour le deploiement distant.")
        if not self.remote_password:
            raise RuntimeError("HIVE_SSH_PASSWORD manquant pour le deploiement distant.")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self.remote_host,
            username=self.remote_user,
            password=self.remote_password,
            timeout=15,
        )

        output_chunks: list[str] = []
        error_chunks: list[str] = []
        returncode = 0

        try:
            for command in commands:
                remote_command = (
                    f"cd {self.remote_dir} && "
                    f"echo '{self.remote_sudo_password}' | sudo -S {command}"
                )
                logger.info("Deploiement distant: %s", remote_command)
                _, stdout, stderr = client.exec_command(remote_command, get_pty=True)
                out_text = stdout.read().decode("utf-8", errors="replace")
                err_text = stderr.read().decode("utf-8", errors="replace")
                output_chunks.append(out_text)
                error_chunks.append(err_text)
                returncode = stdout.channel.recv_exit_status()
                if returncode != 0:
                    break
        finally:
            client.close()

        return {
            "mode": "proxmox",
            "stdout": "\n".join(chunk for chunk in output_chunks if chunk),
            "stderr": "\n".join(chunk for chunk in error_chunks if chunk),
            "returncode": returncode,
            "host": self.remote_host,
            "user": self.remote_user,
        }
