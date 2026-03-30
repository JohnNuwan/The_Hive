"""Prepare les artefacts Hydra pour plusieurs terminaux MT5 sous Wine."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import paramiko
from pydantic import BaseModel, Field

LOCAL_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))
for extra in ("src/shared", "src/eva-banker"):
    extra_path = LOCAL_ROOT / extra
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from scripts.deploy.start_training_proxmox import (
    HOST,
    USER,
    _require_remote_credentials,
    ensure_remote_parent,
    upload_file,
)
from shared import HydraAccountRole, HydraScalingMode, PropFirmAccount, get_settings

TEMPLATE_DIR = LOCAL_ROOT / "infra" / "hydra"


class HydraProvisionServer(BaseModel):
    """
    Configuration serveur pour la provision Hydra.

    Attributes:
        remote_repo_dir (str): Emplacement du depot sur le serveur.
        hydra_root (str): Racine des artefacts Hydra.
        wine_python_exe (str): Python Windows execute sous Wine.
        mt5_executable (str): Executable MT5 a lancer.
        terminal_host (str): Hote HTTP local de l'executeur.
        base_port (int): Port de base des executeurs.
        cpu_set (str): Pinning CPU recommande.
        banker_url (str): API banker a utiliser pour l'enregistrement.
    """

    remote_repo_dir: str
    hydra_root: str
    wine_python_exe: str
    mt5_executable: str
    terminal_host: str = "127.0.0.1"
    base_port: int = 19100
    cpu_set: str = "0-3"
    banker_url: str = "http://127.0.0.1:8100"


class HydraProvisionAccount(BaseModel):
    """
    Definition complete d'un compte Hydra a provisionner.

    Attributes:
        id (UUID): Identifiant Hydra du compte.
        password (str): Mot de passe MT5.
        account (PropFirmAccount): Configuration banker associee.
    """

    id: UUID
    password: str
    account: PropFirmAccount


class HydraProvisionManifest(BaseModel):
    """
    Manifeste complet de provision Wine Hydra.

    Attributes:
        server (HydraProvisionServer): Parametres du serveur.
        accounts (list[HydraProvisionAccount]): Comptes a preparer.
    """

    server: HydraProvisionServer
    accounts: list[HydraProvisionAccount]


def parse_args() -> argparse.Namespace:
    """
    Analyse la ligne de commande du provisionneur Hydra.

    Returns:
        argparse.Namespace: Arguments normalises.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Prepare les terminaux Hydra MT5 sous Wine.")
    parser.add_argument(
        "--manifest",
        default=str(TEMPLATE_DIR / "examples" / "hydra_accounts.sample.json"),
        help="Manifeste JSON des comptes Hydra.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_ROOT / "data" / "hydra" / "provision"),
        help="Dossier local de sortie.",
    )
    parser.add_argument(
        "--deploy-remote",
        action="store_true",
        help="Copie les artefacts sur le serveur Linux.",
    )
    parser.add_argument(
        "--register-banker",
        action="store_true",
        help="Enregistre les comptes dans le banker via /hydra/accounts.",
    )
    parser.add_argument(
        "--banker-url",
        default=None,
        help="Surcharge l'URL du banker pour l'enregistrement API.",
    )
    parser.add_argument(
        "--remote-root",
        default=settings.hydra_remote_root,
        help="Racine distante de depot des artefacts Hydra.",
    )
    return parser.parse_args()


def _render_template(template: str, values: dict[str, str]) -> str:
    """
    Rend un template simple a placeholders.

    Args:
        template (str): Texte source.
        values (dict[str, str]): Valeurs de remplacement.

    Returns:
        str: Texte rendu.
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def _slugify(value: str) -> str:
    """
    Produit un identifiant de fichier stable.

    Args:
        value (str): Nom libre.

    Returns:
        str: Version simplifiee utilisable en chemin.
    """
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _linux_to_windows_path(path: str) -> str:
    """
    Convertit un chemin Linux du serveur en chemin Windows vu par Wine.

    Args:
        path (str): Chemin Linux absolu.

    Returns:
        str: Chemin Windows de type `Z:\...`.
    """
    normalized = str(path).replace("/", "\\")
    if normalized.startswith("\\"):
        normalized = normalized[1:]
    return f"Z:\\{normalized}"


def _load_manifest(path: Path) -> HydraProvisionManifest:
    """
    Charge et normalise le manifeste Hydra Wine.

    Args:
        path (Path): Fichier JSON source.

    Returns:
        HydraProvisionManifest: Manifeste valide.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    server = HydraProvisionServer(**payload["server"])
    accounts: list[HydraProvisionAccount] = []
    for raw_account in payload.get("accounts") or []:
        account_payload = dict(raw_account)
        password = str(account_payload.pop("password"))
        if "role" not in account_payload:
            account_payload["role"] = HydraAccountRole.SLAVE
        if "scaling_mode" not in account_payload:
            account_payload["scaling_mode"] = HydraScalingMode.FIXED
        account = PropFirmAccount(**account_payload)
        accounts.append(
            HydraProvisionAccount(
                id=account.id,
                password=password,
                account=account,
            )
        )
    return HydraProvisionManifest(server=server, accounts=accounts)


def generate_hydra_artifacts(manifest: HydraProvisionManifest, output_dir: Path) -> list[Path]:
    """
    Genere les artefacts Wine/systemd et payloads banker.

    Args:
        manifest (HydraProvisionManifest): Definition source.
        output_dir (Path): Racine locale cible.

    Returns:
        list[Path]: Fichiers ecrits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_files: list[Path] = []
    service_template = (TEMPLATE_DIR / "systemd" / "hydra-terminal@.service.template").read_text(encoding="utf-8")
    launcher_template = (TEMPLATE_DIR / "bin" / "run_hydra_terminal.sh.template").read_text(encoding="utf-8")

    registry_payload = []
    for index, item in enumerate(manifest.accounts):
        account = item.account
        slug = _slugify(account.name or str(account.id))
        account_dir = output_dir / slug
        account_dir.mkdir(parents=True, exist_ok=True)

        remote_account_dir = f"{manifest.server.hydra_root}/accounts/{slug}"
        wineprefix = account.wineprefix or f"{manifest.server.hydra_root}/mt5/{slug}/prefix"
        terminal_path = account.terminal_path or manifest.server.mt5_executable
        terminal_port = manifest.server.base_port + index
        executor_url = account.executor_url or f"http://{manifest.server.terminal_host}:{terminal_port}"

        banker_payload = account.model_copy(
            update={
                "executor_url": executor_url,
                "wineprefix": wineprefix,
                "terminal_path": terminal_path,
                "role": HydraAccountRole.SLAVE,
            }
        )
        registry_payload.append(banker_payload.model_dump(mode="json"))

        env_path = account_dir / ".env"
        env_lines = [
            f'HYDRA_ACCOUNT_UUID="{account.id}"',
            f'HYDRA_TERMINAL_HOST="{manifest.server.terminal_host}"',
            f'HYDRA_TERMINAL_PORT="{terminal_port}"',
            f'HYDRA_TERMINAL_PATH="{terminal_path}"',
            f'WINEPREFIX="{wineprefix}"',
            f'MT5_ACCOUNT_ID="{account.login}"',
            f'MT5_PASSWORD="{item.password}"',
            f'MT5_SERVER="{account.server}"',
            'MOCK_MT5=false',
            'PAPER_TRADING=false',
        ]
        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        rendered_files.append(env_path)

        windows_pythonpath = ";".join(
            [
                _linux_to_windows_path(f"{manifest.server.remote_repo_dir}/src/eva-banker"),
                _linux_to_windows_path(f"{manifest.server.remote_repo_dir}/src/shared"),
            ]
        )
        launcher_content = _render_template(
            launcher_template,
            {
                "WINEPREFIX": wineprefix,
                "WINDOWS_PYTHONPATH": windows_pythonpath,
                "ENV_FILE": f"{remote_account_dir}/.env",
                "ACCOUNT_DIR": remote_account_dir,
                "MT5_EXECUTABLE": terminal_path,
                "WINE_PYTHON_EXE": manifest.server.wine_python_exe,
            },
        )
        launcher_path = account_dir / "run_hydra_terminal.sh"
        launcher_path.write_text(launcher_content.strip() + "\n", encoding="utf-8")
        launcher_path.chmod(0o755)
        rendered_files.append(launcher_path)

        service_content = _render_template(
            service_template,
            {
                "ACCOUNT_NAME": account.name,
                "REMOTE_REPO_DIR": manifest.server.remote_repo_dir,
                "ENV_FILE": f"{remote_account_dir}/.env",
                "LAUNCHER": f"{remote_account_dir}/run_hydra_terminal.sh",
                "CPU_SET": manifest.server.cpu_set,
            },
        )
        service_path = account_dir / f"hydra-terminal-{slug}.service"
        service_path.write_text(service_content.strip() + "\n", encoding="utf-8")
        rendered_files.append(service_path)

        banker_payload_path = account_dir / "banker_account.json"
        banker_payload_path.write_text(
            json.dumps(banker_payload.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rendered_files.append(banker_payload_path)

    registry_path = output_dir / "hydra_registry.json"
    registry_path.write_text(json.dumps(registry_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rendered_files.append(registry_path)
    return rendered_files


def deploy_remote(artifacts: list[Path], output_dir: Path, remote_root: str) -> None:
    """
    Copie les artefacts Hydra vers le serveur Linux.

    Args:
        artifacts (list[Path]): Fichiers a envoyer.
        output_dir (Path): Racine locale commune.
        remote_root (str): Racine distante cible.
    """
    ssh_password, _ = _require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=ssh_password, timeout=15)
        sftp = client.open_sftp()
        for artifact in artifacts:
            relative = artifact.relative_to(output_dir)
            remote_path = f"{remote_root}/{relative.as_posix()}"
            ensure_remote_parent(sftp, remote_path)
            upload_file(sftp, artifact, remote_path)
        sftp.close()
    finally:
        client.close()


def register_accounts(manifest: HydraProvisionManifest, banker_url: str) -> None:
    """
    Enregistre les comptes Hydra via l'API banker.

    Args:
        manifest (HydraProvisionManifest): Manifeste de comptes.
        banker_url (str): URL de l'API banker.
    """
    with httpx.Client(timeout=10.0) as client:
        for index, item in enumerate(manifest.accounts):
            account = item.account
            executor_url = account.executor_url or f"http://{manifest.server.terminal_host}:{manifest.server.base_port + index}"
            payload = account.model_copy(
                update={
                    "executor_url": executor_url,
                    "role": HydraAccountRole.SLAVE,
                }
            ).model_dump(mode="json")
            response = client.post(f"{banker_url.rstrip('/')}/hydra/accounts", json=payload)
            response.raise_for_status()


def main() -> None:
    """
    Point d'entree du provisionneur Hydra Wine.
    """
    args = parse_args()
    manifest = _load_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir)
    artifacts = generate_hydra_artifacts(manifest, output_dir)
    print(f"Artefacts Hydra Wine generes dans {output_dir}")
    for artifact in artifacts:
        print(f" - {artifact}")

    if args.deploy_remote:
        deploy_remote(artifacts, output_dir, args.remote_root)
        print(f"Artefacts copies sur {HOST}:{args.remote_root}")

    if args.register_banker:
        register_accounts(manifest, args.banker_url or manifest.server.banker_url)
        print(f"Comptes Hydra enregistres via {args.banker_url or manifest.server.banker_url}")


if __name__ == "__main__":
    main()
