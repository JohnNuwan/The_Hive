"""Genere et optionnellement deploie les configurations WireGuard Hydra."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import paramiko

LOCAL_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from scripts.deploy.start_training_proxmox import (
    HOST,
    REMOTE_DIR,
    USER,
    _require_remote_credentials,
    ensure_remote_parent,
    run_command,
    upload_file,
)

TEMPLATE_DIR = LOCAL_ROOT / "infra" / "wireguard"


def parse_args() -> argparse.Namespace:
    """
    Analyse la ligne de commande du generateur WireGuard.

    Returns:
        argparse.Namespace: Arguments normalises.
    """
    parser = argparse.ArgumentParser(description="Genere les configurations WireGuard Hydra.")
    parser.add_argument(
        "--manifest",
        default=str(TEMPLATE_DIR / "peers.sample.json"),
        help="Fichier JSON de definition serveur + peers.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_ROOT / "data" / "hydra" / "wireguard"),
        help="Dossier local de sortie des artefacts WireGuard.",
    )
    parser.add_argument(
        "--interface",
        default=os.getenv("WIREGUARD_INTERFACE", "wg0"),
        help="Nom de l'interface WireGuard cible.",
    )
    parser.add_argument(
        "--remote-dir",
        default=f"{REMOTE_DIR}/data/hydra/wireguard",
        help="Dossier distant de depot si --deploy-remote est utilise.",
    )
    parser.add_argument(
        "--deploy-remote",
        action="store_true",
        help="Envoie les artefacts generes sur le serveur Linux.",
    )
    parser.add_argument(
        "--install-remote",
        action="store_true",
        help="Installe la configuration serveur dans /etc/wireguard sur le noeud distant.",
    )
    parser.add_argument(
        "--restart-remote",
        action="store_true",
        help="Redemarre wg-quick@<interface> apres installation distante.",
    )
    return parser.parse_args()


def _render_template(template: str, values: dict[str, str]) -> str:
    """
    Remplace les placeholders `{{VAR}}` d'un template texte.

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


def _load_manifest(path: Path) -> dict[str, Any]:
    """
    Charge le manifeste JSON WireGuard.

    Args:
        path (Path): Emplacement du manifeste.

    Returns:
        dict[str, Any]: Structure chargee.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _build_peer_block(peer: dict[str, Any]) -> str:
    """
    Construit le bloc `[Peer]` cote serveur.

    Args:
        peer (dict[str, Any]): Definition d'un client.

    Returns:
        str: Bloc de configuration serveur.
    """
    return "\n".join(
        [
            "[Peer]",
            f"# {peer['name']}",
            f"PublicKey = {peer['public_key']}",
            f"PresharedKey = {peer.get('preshared_key') or 'REMPLACER_PRESHARED_KEY'}",
            f"AllowedIPs = {peer.get('server_allowed_ips') or peer['address']}",
            "",
        ]
    )


def generate_wireguard_artifacts(manifest: dict[str, Any], output_dir: Path, interface: str) -> list[Path]:
    """
    Genere les fichiers serveur et clients a partir d'un manifeste.

    Args:
        manifest (dict[str, Any]): Definition JSON.
        output_dir (Path): Dossier local cible.
        interface (str): Nom d'interface serveur.

    Returns:
        list[Path]: Liste des artefacts ecrits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clients_dir = output_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    server = manifest["server"]
    peers = manifest.get("peers") or []
    server_template = (TEMPLATE_DIR / "wg0.conf.template").read_text(encoding="utf-8")
    client_template = (TEMPLATE_DIR / "peer-client.conf.template").read_text(encoding="utf-8")

    rendered_server = _render_template(
        server_template,
        {
            "SERVER_ADDRESS": str(server["address"]),
            "SERVER_PORT": str(server["listen_port"]),
            "SERVER_PRIVATE_KEY": str(server["private_key"]),
            "PEER_BLOCKS": "\n".join(_build_peer_block(peer) for peer in peers).strip(),
        },
    )
    rendered_files: list[Path] = []
    server_path = output_dir / f"{interface}.conf"
    server_path.write_text(rendered_server.strip() + "\n", encoding="utf-8")
    rendered_files.append(server_path)

    for peer in peers:
        rendered_client = _render_template(
            client_template,
            {
                "CLIENT_ADDRESS": str(peer["address"]),
                "CLIENT_PRIVATE_KEY": str(peer["private_key"]),
                "CLIENT_DNS": str(peer.get("dns") or server.get("dns") or "1.1.1.1"),
                "SERVER_PUBLIC_KEY": str(server["public_key"]),
                "PRESHARED_KEY": str(peer.get("preshared_key") or "REMPLACER_PRESHARED_KEY"),
                "CLIENT_ALLOWED_IPS": str(peer.get("client_allowed_ips") or server["address"]),
                "SERVER_PUBLIC_ENDPOINT": str(server["public_endpoint"]),
                "SERVER_PORT": str(server["listen_port"]),
            },
        )
        client_path = clients_dir / f"{peer['name']}.conf"
        client_path.write_text(rendered_client.strip() + "\n", encoding="utf-8")
        rendered_files.append(client_path)

    manifest_copy = output_dir / "manifest.rendered.json"
    manifest_copy.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rendered_files.append(manifest_copy)
    return rendered_files


def deploy_remote(artifacts: list[Path], output_dir: Path, remote_dir: str, interface: str, install_remote: bool, restart_remote: bool) -> None:
    """
    Envoie les artefacts WireGuard sur le serveur cible.

    Args:
        artifacts (list[Path]): Fichiers a copier.
        output_dir (Path): Racine locale commune.
        remote_dir (str): Racine distante.
        interface (str): Nom d'interface WireGuard.
        install_remote (bool): Installe dans /etc/wireguard si vrai.
        restart_remote (bool): Redemarre le service distant si vrai.
    """
    ssh_password, sudo_password = _require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=ssh_password, timeout=15)
        sftp = client.open_sftp()
        for artifact in artifacts:
            relative = artifact.relative_to(output_dir)
            remote_path = f"{remote_dir}/{relative.as_posix()}"
            ensure_remote_parent(sftp, remote_path)
            upload_file(sftp, artifact, remote_path)
        sftp.close()

        if install_remote:
            remote_server_path = f"{remote_dir}/{interface}.conf"
            install_command = (
                f"echo '{sudo_password}' | sudo -S bash -lc "
                f"'install -m 600 {remote_server_path} /etc/wireguard/{interface}.conf'"
            )
            output, error, code = run_command(client, install_command, timeout=60)
            if code != 0:
                raise RuntimeError(error or output or "Installation WireGuard distante echouee.")
            if restart_remote:
                restart_command = (
                    f"echo '{sudo_password}' | sudo -S bash -lc "
                    f"'systemctl restart wg-quick@{interface} && systemctl enable wg-quick@{interface}'"
                )
                output, error, code = run_command(client, restart_command, timeout=60)
                if code != 0:
                    raise RuntimeError(error or output or "Redemarrage WireGuard distant echoue.")
    finally:
        client.close()


def main() -> None:
    """
    Point d'entree du generateur WireGuard Hydra.
    """
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest = _load_manifest(manifest_path)
    artifacts = generate_wireguard_artifacts(manifest, output_dir, args.interface)
    print(f"Artefacts WireGuard generes dans {output_dir}")
    for artifact in artifacts:
        print(f" - {artifact}")

    if args.deploy_remote:
        deploy_remote(
            artifacts,
            output_dir,
            args.remote_dir,
            args.interface,
            install_remote=args.install_remote,
            restart_remote=args.restart_remote,
        )
        print(f"Artefacts WireGuard copies sur {HOST}:{args.remote_dir}")


if __name__ == "__main__":
    main()
