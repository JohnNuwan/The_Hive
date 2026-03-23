"""Synchronise les correctifs locaux vers le serveur Proxmox puis redeploie Docker."""

from __future__ import annotations

import os
import posixpath
from pathlib import Path

import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)
REMOTE_ROOT = "/home/aza/The_Hive"

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

def read_local_env_value(key: str, env_path: Path | None = None) -> str:
    """Lit une cle du fichier .env local (derniere occurrence prioritaire)."""
    target = env_path or Path(".env")
    if not target.exists():
        return ""

    value = ""
    for raw_line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        current_key, current_value = line.split("=", 1)
        if current_key.strip() != key:
            continue

        normalized = current_value.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
            normalized = normalized[1:-1]
        value = normalized

    return value.strip()


def redact_env_value(key: str, value: str) -> str:
    """Masque les valeurs sensibles avant affichage dans les logs."""
    upper_key = key.upper()
    sensitive_markers = ("TOKEN", "PASSWORD", "SECRET", "KEY")
    if not any(marker in upper_key for marker in sensitive_markers):
        return value
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}***{value[-4:]}"


def sanitize_command_for_log(command: str) -> str:
    """Masque le mot de passe sudo dans les logs de commandes."""
    if not SUDO_PASS:
        return command
    return command.replace(SUDO_PASS, "***")



FILES_TO_UPLOAD = [
    "docker-compose.yml",
    "src/openclaw/setup.py",
    "src/eva-builder/eva_builder/services/bmad_prompts.py",
    "src/eva-builder/eva_builder/services/factory.py",
    "src/eva-core/eva_core/services/council.py",
    "src/eva-core/eva_core/services/llm.py",
    "src/eva-core/eva_core/main.py",
    "src/eva-core/eva_core/memory_layer.py",
    "src/eva-compliance/eva_compliance/main.py",
]

ENV_UPDATES = {
    "HIVE_REDIS_HOST": "redis",
    "HIVE_QDRANT_HOST": "qdrant",
    "HIVE_NEO4J_HOST": "neo4j",
    "HIVE_MQTT_HOST": "mosquitto",
    "HIVE_VLLM_HOST": "vllm",
    "HIVE_VLLM_PORT": "8000",
    "HIVE_LLM_BACKEND": "vllm",
    "LLM_BACKEND": "vllm",
    "VLLM_MODEL_NAME": "Qwen/Qwen2.5-1.5B-Instruct",
    "COUNCIL_MODEL_GENERAL": "Qwen/Qwen2.5-1.5B-Instruct",
    "COUNCIL_MODEL_RESEARCH": "Qwen/Qwen2.5-1.5B-Instruct",
    "COUNCIL_MODEL_BANKER": "Qwen/Qwen2.5-1.5B-Instruct",
    "COUNCIL_MODEL_CODE": "Qwen/Qwen2.5-1.5B-Instruct",
    "EVA_BUILDER_LLM_BACKEND": "vllm",
    "EVA_BUILDER_LLM_URL": "http://vllm:8000/v1/chat/completions",
    "EVA_BUILDER_LLM_MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
}

# Propagation optionnelle du token HF local vers le serveur.
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip() or read_local_env_value("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    ENV_UPDATES["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


def run_stream(client: paramiko.SSHClient, command: str) -> int:
    """Execute une commande distante en stream et retourne le code de sortie."""
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line, end="")
    err = stderr.read().decode(errors="replace")
    if err.strip():
        print(err)
    return stdout.channel.recv_exit_status()


def ensure_remote_dir(client: paramiko.SSHClient, remote_path: str) -> None:
    """Cree le dossier parent distant si necessaire."""
    remote_dir = posixpath.dirname(remote_path)
    run_stream(client, f"mkdir -p '{remote_dir}'")


def sync_files(client: paramiko.SSHClient) -> None:
    """Upload les fichiers critiques vers le serveur distant."""
    sftp = client.open_sftp()
    try:
        for rel_path in FILES_TO_UPLOAD:
            local_path = Path(rel_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Fichier local introuvable: {rel_path}")

            remote_path = posixpath.join(REMOTE_ROOT, rel_path.replace("\\", "/"))
            ensure_remote_dir(client, remote_path)
            print(f"Upload: {rel_path} -> {remote_path}")
            sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def apply_env_updates(client: paramiko.SSHClient) -> None:
    """Applique les variables d'environnement dans le .env distant sans sur-quoting."""
    remote_env_path = posixpath.join(REMOTE_ROOT, ".env")
    sftp = client.open_sftp()
    try:
        try:
            with sftp.open(remote_env_path, "rb") as env_file:
                raw_content = env_file.read().decode("utf-8", errors="replace")
        except OSError:
            raw_content = ""

        lines = raw_content.splitlines()
        rendered_lines: list[str] = []
        already_updated: set[str] = set()

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                rendered_lines.append(line)
                continue

            key, _ = line.split("=", 1)
            clean_key = key.strip()
            if clean_key in ENV_UPDATES:
                value = ENV_UPDATES[clean_key]
                print(f".env update: {clean_key}={redact_env_value(clean_key, value)}")
                rendered_lines.append(f"{clean_key}={value}")
                already_updated.add(clean_key)
            else:
                rendered_lines.append(line)

        for key, value in ENV_UPDATES.items():
            if key in already_updated:
                continue
            print(f".env add: {key}={redact_env_value(key, value)}")
            rendered_lines.append(f"{key}={value}")

        new_content = "\n".join(rendered_lines).rstrip("\n") + "\n"
        with sftp.open(remote_env_path, "wb") as env_file:
            env_file.write(new_content.encode("utf-8"))
    finally:
        sftp.close()


def deploy_stack(client: paramiko.SSHClient) -> None:
    """Rebuild et redemarre uniquement les services critiques modifies."""
    commands = [
        f"cd '{REMOTE_ROOT}' && echo '{SUDO_PASS}' | sudo -S docker compose pull vllm",
        f"cd '{REMOTE_ROOT}' && echo '{SUDO_PASS}' | sudo -S docker compose build core builder compliance",
        f"cd '{REMOTE_ROOT}' && echo '{SUDO_PASS}' | sudo -S docker compose up -d --no-deps --force-recreate --scale core=1 --scale builder=1 vllm core builder",
        f"cd '{REMOTE_ROOT}' && echo '{SUDO_PASS}' | sudo -S docker compose up -d --no-deps --force-recreate --scale kernel=1 --scale nervous=1 --scale sentinel=1 --scale compliance=1 --scale shadow=1 --scale wraith=1 --scale substrate=1 --scale accountant=1 --scale lab=1 --scale rwa=1 --scale muse=1 --scale sage=1 --scale researcher=1 kernel nervous sentinel compliance shadow wraith substrate accountant lab rwa muse sage researcher",
    ]

    for command in commands:
        print(f"Run: {sanitize_command_for_log(command)}")
        code = run_stream(client, command)
        if code != 0:
            raise RuntimeError(f"Commande echouee (code={code}): {command}")


def main() -> int:
    """Point d'entree du script de synchronisation/deploiement."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"Connexion SSH {USER}@{HOST}...")
        client.connect(HOST, username=USER, password=PASS, timeout=20)

        sync_files(client)
        apply_env_updates(client)
        deploy_stack(client)

        print("Deploiement termine.")
        return 0
    except Exception as exc:
        print(f"Erreur deploiement: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())


