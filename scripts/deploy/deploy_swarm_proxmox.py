import os
import paramiko
import sys
import time

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)
REMOTE_DIR = "/home/aza/The_Hive"
STACK_NAME = "hive"
ENABLE_DOCKER_BANKER = os.getenv("HIVE_ENABLE_DOCKER_BANKER", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def deploy_swarm():
    """Deploie la stack sur Proxmox en mode hybride ou full-docker."""
    print(f"Connexion au serveur Proxmox {HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=15)
        print("Connexion etablie. Synchronisation du depot distant...")

        git_cmds = f"cd {REMOTE_DIR} && git fetch origin feat/sprint-6 && git reset --hard origin/feat/sprint-6"
        _, stdout, _ = client.exec_command(git_cmds)
        print(stdout.read().decode(errors="replace"))

        print("Verification du mode Swarm...")
        swarm_init_cmd = f"echo '{SUDO_PASS}' | sudo -S docker swarm init || true"
        client.exec_command(swarm_init_cmd)
        time.sleep(2)

        compose_base = "docker compose"
        if ENABLE_DOCKER_BANKER:
            compose_base += " --profile with-banker"
            print("Mode compose: avec service banker conteneurise.")
        else:
            print("Mode compose hybride: banker conteneurise desactive (banker.bat local attendu).")
            print("Conseil: verifier BANKER_API_HOST dans .env cote serveur (IP du PC local).")
            check_env_cmd = f"cd {REMOTE_DIR} && (grep -E '^BANKER_API_HOST=' .env || true)"
            _, env_out, _ = client.exec_command(check_env_cmd)
            banker_host_line = env_out.read().decode(errors="replace").strip()
            if banker_host_line:
                print(f"BANKER_API_HOST detecte: {banker_host_line}")
            else:
                print("ATTENTION: BANKER_API_HOST absent du .env distant.")

        print(f"Deploiement de la stack '{STACK_NAME}' via docker-compose.yml...")

        build_cmd = f"cd {REMOTE_DIR} && echo '{SUDO_PASS}' | sudo -S {compose_base} build"
        print("Build des images en cours...")
        _, stdout, _ = client.exec_command(build_cmd, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="")

        deploy_cmd = f"cd {REMOTE_DIR} && echo '{SUDO_PASS}' | sudo -S {compose_base} up -d --remove-orphans"
        _, stdout, _ = client.exec_command(deploy_cmd, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="")

        status_cmd = f"cd {REMOTE_DIR} && echo '{SUDO_PASS}' | sudo -S {compose_base} ps"
        _, stdout, _ = client.exec_command(status_cmd)
        print("\nEtat des services Docker Compose:")
        print(stdout.read().decode(errors="replace"))

    except Exception as e:
        print(f"Echec du deploiement: {e}")
        sys.exit(1)
    finally:
        client.close()
        print("Connexion fermee.")


if __name__ == "__main__":
    deploy_swarm()
