import os
import paramiko
import sys

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def debug_remote():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)

        print("--- Docker PS ---")
        stdin, stdout, stderr = client.exec_command(f"echo '{SUDO_PASS}' | sudo -S docker ps -a | grep lab")
        print(stdout.read().decode())

        print("--- Docker Logs (tail 50) ---")
        stdin, stdout, stderr = client.exec_command(f"echo '{SUDO_PASS}' | sudo -S docker logs --tail 50 hive-lab")
        print(stdout.read().decode())
        print("ERRORS:", stderr.read().decode())

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    debug_remote()