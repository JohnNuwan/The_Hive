import os
import paramiko


def main():
    host = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
    user = os.getenv("HIVE_SSH_USER", "aza")
    password = os.getenv("HIVE_SSH_PASSWORD")

    if not password:
        raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

    print("Connexion au serveur Proxmox...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(host, username=user, password=password)

        print("Mise a jour et redemarrage de The Banker...")
        command = "cd /home/aza/The_Hive && git pull origin feat/sprint-6 && docker rm -f the_hive-banker-1 && docker compose up -d banker"
        stdin, stdout, stderr = client.exec_command(command)

        for line in stdout:
            print(line.strip())

        err = stderr.read().decode()
        if err:
            print(f"Erreur / Warning: {err}")

    except Exception as e:
        print(f"Erreur SSH : {e}")
    finally:
        client.close()
        print("Deconnexion.")


if __name__ == "__main__":
    main()