import os
import paramiko
import time


def main():
    host = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
    user = os.getenv("HIVE_SSH_USER", "aza")
    password = os.getenv("HIVE_SSH_PASSWORD")

    if not password:
        raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

    print("Verification du statut de l'entrainement actuel...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(host, username=user, password=password)

        print("Relance de l'entrainement:")
        client.exec_command("nohup /home/aza/The_Hive/scripts/auto_train_gnn.sh >> /home/aza/The_Hive/hive_gnn_training.log 2>&1 &")

        time.sleep(2)

        print("Nouveaux logs d'erreur (hive_gnn_training.log):")
        stdin, stdout, stderr = client.exec_command("tail -n 30 /home/aza/The_Hive/hive_gnn_training.log")
        print(stdout.read().decode())

    except Exception as e:
        print(f"Erreur SSH : {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()