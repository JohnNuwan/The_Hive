import os
import paramiko
from pathlib import Path


def main():
    host = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
    user = os.getenv("HIVE_SSH_USER", "aza")
    password = os.getenv("HIVE_SSH_PASSWORD")
    sudo_password = os.getenv("HIVE_SUDO_PASSWORD", password)

    if not password:
        raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

    print("Connexion au serveur Proxmox...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(host, username=user, password=password)

        print("Televersement du script d'ingestion...")
        sftp = client.open_sftp()
        local_script = os.path.join(os.path.dirname(__file__), "ingest_biblio.py")
        remote_script = "/home/aza/The_Hive/scripts/ingest_biblio.py"
        sftp.put(local_script, remote_script)
        sftp.close()

        print("Verification du depot Biblio_IA...")
        stdin, stdout, stderr = client.exec_command(
            "if [ ! -d '/home/aza/Biblio_IA' ]; then git clone https://github.com/JohnNuwan/Biblio_IA.git /home/aza/Biblio_IA; else cd /home/aza/Biblio_IA && git pull; fi"
        )

        print("Execution de l'ingestion dans Neo4j / Mem0...")

        setup_cmds = [
            f"echo '{sudo_password}' | sudo -S apt-get update",
            f"echo '{sudo_password}' | sudo -S apt-get install -y python3-venv",
            "python3 -m venv /home/aza/biblio_env",
            "/home/aza/biblio_env/bin/pip install tqdm mem0 neo4j aiohttp pydantic qdrant-client",
            "export NEO4J_URI=bolt://localhost:7687",
            "export NEO4J_USER=neo4j",
            "export NEO4J_PASSWORD=${NEO4J_PASSWORD:-devpassword}",
            "export QDRANT_HOST=localhost",
            "export OLLAMA_HOST=localhost",
            "/home/aza/biblio_env/bin/python /home/aza/The_Hive/scripts/ingest_biblio.py",
        ]

        command = " && ".join(setup_cmds)
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