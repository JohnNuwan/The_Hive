#!/usr/bin/env python3
"""THE HIVE - Push History Data
Ce script synchronise le dossier local `data/history/` vers le serveur Proxmox
pour s'assurer que les modeles (MuZero, DreamerV3) disposent des donnees fraiches
lors de l'entrainement nocturne.
"""

import os
import paramiko
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PushHistory")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    host = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
    user = os.getenv("HIVE_SSH_USER", "aza")
    password = os.getenv("HIVE_SSH_PASSWORD")

    if not password:
        logger.error("Variable d'environnement HIVE_SSH_PASSWORD manquante.")
        return

    local_dir = Path(__file__).resolve().parents[1] / "data" / "history"
    
    # Destination probable sur le serveur selon l'architecture The Hive
    remote_dir = "/app/eva-lab/data/history"
    fallback_remote_dir = "/home/aza/The_Hive/data/history"

    if not local_dir.exists():
        logger.warning(f"Le dossier local {local_dir} n'existe pas. Rien a envoyer.")
        return

    csv_files = list(local_dir.glob("*.csv"))
    if not csv_files:
        logger.info("Aucun fichier CSV trouve dans data/history. Rien a envoyer.")
        return

    logger.info(f"Connexion au serveur {host}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(host, username=user, password=password, timeout=10)
        sftp = client.open_sftp()
        
        # Test de l'existance du dossier principal
        target_dir = remote_dir
        try:
            sftp.stat(target_dir)
        except IOError:
            logger.info(f"Le dossier {target_dir} n'existe pas, tentative avec {fallback_remote_dir}")
            target_dir = fallback_remote_dir
            try:
                sftp.stat(target_dir)
            except IOError:
                logger.error(f"Le dossier de destination {target_dir} n'existe pas sur le serveur non plus.")
                return

        logger.info(f"Televersement de {len(csv_files)} fichiers CSV vers {target_dir}...")
        
        for file_path in csv_files:
            remote_path = f"{target_dir}/{file_path.name}"
            logger.info(f" -> Envoi de {file_path.name}...")
            sftp.put(str(file_path), remote_path)
            
        logger.info("Synchronisation terminee avec succes !")
        sftp.close()
        
    except Exception as e:
        logger.error(f"Erreur de connexion ou de transfert: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
