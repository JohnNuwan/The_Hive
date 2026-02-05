import time
import requests
import logging
import os

# Configuration (Peut être surchargée par .env)
WATCHDOG_URL = os.getenv("WATCHDOG_URL", "http://10.0.1.150/ping")
INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [WATCHDOG] - %(levelname)s - %(message)s'
)

def send_heartbeat():
    """Envoie un ping à l'ESP32 pour réarmer son timer de sécurité."""
    logging.info(f"Démarrage du service Heartbeat vers {WATCHDOG_URL}...")
    
    while True:
        try:
            # On simule une vérification de santé locale avant d'envoyer le ping
            # Si le script heartbeat tourne, c'est que l'OS n'est pas freezé.
            response = requests.post(WATCHDOG_URL, timeout=5)
            if response.status_code == 200:
                logging.debug("Heartbeat envoyé avec succès ✅")
            else:
                logging.warning(f"Réponse Watchdog anormale : {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Échec de connexion au Watchdog ESP32 : {e}")
            logging.info("Le système continue de fonctionner, mais le Watchdog pourrait trigger un reset sil ne reçoit rien.")
            
        time.sleep(INTERVAL)

if __name__ == "__main__":
    send_heartbeat()
