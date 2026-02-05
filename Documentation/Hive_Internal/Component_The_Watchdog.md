# The Hive Component: THE WATCHDOG (Le Chien de Garde)

## 1. Identité & Rôle
*   **Nom** : The Watchdog (Deadman Switch).
*   **Mission** : Redémarrage physique en cas de plantage logiciel total (Kernel Panic, Freeze).
*   **Nature** : Hardware Externe + Micro-contrôleur.

## 2. Architecture Matérielle
*   **Micro-contrôleur** : ESP32 (Wi-Fi) ou Arduino Nano (USB/Serial) ou Raspberry Pi Zero.
    *   *Choix Genesis* : ESP32 alimenté par USB externe (Chargeur téléphone), PAS par le serveur (indépendance électrique).
*   **Connexion** :
    *   Pin GPIO ESP32 -> Optocoupleur -> Pins `RESET_SW` de la carte mère du serveur.
    *   Connexion Wi-Fi au réseau local.

## 3. Protocole "Heartbeat"
Une sécurité active "Fail-Safe".

1.  **L'Émetteur (Sur le Serveur)** :
    *   Script Python simple (`heartbeat.py`) lancé par Systemd.
    *   Toutes les 60 secondes, il envoie une requête HTTP `POST /ping` à l'IP de l'ESP32.
    *   Condition : Le script ne s'exécute que si les services critiques (Docker, SSH) répondent.

2.  **Le Récepteur (L'ESP32)** :
    *   Boucle infinie : `LastPingTime = Now()`.
    *   Timer : Si `Now() - LastPingTime > 180 secondes` (3 minutes de silence) :
        *   Log "Watchdog Triggered".
        *   Envoi impulsion LOW (500ms) sur le GPIO Reset.
        *   Attend 10 minutes (Temps de boot) avant de réarmer la surveillance.

## 4. Sécurité Anti-Reboot Infini
*   L'ESP32 garde en mémoire (EEPROM) le nombre de reboots forcés consécutifs.
*   Si > 3 reboots en 1 heure -> STOP. Envoie notification d'urgence (SMS/Email via son propre Wi-Fi) "CRITICAL FAILURE: MANUAL INTERVENTION REQUIRED".

## 5. Roadmap Dév
*   **Step 1** : Code C++ (Arduino) pour l'ESP32 (Serveur Web simple).
*   **Step 2** : Câblage sur breadboard et test avec une LED (simulant le Reset).
*   **Step 3** : Installation physique dans le boîtier.
