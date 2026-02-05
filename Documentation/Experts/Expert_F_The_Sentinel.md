# Expert F: THE SENTINEL (La Cybersécurité)

## 1. Identité & Rôle
*   **Nom** : The Sentinel
*   **Rôle** : SOC (Security Operations Center), Firewall Humain/IA.
*   **Type** : Permanent (Temps Réel).

## 2. Architecture Technique
*   **Moteur** : **Wazuh** (SIEM/XDR Open Source).
*   **IA** : `Cyber-Llama` (Fine-tuné sur les logs sécurité).
*   **Hardware** : Accélération Packet Inspection via TPU (Suricata).

## 3. Capacités Défensives
*   **IP Ban** : Bannissement automatique (iptables/Fail2Ban) après 3 tentatives SSH échouées.
*   **Anomaly Detection** : "Pourquoi le processus Python du Trading essaie de contacter une IP en Russie ?" -> KILL & ALERT.

## 4. Capacités Offensives (The Red Teamer)
*   *Zone* : Uniquement dans CT 500 (Arena).
*   *Mission* : Lancer des scans vulnérabilité (OpenVAS/Nuclei) contre les services internes pour vérifier qu'on n'a pas laissé de faille.

## 5. Roadmap Dév
*   **J1** : Installation Agent Wazuh sur VM 100 et VM 200.
*   **J2** : Configurer Fail2Ban sur le SSH Gateway.
*   **J3** : Script de notification Discord/Nexus en cas d'alerte critique.
