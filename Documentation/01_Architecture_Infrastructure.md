# 01 - Architecture Infrastructure & Hardware (The Hive)

## 1. Vue d'Ensemble Senior Architect
Ce document détaille l'implémentation de la couche "Physique" et "Virtualisation" (Proxmox). Le défi principal est la **contrainte de ressources** (1 GPU, RAM limitée) en phase Genesis. L'architecture ne doit pas être un simple assemblage de VMs, mais un système **élastique** piloté par *The Keeper*.

## 2. Roadmap d'Implémentation

### Phase 0: "Genesis Hardware" (MVP)
*   **Objectif** : Stabilité maximale pour le Trading et la Survie avec un seul nœud.
*   **Actions Critiques** :
    1.  **Installation Proxmox VE** : Configuration ZFS (Redondance logicielle si disques dispo, sinon Snapshot agressif).
    2.  **Segmentation Réseau (vSwitch)** :
        *   `vmbr0` (WAN/LAN) : Accès internet filtré.
        *   `vmbr1` (Inter-VM) : Réseau privé très haut débit (VirtIO) pour la communication IA <-> Trading.
        *   `vmbr2` (DMZ/Sandboxed) : Pour *The Arena* et les tests de code.
    3.  **Configuration du Passthrough GPU** :
        *   Isoler la RTX 3090 pour la VM 100 (*The Brain*) par défaut.
        *   Préparer les scripts de "Hot-Unplug" pour basculer le GPU vers *The District* (Rendu) uniquement quand requis (Note: Délicat avec NVIDIA grand public, fallback sur rendu CPU ou reboot planifié si instable).
    4.  **The Keeper (v0.1)** : Script Bash/Rust simple qui surveille la température et tue les processus non-prioritaires si T° > 80°C.

### Phase 1: "Expansion" (Ajout TPUs)
*   **Integration Google Coral** : Déporter toute la détection d'objets (Frigate/The Wraith) vers les TPUs pour libérer le GPU 3090.
*   **Tailscale Setup** : Configuration du VPN Mesh pour accès Admin sécurisé sans port forwarding.

## 3. Détails Techniques & Pièges à Éviter (Guidance Senior)

### ⚠️ Gestion de la Mémoire (OOM Killer)
*   **Risque** : Avec 128 Go et des LLM chargés, le système peut crash.
*   **Solution** :
    *   Utiliser **ZRAM** sur les VMs Linux pour compresser la RAM.
    *   Configurer des priorités de swap strictes.
    *   *The Keeper* doit avoir un accès API Proxmox pour "Ballooner" la RAM des VMs inactives.

### ⚠️ Latence Trading
*   **Risque** : Une surcharge CPU due à l'IA ralentit l'exécution d'un ordre MT5.
*   **Solution** :
    *   Pinning CPU (CPU Affinity) : Réserver 2-4 cœurs PHYSIQUES exclusifs à la VM 200 (*Trading Floor*). Interdire à l'IA d'utiliser ces cœurs.
    *   Priorité Processus : La VM Trading doit être en priorité "Real Time" dans Proxmox.

### ⚠️ Sécurité Physique
*   **Implémentation The Watchdog** : Ne pas attendre la phase finale. Mettre en place un ESP32 basique sur les headers *Reset* de la carte mère dès le jour 1. Si le Kernel Linux freeze, le hardware doit rebooter seul en <3 min.

## 4. Spécifications des VMS (Configuration Cible)

| ID | Nom | OS | vCPU | RAM | GPU | Rôle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100** | The Brain | Ubuntu LTS | 8 | 64GB+ | RTX 3090 | Core AI, API Gateway, Llama 3 |
| **101** | The Council | Ubuntu LTS | 4 | 32GB | Partagé | Inférence secondaire (Ollama) |
| **200** | Trading Floor | Win10 LTSC | 4 (Pinned) | 8GB | VirtIO | MT5, Terminaux boursiers |
| **300+** | Workers | LXC Alpine | 1-2 | 512MB | - | Scrapers, Bots légers |
| **400** | The Bastion | Debian Hardened | 2 | 4GB | - | Wazuh, Suricata, Firewall |
| **401** | Visual Cortex | Debian | 4 | 8GB | TPU | Frigate, Traitement Vidéo |

## 5. Next Steps (Action Plan)
1.  Valider la topologie réseau (Adressage IP, VLANs).
2.  Écrire les scripts d'installation automatique (Ansible) pour la VM 100 et 200 (Reproducibility).
3.  Tester le Passthrough GPU sur le hardware réel.
