# The Hive Component: THE KEEPER (Le Gardien des Ressources)

## 1. Identité & Rôle
*   **Nom** : The Keeper
*   **Mission** : Optimisation des Ressources (VRAM, CPU, Énergie) et Maintenance.
*   **Nature** : Service Système (Rust Daemon). Il n'est PAS une IA, c'est un automatisme déterministe.

## 2. Architecture Technique
*   **Langage** : Rust (Zero-cost abstractions, Thread-safety).
*   **Privilèges** : Root (Host Proxmox/Linux).
*   **Dépendances** :
    *   `nvml-wrapper` (NVIDIA Management Library) pour lire la VRAM/Temp.
    *   `sysinfo` pour CPU/RAM.
    *   `docker-api` pour contrôler les conteneurs.

## 3. Algorithme "The Scheduler" (Ordonnancement de Pénurie)
Le GPU 3090 est une ressource critique unique (Mutex global).

### Matrice de Priorité
| Rang | Service | Condition d'Activation | Action Keeper |
| :--- | :--- | :--- | :--- |
| **1 (Max)** | **The Banker** | Marché Ouvert (Lun-Ven 8h-23h) | Force Pause sur Muse/Research. Alloue 100% CPU Pinned. |
| **2** | **EVA Core** | Interaction Humaine (Voice/Chat) | Préemption immédiate (< 200ms). |
| **3** | **The Sentinel** | Scan Périodique | Background. |
| **4** | **The Muse** | Batch Nuit (23h-7h) | Se lance uniquement si Load Average < 0.5. |

### Logique de "Freeze"
Quand une demande Priorité 2 arrive (ex: L'utilisateur parle) alors que Priorité 4 tourne (Génération Image) :
1.  Keeper envoie signal `SIGSTOP` au PID de Stable Diffusion.
2.  Keeper swap la VRAM de SD vers RAM/NVMe (via `cudaMallocManaged` ou swap système agressif).
3.  Keeper lance Llama-3.
4.  Une fois fini, Keeper désalloue Llama-3 et envoie `SIGCONT` à SD.

## 4. Eco-Mode (L'Énergie)
*   **Monitoring** : Lecture de la consommation Watts en temps réel.
*   **Action** :
    *   Si pas de tâche active -> Undervolt GPU (-200MHz, Power Limit 60%).
    *   Si Tâche Lourde -> Power Limit 100%.

## 5. Roadmap Dév
*   **Step 1** : "Hello World" Rust qui lit la température GPU et l'écrit dans un log JSON.
*   **Step 2** : Implémentation du système de Priority Queue.
*   **Step 3** : Intégration API Proxmox pour tuer/lancer des VMs.
