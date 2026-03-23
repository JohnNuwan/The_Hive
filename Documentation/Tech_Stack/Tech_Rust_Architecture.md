# Tech Architect: RUST (The Shield)

##  Rôle
Rust est utilisé pour la **Sécurité Critique**, la **Haute Performance**, et la **Stabilité**. C'est le "Squelette Indestructible".

##  Composants Rust

### 1. The Kernel (`eva-kernel`)
*   **Type** : Binaire compilé (Standalone Daemon).
*   **Responsabilité** : Faire respecter la Constitution.
*   **Architecture** : Actor Model (Tokio).
*   **Modules** :
    *   `watchdog_financial`: Polling MT5 (via Shared Memory ou ZeroMQ sécurisé).
    *   `watchdog_thermal`: Polling `nvml`.
    *   `vault_client`: Interface avec la YubiKey.

### 2. The Sentinel Engine (`eva-sentinel`)
*   **Type** : Service réseau.
*   **Responsabilité** : Filtrage de paquets (eBPF ou Packet Capture).
*   **Pourquoi Rust -** : Pour analyser 1Gbps de trafic sans latence GC.

### 3. Python Bindings (`py-eva-rust`)
*   Pour les fonctions cryptographiques ou mathématiques lourdes, on expose des fonctions Rust à Python via **PyO3/Maturin**.
    *   Ex: `eva_rust.verify_transaction_signature(tx_json) -> bool`

##  Règles de Dév Rust
*   **Safety** : `unsafe {}` est INTERDIT sauf justification absolue (FFI).
*   **Error Handling** : Pas de `unwrap()`. Utilisation de `Result<T, E>` avec propagation propre (`-`).
*   **Concurrency** : Utilisation de `Arc<Mutex<T>>` ou Channels (`mpsc`) pour la communication inter-thread.

##  Interaction
*   Rust est le "Maître". Si le processus Python crash, Rust le relance. Si Rust crash, le Watchdog Hardware reboot le PC.
