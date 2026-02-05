# The Hive Component: THE TABLET / THE KEY (La Pierre de Loi)

## 1. Identité & Rôle
*   **Nom** : The Tablet (ou The Key en Phase Genesis).
*   **Mission** : Garantir l'intégrité du "Code Légal" (Constitution) et du Kernel. Empêcher l'IA de se réécrire.
*   **Nature** : Stockage Physique en Lecture Seule (Write-Protected).

## 2. Architecture Matérielle (Phase Genesis)
*   **Support** : Clé USB industrielle (SLC) de petite capacité (4Go suffisent).
*   **Spécificité** : Interrupteur physique sur le boîtier de la clé ("Lock Switch").
*   **Montage** : `/mnt/THE_LAW` en Read-Only (`ro,noexec` sauf pour le Kernel).

## 3. Contenu de la Tablette
1.  **Lois.toml** : Le fichier configuration des règles d'Asimov.
    ```toml
    [law_0]
    description = "Intégrité Système"
    max_gpu_temp = 90
    
    [law_2]
    description = "Protection Capital"
    max_daily_loss_percent = 4.0
    ```
2.  **Kernel Checksums** : `sha512sum` de tous les binaires critiques (Watchdog, Banker Core).
3.  **Kernel Binary** : La version compilée et validée du Kernel Rust.

## 4. Séquence de Boot (Chain of Trust)
C'est la première chose que le BIOS/GRUB lance.

1.  **Mount** : Montage de la clé USB.
2.  **Verify** : Le script de boot calcule le hash du `kernel_service` présent sur le disque dur.
3.  **Compare** : Il compare ce hash avec celui stocké sur la clé USB (qui ne peut pas être modifié car switch physique activé).
4.  **Enforce** :
    *   Si Hash OK -> Systemd lance le service.
    *   Si Hash KO (Fichier corrompu ou modifié par une IA hackée) -> **KERNEL PANIC**. Le système refuse de démarrer pour protéger l'extérieur.

## 5. Procédure d'Amendement (Modifier la Loi)
Pour changer une règle (ex: Augmenter le risque à 5%) :
1.  L'Admin doit se lever physiquement.
2.  Débrancher la clé USB du serveur.
3.  Déverrouiller le switch physique.
4.  Brancher sur son laptop Admin sécurisé.
5.  Modifier `Lois.toml` et re-signer les checksums.
6.  Verrouiller le switch.
7.  Rebrancher sur serveur.
8.  Reboot.
*   *Sécurité* : Impossible à faire à distance ou par logiciel.

## 6. Roadmap Dév
*   **Step 1** : Achat clé USB avec switch.
*   **Step 2** : Script bash `verify_integrity.sh` au démarrage.
