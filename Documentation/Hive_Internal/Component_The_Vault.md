# The Hive Component: THE VAULT (Le Coffre-Fort)

## 1. Identité & Rôle
*   **Nom** : The Vault.
*   **Mission** : Gestion des secrets, Signatures Cryptographiques, "Cold Storage" des actifs.
*   **Nature** : Hardware Security Module (HSM).

## 2. Architecture Matérielle
*   **Device** : YubiKey 5 NFC ou Nitrokey HSM.
*   **Interface** : Port USB interne (sur la carte mère) pour éviter l'arrachement physique.

## 3. Gestion des Secrets (Key Management)
Les clés privées ne sont **JAMAIS** stockées sur le disque dur (même chiffrées).

### A. Clés SSH / GPG
*   L'accès Root au serveur nécessite la présence physique de la clé (Challenge-Response).
*   Signature des Commits Git : Tous les commits (surtout ceux du Kernel) sont signés par la clé GPG dans le HSM.

### B. Clés Trading (API Keys)
*   Les API Keys (Binance, FTMO) sont stockées dans le *Secure Element* si possible, ou chiffrées sur disque avec une clé maître qui, elle, est dans le HSM.
*   Au boot, *The Vault* déchiffre les configs en RAM (Tmpfs). Si on retire la clé USB, le serveur "oublie" comment trader au prochain reboot.

### C. Wallet Crypto (Cold)
*   Le "Trésor" (Bénéfices long terme) est sur une adresse dont la clé privée est générée par le HSM (ou un Ledger dédié caché ailleurs).
*   EVA peut voir le solde (Public Key), mais ne peut pas dépenser (Private Key).
*   Pour dépenser, l'Admin doit taper un PIN physique sur le clavier.

## 4. Workflow de Validation
Quand *The Banker* veut retirer des fonds :
1.  Banker génère la transaction.
2.  Banker envoie la Tx au Vault Service.
3.  Vault Service affiche sur l'écran (si existant) ou envoie notif Admin : "Signer Tx de 1000$ vers IBAN X ?".
4.  Admin valide physiquement/biométriquement.
5.  HSM signe.

## 5. Roadmap Dév
*   **Step 1** : Configuration GPG sur YubiKey.
*   **Step 2** : Script Python `vault_unlock.py` qui utilise `gpg-agent` pour déchiffrer les fichiers `.env` sensibles au démarrage.
