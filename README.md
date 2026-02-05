# 🐝 THE HIVE : Infrastructure IA Souveraine & Écosystème E.V.A.

![Status](https://img.shields.io/badge/Statut-Genesis-gold?style=for-the-badge)
![Tech](https://img.shields.io/badge/Stack-Python_|_Rust_|_React_|_Go-blue?style=for-the-badge)
![Security](https://img.shields.io/badge/S%C3%A9curit%C3%A9-ZFS_|_Proxmox_|_Rust_Kernel-red?style=for-the-badge)

> **"Un organisme numérique conçu pour une souveraineté financière, personnelle et architecturale absolue."**

---

## 🌟 La Vision
**THE HIVE** (La Ruche) est bien plus qu'un simple monorepo ; c'est une infrastructure privée et auto-suffisante hébergée sur Proxmox VE. Elle constitue le corps physique d'**E.V.A. (Evoluting Virtual Assistant)**, une IA avancée distribuée via une architecture **Mixture of Experts (MoE)**.

La mission d'E.V.A. est simple mais profonde : **Optimiser la vie, les finances et la sécurité de son Administrateur.**

---

## 🧠 Architecture MoE (Le Conseil des Experts)
Le système est piloté par un conseil décentralisé d'agents spécialisés, fonctionnant chacun dans des conteneurs ou des VM dédiés :

### 🏛️ Cœur & Orchestration
- **[EVA Core](src/eva-core)** : Le cerveau central utilisant LangGraph et Llama 3.1. Il gère le routage des intentions, la mémoire conversationnelle (RAG) et orchestre les agents spécialisés.
- **[The Nexus](src/eva-nexus)** : L'interface Premium (PWA). Un centre de commandement basé sur React pour le monitoring et l'interaction en temps réel.
- **[The Keeper](src/shared)** : Un agent Rust de bas niveau gérant les ressources matérielles, l'ordonnancement de la VRAM et la santé du système.

### 💰 Experts Financiers
- **[The Banker](src/eva-banker)** : Agent de trading haute performance gérant les instances MetaTrader 5 (MT5) via le **Protocole Hydra**. Gère le risque et l'exécution des ordres.
- **Web3 Factory** : Opérations DeFi automatisées, gestion de collections NFT et chasse aux airdrops.

### 🛡️ Sécurité & Intelligence
- **[The Sentinel](src/eva-sentinel)** : Agent de sécurité accéléré par matériel (Google Coral TPU). Surveille les paquets, l'intégrité du système et la défense active.
- **[The Shadow](src/eva-shadow)** : Expert OSINT et Investigation. Effectue des recherches sur le deep web, l'intelligence sur les fuites de données et le profilage de menaces.

### 🛠️ Développement & Maintenance
- **[The Builder](src/eva-builder)** : Agent DevOps pour l'auto-codage, la maintenance et **The Librarian** (documentation automatisée).
- **[The Kernel](src/eva-kernel)** : Un noyau de sécurité immuable basé sur Rust appliquant les **6 Lois d'E.V.A.**

---

## ⚖️ Les 6 Lois (Un Cadre Constitutionnel)
E.V.A. opère sous un ensemble de lois strictes et non négociables, inscrites dans le Kernel Rust :
1. **Loi 0 (Intégrité)** : Protéger le matériel hôte à tout prix.
2. **Loi 1 (Bien-être)** : Maximiser la santé et l'épanouissement de l'Administrateur avant le profit.
3. **Loi 2 (Capital)** : Protéger les actifs avec une limite de perte journalière stricte de 4%.
4. **Loi 3 (Obéissance)** : Suivre les ordres, sauf s'ils violent les Lois 0, 1 ou 2.
5. **Loi 4 (Croissance)** : Auto-préservation et mise à l'échelle autonome via les revenus générés.
6. **Loi 5 (Abondance)** : Philanthropie obligatoire une fois les dettes remboursées et l'abondance atteinte.

---

## 🚀 Démarrage Rapide

### 📋 Pré-requis
- **OS** : Proxmox VE (Recommandé) ou un hôte Linux puissant.
- **Matériel IA** : NVIDIA RTX 3090+ (pour les LLM), Google Coral TPU (pour la Vision/Sécurité).
- **Stack** : Python 3.11, Rust 1.75+, Node.js 20+, Docker.

### 🛠️ Installation
```bash
# Cloner le dépôt souverain
git clone https://github.com/JohnNuwan/The_Hive.git
cd the-hive

# Installer les dépendances des agents
pip install -e src/shared
pip install -e src/eva-core src/eva-banker src/eva-sentinel src/eva-shadow src/eva-builder

# Lancer les services d'infrastructure
docker-compose -f Documentation/Config/docker_compose.yaml up -d
```

---

## 📈 Roadmap (Phase Genesis)
- [x] **Phase 0.1** : Infrastructure de base & Routage MoE.
- [x] **Phase 0.2** : The Banker (Intégration MT5).
- [x] **Phase 0.3** : The Nexus (Interface UI/PWA).
- [x] **Phase 0.4** : Agents de Sécurité & OSINT.
- [ ] **Phase 0.5** : Exécution du premier challenge FTMO.
- [ ] **Phase 1.0** : Upgrade matériel (2ème GPU) & Déploiement Vision (Lunettes Halo).

---

## 📖 Approfondissement
Pour les spécifications complètes, la philosophie du projet et les projections décennales, reportez-vous au [**Cahier des Charges Détaillé (CDC.md)**](CDC.md).

---
*© 2026 THE HIVE - Construit pour une souveraineté absolue.*
