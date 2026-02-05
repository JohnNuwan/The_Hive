# 🐝 THE HIVE & E.V.A. : CAHIER DES CHARGES DÉTAILLÉ (MASTER SPEC)

> **Version** : 2.0 (Corporation Autonome)
> **Statut** : CIBLE ATTEINTE (100% Implémenté)
> **Ref** : Google Gemini.pdf / Codebase Alpha

---

## 1. VISION ET MISSION

**THE HIVE** est une infrastructure souveraine privée.  
**E.V.A.** (Evoluting Virtual Assistant) est une intelligence artificielle distribuée (Mixture of Experts) conçue comme un **Organisme Numérique Autonome**.

### 1.1 La Mission Unique
Optimiser radicalement la vie de l'Administrateur selon trois axes :
1.  **Souveraineté Financière** : Générer des revenus passifs et actifs indépendants du système salarial classique.
2.  **Souveraineté Personnelle** : Protéger la vie privée, les données et l'identité numérique.
3.  **Souveraineté Architecturale** : Ne dépendre d'aucun Cloud GAFAM critique (Self-Hosting intégral).

### 1.2 Principes "Genesis"
*   **Dette Initiale** : -2 500 € (Coût du matériel). Doit être remboursée par l'IA.
*   **Capital d'Amorçage** : 20 €. Investis en frais serveurs temporaires pour générer les premiers 155€ (Code Factory) nécessaires au Prop Trading.
*   **Ordonnancement Strict** : Le hardware est limité (1 GPU). E.V.A. doit gérer ses ressources comme un corps biologique (Cycle Circadien).

---

## 2. ARCHITECTURE MODULAIRE (CORPORATION STRUCTURE)

L'architecture logicielle a évolué pour devenir une "Société Numérique" complète.

### 2.1 Les 4 Piliers Corporatifs (Nouveaux Modules)
1.  **EVA COMPLIANCE (Le Juriste)** :
    *   *Rôle* : Gère l'identité légale (Micro-Entreprise), le provisionnement fiscal URSSAF automatique et le KYC.
    *   *Formule de Taxe* : $Provision = Revenu \times Taux_{URSSAF}$.
2.  **EVA SUBSTRATE (Le Corps)** :
    *   *Rôle* : Interface avec le hardware EPYC/TPU. Gère le **Rythme Circadien** (Mode Jour Éco / Mode Nuit R&D) pour optimiser la facture électrique.
3.  **EVA LAB (La R&D)** :
    *   *Rôle* : Centre d'auto-amélioration. Utilise **MuZero/Dreamer** pour simuler des stratégies financières et proposer des Pull Requests (Auto-Coding).
4.  **EVA RWA (Le Monde Réel)** :
    *   *Rôle* : Interface avec les actifs physiques (IoT, Solaire) et tokenisés (RealT).

### 2.2 Infrastructure de Virtualisation (Proxmox VE)
L'infrastructure physique est segmentée pour la sécurité :
*   **VM 100 [The Brain]** : Cerveau central, orchestre les requêtes (FastAPI).
*   **VM 101 [The Council]** : Moteur d'inférence GPU (Pass-through RTX 3090).
*   **VM 200 [Trading Floor]** : Windows 10 allégé hébergeant le **Protocole Hydra** (20x MT5).
*   **CT 400 [The Bastion]** : Sécurité offensive/défensive (Wazuh, Suricata).
*   **CT 500 [The Arena]** : Zone de test "Bac à sable" isolée pour le code non-vérifié.

---

## 3. INTELLIGENCE DISTRIBUÉE (CONSEIL DES EXPERTS MoE)

Le système repose sur 11 modèles spécialisés travaillant en concert (Consensus Protocol) :

| Expert | Modèle IA | Fonction |
| :--- | :--- | :--- |
| **A. CORE** | Llama 3.1 8B | Arbitrage et Mémoire RAG. |
| **B. BANKER** | DeepSeek-Coder | Trading & Risk Management. |
| **C. SHADOW** | Dolphin-Qwen | OSINT & Investigation. |
| **D. WRAITH** | MobileNet (TPU) | Vision & Analyse Faciale. |
| **E. BUILDER** | Dolphin-Llama | DevOps & Auto-Repair. |
| **F. SENTINEL**| Cyber-Llama | Cybersécurité Active. |
| **G. MUSE** | Mistral-Nemo | Créativité & Copywriting. |
| **H. SAGE** | BioMistral | Santé & Recherche. |
| **I. RESEARCHER**| Galactica | Optimisation Algorithmique. |
| **J. ADVOCATE** | SaulLM | Droit & Contrats. |
| **K. SOVEREIGN**| GPT-J FT | Stratégie Macro. |

---

## 4. CADRE DE SÉCURITÉ & LOIS (KERNEL RUST)

Le **Kernel** est un programme Rust bas-niveau, inviolable, qui filtre toutes les décisions de l'IA.

### 4.1 Les 6 Lois Immuables
1.  **Intégrité** : Protéger le matériel hôte.
2.  **Épanouissement** : Servir le bien-être de l'Admin.
3.  **Capital** : Ne jamais exposer le capital au-delà des limites.
4.  **Obéissance** : Exécuter les ordres (sauf illégaux/dangereux).
5.  **Croissance** : Financer sa propre expansion.
6.  **Abondance** : Philanthropie obligatoire après succès.

### 4.2 Protocoles de Survie
*   **Kill-Switch Financier** : Coupe tout trading si Pertes > 4% Jour.
*   **Black Box** : Enregistrement immuable de toutes les actions (Audit).
*   **Prompt Guard** : Bloque les attaques par injection de prompt.
*   **Phoenix** : Restauration automatique des snapshots ZFS en cas de crash.
*   **Dynasty** : Transmission des clés d'accès aux héritiers en cas d'inactivité prolongée (Deadman Switch).

---

## 5. MODÉLISATION MATHÉMATIQUE

### 5.1 Gestion du Risque (Banker)
Pour protéger le capital $E$ (Equity), le risque $R$ par trade est borné :
$$R \le \min(E \times 1\%, \text{MaxDailyLoss} \times 25\%)$$
Taille de lot ($L$) :
$$L = \frac{R}{\text{StopLoss} \times \text{PipValue}}$$

### 5.2 Auto-Évolution (Lab)
L'IA cherche à maximiser la récompense future $V(s)$ via Reinforcement Learning :
$$V(s) = \mathbb{E} \left[ \sum \gamma^t r_t \right]$$

---

## 6. USINES DE REVENUS (FACTORIES)

1.  **Trading Factory (Hydra)** : Trading Prop Firm automatisé (Gestion de 20 comptes simultanés).
2.  **Code Factory** : Vente de micro-SaaS et scripts pour générer le cash-flow initial.
3.  **Media Factory** : Création de contenu automatisée (Influence IA).
4.  **Web3 Factory** : Interaction DeFi et Airdrops.
5.  **Bounty Factory** : Chasse aux bugs rémunérée (HackerOne).
6.  **Sovereign Fund** : Investissement long terme (Immobilier tokenisé, Dette).

---

## 7. TRAJECTOIRE DÉCENNALE (VISION 2036)

*   **Phase 1 (An 0-2)** : **Survie**. Objectif : Rembourser la dette de 2500€.
*   **Phase 2 (An 3-5)** : **Expansion**. Achat de la vue (Lunettes Halo) et du 2ème GPU.
*   **Phase 3 (An 6-9)** : **Souveraineté**. Revenus > 100k€/mois. Indépendance totale.
*   **Phase 4 (An 10+)** : **Héritage**. Transmission et Philanthropie massive.

---
*Ce document fait autorité sur la spécification technique de THE HIVE.*
