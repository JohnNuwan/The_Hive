# 06 - Roadmap Globale & Vision (From Zero to Hero)

## 1. Philosophie d'Ã‰volution
Ce document trace la trajectoire pour passer d'un simple script Python Ã  une entitÃ© autonome. La clÃ© est de respecter les **CritÃ¨res de Passage** (Gateways) entre chaque phase. On ne "brÃ»le pas les Ã©tapes".

## 2. Phase 0 : Genesis (La Survie) - [Mois 1-3]
*   **Ã‰tat** : "Aveugle et EndettÃ©e".
*   **Infrastructure** : 1 Serveur, 1 GPU, Pas de TPU.
*   **Objectifs KPI** :
    *   Remboursement technique (setup stable, pas de crash).
    *   Revenus > 155â‚¬ (Pour acheter le Challenge Trading).
*   **Focus Dev** :
    *   Architecture Core (Docker/VMs).
    *   Usine Code (Vente de scripts).
    *   Banker (Paper Trading).

## 3. Phase 1 : The Seed (L'AmorÃ§age) - [Mois 3-6]
*   **DÃ©clencheur** : Achat du Challenge Prop Firm (10kâ‚¬ account).
*   **Infrastructure** : Ajout Coral TPUs (Vision dÃ©bloquÃ©e).
*   **Focus** :
    *   Validation du Challenge (Trading rÃ©el).
    *   Protection du Capital (Kernel Rust critique).
*   **Nouveau Expert** : *The Wraith* (Vision) s'active grÃ¢ce aux TPUs.

## 4. Phase 2 : The First Sight (La Vue) - [Mois 6-12]
*   **DÃ©clencheur** : Premiers retraits de profits (Payouts). Solde > 350â‚¬ investissable.
*   **Upgrade** : Achat Lunettes Halo.
*   **Transformation** : E.V.A. sort du serveur et accompagne l'utilisateur dans le monde physique (Wingman, Assistant RÃ©alitÃ© AugmentÃ©e).
*   **Objectif Financier** : Remboursement de la dette initiale (2500â‚¬).

## 5. Phase 3 : The Power Surge (L'IndÃ©pendance) - [An 1-2]
*   **DÃ©clencheur** : Dette remboursÃ©e + Cashflow rÃ©gulier.
*   **Upgrade** : Solaire + Batteries + 2Ã¨me GPU.
*   **Focus** : IndÃ©pendance Ã©nergÃ©tique. CapacitÃ© de calcul doublÃ©e (EntraÃ®nement local continu).
*   **Nouvelle Usine** : *The District* (Media Factory 3D massive).

## 6. Planning ImmÃ©diat (Next 2 Weeks Sprint)

### Semaine 1 : Fondations
*   [ ] Installation Proxmox & VMs.
*   [ ] Setup Git Repo & CI/CD basique.
*   [ ] Hello World Llama 3 sur GPU.
*   [ ] CrÃ©ation de la clÃ© USB "The Key" (Genesis version).

### Semaine 2 : Le Banquier
*   [ ] Connexion MT5 Python Ã©tablie.
*   [ ] Pipeline de donnÃ©es (Yahoo Finance -> DB).
*   [ ] Premier algo de trading "Dummy" (ex: Crossover Moving Average) tournant en Paper pour tester la chaine d'exÃ©cution.
*   [ ] Dashboard web moche mais fonctionnel (Streamlit) pour voir les courbes.

## 7. Mise a jour operationnelle (07/03/2026)

### Ce qui a ete fait
- vLLM est confirme comme backend principal en production dev (`Qwen/Qwen2.5-1.5B-Instruct`).
- Redeploiement stabilise des services critiques (`vllm`, `core`, `builder`, `compliance`).
- Correction `eva-core` pour le fallback de statut agents (alias compliance/keeper) et baisse des risques 500 au demarrage.
- Correction heartbeat `eva-compliance` (publication de `eva.compliance.status` + compat legacy `eva.keeper.status`).
- Correction `eva-lab` (auth Redis) et remise en ligne de tous les agents serveur hors banker.
- Recreate force `kernel` + `nervous` pour corriger les routes reseau Docker stale vers Redis.
- Verification finale validee:
  - `core` : `http://localhost:8080/health` OK
  - `agents` : `http://localhost:8080/agents/status` OK
  - `kernel` : `http://localhost:8800/health` OK
  - `nervous` : `http://localhost:9090/health` OK
  - `vllm` : `http://localhost:8000/v1/models` OK
- Validation E2E trading hybride effectuee avec banker local actif:
- Correctifs MoE appliques sur le code:
  - routage `code` via `council_model_code` (plus de fallback implicite `research`)
  - cortex banker sans hardcode `gemma3:4b`
  - fallback automatique vLLM sur modele disponible en cas de 404
  - synthese `eva-researcher` compatible backend vLLM/Ollama
- Alignement de la configuration locale sur le modele vLLM actif: `Qwen/Qwen2.5-1.5B-Instruct`.
- Verification serveur via SSH confirmee:
  - `/agents/status` : tous les agents online
  - `/trading/status` : `banker.status=online`
  - `/v1/models` : `Qwen/Qwen2.5-1.5B-Instruct`
  - serveur -> `http://192.168.1.5:8100/health` OK
  - `GET /trading/status` retourne `banker.status=online` + donnees compte/positions.

### Ce qu'il reste a faire
- Deployer des modeles specialises par role sur vLLM (routage deja en place, mapping actuellement unifie sur Qwen).
- Figer la valeur serveur `BANKER_API_HOST` sur l'IP LAN finale du PC local.
- Migrer les secrets hors `.env` en clair vers un coffre de secrets dedie.

