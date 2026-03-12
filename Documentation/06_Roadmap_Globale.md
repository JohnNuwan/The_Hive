# 06 - Roadmap Globale & Vision (From Zero to Hero)

## 1. Philosophie d'Ãƒâ€°volution
Ce document trace la trajectoire pour passer d'un simple script Python ÃƒÂ  une entitÃƒÂ© autonome. La clÃƒÂ© est de respecter les **CritÃƒÂ¨res de Passage** (Gateways) entre chaque phase. On ne "brÃƒÂ»le pas les ÃƒÂ©tapes".

## 2. Phase 0 : Genesis (La Survie) - [Mois 1-3]
*   **Ãƒâ€°tat** : "Aveugle et EndettÃƒÂ©e".
*   **Infrastructure** : 1 Serveur, 1 GPU, Pas de TPU.
*   **Objectifs KPI** :
    *   Remboursement technique (setup stable, pas de crash).
    *   Revenus > 155Ã¢â€šÂ¬ (Pour acheter le Challenge Trading).
*   **Focus Dev** :
    *   Architecture Core (Docker/VMs).
    *   Usine Code (Vente de scripts).
    *   Banker (Paper Trading).

## 3. Phase 1 : The Seed (L'AmorÃƒÂ§age) - [Mois 3-6]
*   **DÃƒÂ©clencheur** : Achat du Challenge Prop Firm (10kÃ¢â€šÂ¬ account).
*   **Infrastructure** : Ajout Coral TPUs (Vision dÃƒÂ©bloquÃƒÂ©e).
*   **Focus** :
    *   Validation du Challenge (Trading rÃƒÂ©el).
    *   Protection du Capital (Kernel Rust critique).
*   **Nouveau Expert** : *The Wraith* (Vision) s'active grÃƒÂ¢ce aux TPUs.

## 4. Phase 2 : The First Sight (La Vue) - [Mois 6-12]
*   **DÃƒÂ©clencheur** : Premiers retraits de profits (Payouts). Solde > 350Ã¢â€šÂ¬ investissable.
*   **Upgrade** : Achat Lunettes Halo.
*   **Transformation** : E.V.A. sort du serveur et accompagne l'utilisateur dans le monde physique (Wingman, Assistant RÃƒÂ©alitÃƒÂ© AugmentÃƒÂ©e).
*   **Objectif Financier** : Remboursement de la dette initiale (2500Ã¢â€šÂ¬).

## 5. Phase 3 : The Power Surge (L'IndÃƒÂ©pendance) - [An 1-2]
*   **DÃƒÂ©clencheur** : Dette remboursÃƒÂ©e + Cashflow rÃƒÂ©gulier.
*   **Upgrade** : Solaire + Batteries + 2ÃƒÂ¨me GPU.
*   **Focus** : IndÃƒÂ©pendance ÃƒÂ©nergÃƒÂ©tique. CapacitÃƒÂ© de calcul doublÃƒÂ©e (EntraÃƒÂ®nement local continu).
*   **Nouvelle Usine** : *The District* (Media Factory 3D massive).

## 6. Planning ImmÃƒÂ©diat (Next 2 Weeks Sprint)

### Semaine 1 : Fondations
*   [ ] Installation Proxmox & VMs.
*   [ ] Setup Git Repo & CI/CD basique.
*   [ ] Hello World Llama 3 sur GPU.
*   [ ] CrÃƒÂ©ation de la clÃƒÂ© USB "The Key" (Genesis version).

### Semaine 2 : Le Banquier
*   [ ] Connexion MT5 Python ÃƒÂ©tablie.
*   [ ] Pipeline de donnÃƒÂ©es (Yahoo Finance -> DB).
*   [ ] Premier algo de trading "Dummy" (ex: Crossover Moving Average) tournant en Paper pour tester la chaine d'exÃƒÂ©cution.
*   [ ] Dashboard web moche mais fonctionnel (Streamlit) pour voir les courbes.

## 7. Mise a jour operationnelle (08/03/2026)

### Ce qui a ete fait
- vLLM reste le backend principal du serveur de dev.
- Le mode hybride trading est valide: serveur OK, `banker.bat` local OK, `MT5` reel cote poste local.
- Le banker ne depend plus d'une liste d'actifs hardcodee: l'univers MT5 est decouvert dynamiquement et classe en `forex`, `cfd`, `crypto`.
- Les filtres week-end/session et l'anti-spam Telegram sont corriges.
- Les tests banker modifies passent localement.
- Le `GNN` multi-timeframe est entraine a `500` epochs cumulees au total sur GPU.
- Le pipeline nocturne EVA Lab a ete fiabilise: sync Proxmox, resume JSON persistant, packaging `eva-lab` corrige.
- La pile runtime JAX CUDA validee sur serveur est maintenant: `jax 0.4.23`, `jaxlib 0.4.23+cuda11.cudnn86`, `haiku 0.0.11`, backend `gpu`.
- MuZero a ete corrige cote code et tourne a present sur GPU sur le serveur de dev.
- `eva-builder` a ete renforce pendant l'attente trading: `CyberForge` partage entre API et factory, pipeline BMAD avec validation automatique Python et historique coherent.
- Les tests `eva-builder` passes localement montent a `16 passed`.
- `eva-builder` sait maintenant synchroniser et exploiter un catalogue d'APIs publiques depuis `public-apis/public-apis` pour enrichir les prompts produit et SaaS.
- `eva-builder` expose aussi une passerelle de mutation securisee avec `dry-run` par defaut et activation explicite par variable d'environnement.
- `eva-builder` expose aussi une passerelle de deploiement structuree, en `dry-run` par defaut, pour les cibles `local` et `proxmox`.
- Le `banker` n'ouvre plus de nouvelles positions tant qu'EVA Lab ne fournit pas un `champion` live valide.
- Le mode `research-first` est maintenant privilegie: entrainement massif, notifications Telegram des meilleurs candidats, puis seulement bascule vers le live.
- `eva-nexus` est maintenant recable proprement sur Builder: proxy Docker ajoute, Muse repasse par le proxy local, liens Grafana alignes sur l'hote courant, et cockpit Builder disponible dans Enterprise.
- Le frontend Nexus passe maintenant `npm run lint` puis `npm run build` localement.

### Ce qui est en cours
- `MuZero scalp` est en execution sur GPU.
- `MuZero intraday` et `MuZero swing` attendent la fin du premier horizon.
- `DreamerV3` demarrera automatiquement apres la sequence MuZero.
- La partie trading est donc surtout en attente active de fin d'entrainement et de verification d'artefacts.

### Ce qu'il reste a faire
- Finaliser les runs `MuZero` de recherche, puis verifier les checkpoints/rapports generes sur un echantillon suffisant.
- Rendre `eva-trainer` nativement compatible JAX CUDA sans patch runtime a chaque lancement.
- Injecter uniquement les modeles promus par la gate dans la boucle de decision/execution live du banker.
- Etendre l'historique et l'univers au-dela des `6` symboles actuels.
- Connecter ensuite les exchanges crypto (`Binance`, `Kraken`, `Coinbase`) au meme pipeline.
- Finaliser la boucle ADN/evolution/champion-challenger sur resultats reels.
- Migrer les secrets hors `.env` en clair vers un coffre dedie.
- Basculer les flux Builder Nexus de `dry-run` vers deploiement/mutation live uniquement apres validation serveur complete.

### Travail recommande pendant l'attente
- Priorite 1 deja engagee: `eva-builder`, avec consolidation des services et de la boucle de validation locale pendant que le training tourne.
- Priorite 2 deja engagee: `eva-nexus`, avec recablage des proxies et cockpit Builder dans Enterprise.
- Priorite 3: `RLM` seulement en appui, pas comme chantier principal, car sa boucle existe deja.

