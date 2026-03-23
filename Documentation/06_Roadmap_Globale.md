# 06 - Roadmap Globale et Vision

## 1. Philosophie d evolution
Ce document trace la trajectoire pour passer d un simple script Python a une entite autonome. La cle est de respecter les criteres de passage entre chaque phase. On ne saute pas les etapes.

## 2. Phase 0 : Genesis (La survie) - [Mois 1-3]
- **Etat** : aveugle et endettee.
- **Infrastructure** : 1 serveur, 1 GPU, pas de TPU.
- **Objectifs KPI** :
  - remboursement technique : setup stable, pas de crash
  - revenus > 155 EUR pour acheter le challenge trading
- **Focus dev** :
  - architecture Core (Docker/VMs)
  - usine code pour la vente de scripts
  - Banker en paper trading

## 3. Phase 1 : The Seed (L amorcage) - [Mois 3-6]
- **Declencheur** : achat du challenge Prop Firm (compte 10k EUR)
- **Infrastructure** : ajout de Coral TPU pour debloquer la vision
- **Focus** :
  - validation du challenge en trading reel
  - protection du capital avec Kernel Rust critique
- **Nouvel expert** : *The Wraith* s active grace aux TPU.

## 4. Phase 2 : The First Sight (La vue) - [Mois 6-12]
- **Declencheur** : premiers retraits de profits, solde > 350 EUR investissables
- **Upgrade** : achat des lunettes Halo
- **Transformation** : EVA sort du serveur et accompagne l utilisateur dans le monde physique
- **Objectif financier** : rembourser la dette initiale de 2500 EUR

## 5. Phase 3 : The Power Surge (L independance) - [An 1-2]
- **Declencheur** : dette remboursee et cashflow regulier
- **Upgrade** : solaire, batteries et second GPU
- **Focus** : independance energetique et capacite de calcul doublee
- **Nouvelle usine** : *The District* pour la media factory 3D massive

## 6. Planning immediat (Sprint 2 semaines)

### Semaine 1 : Fondations
- [ ] Installation Proxmox et VMs
- [ ] Setup Git repo et CI/CD basique
- [ ] Hello World Llama 3 sur GPU
- [ ] Creation de la cle USB "The Key" version Genesis

### Semaine 2 : Le banquier
- [ ] Connexion MT5 Python etablie
- [ ] Pipeline de donnees Yahoo Finance vers la base
- [ ] Premier algo de trading dummy en paper pour tester la chaine d execution
- [ ] Dashboard web simple mais fonctionnel pour voir les courbes

## 7. Mise a jour operationnelle (08/03/2026)

### Ce qui a ete fait
- vLLM reste le backend principal du serveur de dev.
- Le mode hybride trading est valide: serveur OK, `banker.bat` local OK, `MT5` reel cote poste local.
- Le banker ne depend plus d une liste d actifs hardcodee: l univers MT5 est decouvert dynamiquement et classe en `forex`, `cfd`, `crypto`.
- Les filtres week-end/session et l anti-spam Telegram sont corriges.
- Les tests banker modifies passent localement.
- Le `GNN` multi-timeframe est entraine a `500` epochs cumulees au total sur GPU.
- Le pipeline nocturne EVA Lab a ete fiabilise: sync Proxmox, resume JSON persistant, packaging `eva-lab` corrige.
- La pile runtime JAX CUDA validee sur serveur est maintenant: `jax 0.4.23`, `jaxlib 0.4.23+cuda11.cudnn86`, `haiku 0.0.11`, backend `gpu`.
- MuZero a ete corrige cote code et tourne a present sur GPU sur le serveur de dev.
- `eva-builder` a ete renforce pendant l attente trading: `CyberForge` partage entre API et factory, pipeline BMAD avec validation automatique Python et historique coherent.
- Les tests `eva-builder` locaux montent a `16 passed`.
- `eva-builder` sait maintenant synchroniser et exploiter un catalogue d APIs publiques depuis `public-apis/public-apis` pour enrichir les prompts produit et SaaS.
- `eva-builder` expose aussi une passerelle de mutation securisee avec `dry-run` par defaut et activation explicite par variable d environnement.
- `eva-builder` expose aussi une passerelle de deploiement structuree, en `dry-run` par defaut, pour les cibles `local` et `proxmox`.
- Le `banker` n ouvre plus de nouvelles positions tant qu EVA Lab ne fournit pas un `champion` live valide.
- Le mode `research-first` est maintenant privilegie: entrainement massif, notifications Telegram des meilleurs candidats, puis seulement bascule vers le live.
- `eva-nexus` est maintenant recable proprement sur Builder: proxy Docker ajoute, Muse repasse par le proxy local, liens Grafana alignes sur l hote courant, et cockpit Builder disponible dans Enterprise.
- Le frontend Nexus passe maintenant `npm run lint` puis `npm run build` localement.

### Ce qui est en cours
- `MuZero scalp` est en execution sur GPU.
- `MuZero intraday` et `MuZero swing` attendent la fin du premier horizon.
- `DreamerV3` demarrera automatiquement apres la sequence MuZero.
- La partie trading est donc surtout en attente active de fin d entrainement et de verification d artefacts.

### Ce qu il reste a faire
- Finaliser les runs `MuZero` de recherche, puis verifier les checkpoints et rapports generes sur un echantillon suffisant.
- Rendre `eva-trainer` nativement compatible JAX CUDA sans patch runtime a chaque lancement.
- Injecter uniquement les modeles promus par la gate dans la boucle de decision et d execution live du banker.
- Etendre l historique et l univers au-dela des `6` symboles actuels.
- Connecter ensuite les exchanges crypto (`Binance`, `Kraken`, `Coinbase`) au meme pipeline.
- Finaliser la boucle ADN, evolution et champion-challenger sur resultats reels.
- Migrer les secrets hors `.env` en clair vers un coffre dedie.
- Basculer les flux Builder Nexus de `dry-run` vers deploiement et mutation live uniquement apres validation serveur complete.

### Travail recommande pendant l attente
- Priorite 1 deja engagee: `eva-builder`, avec consolidation des services et de la boucle de validation locale pendant que le training tourne.
- Priorite 2 deja engagee: `eva-nexus`, avec recablage des proxies et cockpit Builder dans Enterprise.
- Priorite 3: `RLM` seulement en appui, pas comme chantier principal, car sa boucle existe deja.
