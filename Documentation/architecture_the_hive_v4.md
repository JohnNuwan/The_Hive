# ARCHITECTURE DE TRADING ANTI-FRAGILE — THE HIVE V4.0

Ce document sert de manuel de référence pour l'architecture système unifiée de **THE HIVE v4.0**, combinant l'apprentissage par renforcement profond (MuZero, DreamerV3), la modélisation sémantique auto-supervisée (Market-JEPA), l'intégration relationnelle (GNN), le stockage de séries temporelles (TimescaleDB) et la mémoire associative (HippoRAG 2).

---

## 1. La Philosophie Systémique de la v4.0

THE HIVE v4.0 vise à introduire des propriétés d'**anti-fragilité** et de **résilience absolue** face aux variations chaotiques des marchés financiers. Plutôt que de subir la dérive de distribution des prix bruts (*data drift*), le système opère à travers un double filtre :
1. Un filtre relationnel (**GNN**) qui modélise les corrélations structurelles entre tous les actifs de la flotte.
2. Un filtre sémantique auto-supervisé (**Market-JEPA**) qui projette les observations bruitées dans un espace latent ultra-stable et robuste.

---

## 2. Encodage Auto-Supervisé : Market-JEPA & Perte VICReg

Le composant `jepa_encoder.py` implémente en JAX/Haiku l'architecture **Market-JEPA (Joint Embedding Predictive Architecture)**. 

### Rôle et Fonctionnement :
* **Débruitage sémantique** : Contrairement aux auto-encodeurs classiques (qui reconstruisent les pixels ou les prix bruts, gaspillant de la capacité sur le bruit à haute fréquence), JEPA apprend à prédire les représentations latentes futures du marché directement dans l'espace latent.
* **VICReg Loss (Variance-Invariance-Covariance)** : Pour éviter que les réseaux ne subissent un effondrement des représentations (où tous les états convergent vers un vecteur nul ou constant), nous appliquons la perte VICReg :
  * **Invariance (similitude)** : Force la représentation prédite à être proche de la représentation cible réelle future.
  * **Variance** : Force l'écart-type de chaque dimension latente à rester au-dessus d'un seuil $\gamma = 1.0$, garantissant que chaque lot contient des informations diverses.
  * **Covariance** : Minimise les covariances croisées hors-diagonale pour décorréler les dimensions latentes et supprimer toute redondance.

```mermaid
graph LR
    Obs_t[Observation t] -->|Context Encoder| Z_x[Latent z_x]
    Obs_tk[Observation t+k] -->|Target Encoder| Z_y[Latent Cible z_y]
    Z_x -->|Predictor| Z_y_hat[Latent Prédit z_y_hat]
    Z_y_hat <-->|VICReg Loss| Z_y
```

---

## 3. Agent MuZero JAX & Modèle de Monde Conditionné

L'intégration au sein de l'agent MuZero JAX (`jax_agent.py` et `jax_networks.py`) s'effectue par injection directe :

* **Injection Dynamique de Poids** : Au démarrage de `JAXMuZeroAgent`, si `use_jepa_encoder = True` est activé, l'agent charge les paramètres du fichier `jepa_encoder_latest.pkl`. Les tenseurs Haiku sous le namespace `context_encoder/` sont automatiquement mappés et injectés dans les variables correspondantes du `RepresentationNetwork` de MuZero.
* **Stabilité du MCTS** : En travaillant dans l'espace latent stable et normalisé ($\tanh$) produit par le JEPA, les pas de dynamique recurrents de MuZero subissent beaucoup moins d'erreurs d'accumulation temporelle, ce qui accélère la convergence des simulations de l'arbre MCTS sous JIT.

---

## 4. Infrastructure de Données : Ingestion de Flotte Dédupliquée

L'alimentation de notre base de données et des pipelines de Shadow Learning repose sur l'ingestion automatisée multi-comptes (`mt5_history_pipeline.py`) :

* **Duplication Impossible** : Pour éviter toute distorsion statistique durant l'entraînement de MuZero, chaque trade importé est verrouillé via une clé unique composite :
  $$\text{position\_key} = \text{account\_key} : \text{position\_id}$$
  Cette clé est archivée dans un registre persistant `state.json`. Les positions déjà connues ne sont jamais ingérées deux fois.
* **TimescaleDB comme source de vérité** : Bien que le chargeur Eva-Lab (`load_history_frame`) intègre un repli silencieux vers les fichiers CSV sur disque, TimescaleDB (`timescale_store.py`) est notre stockage cible pour l'historique et les manifestes. Il fournit une isolation optimale des séries temporelles via son architecture d'hypertable compressée.

---

## 5. Diagnostic de HippoRAG 2 : La Mémoire Associative Non Exploitée

### Le Constat Actuel :
> [!WARNING]
> Actuellement, le module de mémoire associative de pointe **HippoRAG 2** (`MemoryBridge` & `GraphMemory` sur Neo4j) est **totalement découplé** de nos réseaux quantitatifs de Deep Learning (JEPA/MuZero). Il est cantonné au rôle d'assistant de recherche sémantique pour l'LLM de supervision (Eva-Core).

### Pourquoi est-ce sous-exploité ?
L'agent de trading prend des décisions en se basant uniquement sur des vecteurs d'indicateurs numériques locaux. Il n'a aucune conscience de la "mémoire à long terme" du système :
* Il ignore si le marché actuel présente une analogie historique avec une crise macroéconomique passée consignée dans le graphe de connaissances Neo4j.
* Il n'a aucun moyen de projeter une interrogation sémantique sous forme d'embedding de contexte pour guider sa prise de décision.

### Piste d'Unification (THE HIVE v5.0) :
Pour combler cette lacune, nous devons introduire le concept de **Conditionnement par Mémoire Graphique** :
1. Lors du self-play, l'agent extrait des triplets sémantiques ou de régime de marché de l'observation courante.
2. Il interroge HippoRAG 2 via une recherche hybride (PPR sur Neo4j) pour extraire les faits associés passés.
3. Ces relations sous forme de vecteurs d'embeddings de graphe de connaissances sont injectées en entrée de l'encodeur JEPA au même titre que les embeddings de structure GNN.
4. L'agent MuZero dispose ainsi d'une mémoire associative unifiée capable de surmonter la dérive de distribution.
