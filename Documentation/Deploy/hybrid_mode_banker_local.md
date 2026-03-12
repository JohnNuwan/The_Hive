# Mode Hybride: serveur Docker + Banker local

Ce mode correspond a ton usage actuel:
- Serveur Proxmox (192.168.1.6): tous les services Docker.
- PC Windows local: uniquement `banker.bat` pour MT5 natif.

Depuis le 08/03/2026, le mode recommande ne passe plus par un appel navigateur
direct vers `127.0.0.1:8100`. Le `banker` local ouvre un reverse tunnel SSH
vers le serveur, puis Nexus appelle ce tunnel via son proxy Nginx. Cela evite
les avertissements Firefox lies aux acces reseau local et garde l'UI en
same-origin.

## 1) Variables cote serveur (`.env`)

Definir au minimum:

```env
HIVE_REDIS_HOST="redis"
HIVE_QDRANT_HOST="qdrant"
HIVE_NEO4J_HOST="neo4j"
HIVE_MQTT_HOST="mosquitto"
HIVE_COMFYUI_HOST="comfyui"
```

## 2) Deploiement serveur

Par defaut, le service `banker` Docker est desactive (profile `with-banker`).

```bash
docker compose up -d --remove-orphans
```

Si tu veux temporairement lancer aussi le banker en conteneur:

```bash
docker compose --profile with-banker up -d --remove-orphans
```

## 3) Lancement local Banker (PC Windows)

Definir sur le PC local (facultatif si valeur par defaut):

```bat
set HIVE_SERVER_HOST=192.168.1.6
set HIVE_SSH_USER=aza
set HIVE_TUNNEL_REMOTE_PORT=18100
set HIVE_TUNNEL_RELAY_PORT=18101
```

Puis lancer:

```bat
banker.bat
```
Le script `banker.bat` :
- expose l'API locale sur `0.0.0.0:8100`
- ouvre un reverse tunnel SSH vers `127.0.0.1:18100` sur le serveur
- lance un relay distant sur `0.0.0.0:18101`
- refuse une seconde instance locale si `8100` repond deja

Nexus appelle ensuite `host.docker.internal:18101` cote serveur. Il n'y a donc
plus de fetch navigateur vers `127.0.0.1:8100`.

Politique live recommandee:
- `BANKER_REQUIRE_VALID_CHAMPION=true`
- `MUZERO_LIVE_SELECTION_POLICY=champion_only`
- `TRAINING_PROFILE=research` pour les runs massifs

Avec ce mode, le `banker` ne prend aucune nouvelle position tant qu'EVA Lab
ne renvoie pas un `champion` ou `legacy_champion` valide. Les rapports de
training et les motifs de blocage sont pousses dans Telegram.

## 4) Verification rapide

- `GET http://127.0.0.1:18101/health` sur le serveur doit repondre `status=ok`.
- `GET http://192.168.1.6:3030/api/banker/health` doit repondre `status=ok`.
- `GET http://192.168.1.6:3030/api/banker/trading/status` doit retourner les donnees MT5 locales.
- Les heartbeats Redis ne doivent plus remonter d'erreurs de resolution DNS/connexion.

## 5) Etat au 07/03/2026 (fait)

- Stack vLLM active sur serveur Proxmox (`192.168.1.6`) avec modele courant `Qwen/Qwen2.5-1.5B-Instruct`.
- Services verifies apres correctifs/redeploy:
  - `core` (`:8080`) OK
  - `kernel` (`:8800`) OK
  - `nervous` (`:9090`) OK
  - `vllm` (`:8000`) OK
- `/agents/status` remonte tous les agents serveur online; `banker` passe online des que `banker.bat` est lance et que le tunnel SSH est etabli.
- Correctif heartbeat compliance applique (`eva.compliance.status` + compat legacy `eva.keeper.status`).
- Validation E2E confirmee avec banker local actif: `/trading/status` renvoie `banker.status = online` avec donnees compte/positions.
- Correctifs modeles appliques:
  - routage Council role `code` corrige (`council_model_code`)
  - banker cortex sans hardcode `gemma3:4b`
  - fallback vLLM auto sur modele disponible en cas de 404
  - configuration locale alignee sur `Qwen/Qwen2.5-1.5B-Instruct`
- Script de sync/deploiement mis a jour:
  - rebuild `compliance` ajoute
  - force recreate `kernel` + `nervous` ajoute
  - `BANKER_API_HOST/PORT` ne sont plus ecrases automatiquement (pour conserver l'IP locale choisie).

## 6) Reste a faire (priorise)

- Lancer `banker.bat` sur le PC local Windows (MT5), puis verifier `http://192.168.1.6:3030/api/banker/health`.
- Convertir la mise en place de la cle SSH de tunnel en procedure outillee si le poste Windows change.
- Charger des modeles specialises par role (routing pret, mapping courant unifie sur Qwen).
- Migrer les secrets sensibles hors `.env` en clair vers un coffre de secrets.


