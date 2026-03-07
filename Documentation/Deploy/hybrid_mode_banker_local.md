# Mode Hybride: serveur Docker + Banker local

Ce mode correspond a ton usage actuel:
- Serveur Proxmox (192.168.1.6): tous les services Docker.
- PC Windows local: uniquement `banker.bat` pour MT5 natif.

## 1) Variables cote serveur (`.env`)

Definir au minimum:

```env
BANKER_API_HOST="IP_DU_PC_LOCAL"
BANKER_API_PORT=8100
HIVE_REDIS_HOST="redis"
HIVE_QDRANT_HOST="qdrant"
HIVE_NEO4J_HOST="neo4j"
HIVE_MQTT_HOST="mosquitto"
HIVE_COMFYUI_HOST="comfyui"
```

Exemple: si le PC local est `192.168.1.20`:

```env
BANKER_API_HOST="192.168.1.20"
BANKER_API_PORT=8100
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
```

Puis lancer:

```bat
banker.bat
```
Le script `banker.bat` tente d'ajouter automatiquement une regle firewall Windows pour le port `8100` (si les droits systeme le permettent).

Verifier que le port `8100` est accessible depuis le serveur.

## 4) Verification rapide

- `core` doit joindre `http://<BANKER_API_HOST>:8100`.
- `GET /trading/status` sur core doit retourner `banker.status = online`.
- Les heartbeats Redis ne doivent plus remonter d'erreurs de resolution DNS/connexion.

## 5) Etat au 07/03/2026 (fait)

- Stack vLLM active sur serveur Proxmox (`192.168.1.6`) avec modele courant `Qwen/Qwen2.5-1.5B-Instruct`.
- Services verifies apres correctifs/redeploy:
  - `core` (`:8080`) OK
  - `kernel` (`:8800`) OK
  - `nervous` (`:9090`) OK
  - `vllm` (`:8000`) OK
- `/agents/status` remonte tous les agents serveur online; `banker` passe online des que `banker.bat` est lance sur le PC local.
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

- Lancer `banker.bat` sur le PC local Windows (MT5), puis verifier l'accessibilite depuis le serveur sur `http://<IP_PC_LOCAL>:8100`.
- Confirmer la valeur finale de `BANKER_API_HOST` cote serveur pour verrouiller le mode hybride.
- Charger des modeles specialises par role (routing pret, mapping courant unifie sur Qwen).
- Migrer les secrets sensibles hors `.env` en clair vers un coffre de secrets.


