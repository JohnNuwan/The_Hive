# Agent Follower Client

L'agent follower permet de faire tourner les comptes clients sur leur propre PC ou VPS, au lieu de concentrer tous les terminaux MT5 sur le poste maitre.
Le mode recommande est maintenant le **Fleet Manager** : une seule application locale peut gerer plusieurs comptes MT5.

## Architecture

```text
Master live -> Relay central -> Hive Follower Fleet Manager -> N agents -> N terminaux MT5
```

- Le master publie des commandes idempotentes vers le relay.
- Chaque compte local possede son propre `client_id`, son terminal MT5 et son fichier d'etat.
- Le Fleet Manager lance un agent par compte actif.
- Chaque agent lit uniquement ses commandes, applique le mapping symbole local, le sizing dynamique, puis execute dans MT5.
- Les secrets MT5 restent sur le PC client.
- Le heartbeat remonte l'etat vers le relay et le dashboard central.

## Commandes locales

Lancer le relay central sur un serveur accessible par les clients :

```powershell
python scripts/run_follower_relay.py
```

Creer une configuration mono-compte exemple :

```powershell
python scripts/run_follower_agent.py --init-config --config data/follower_agent/config.json
```

Creer une configuration multi-comptes exemple :

```powershell
python scripts/run_follower_agent.py --init-fleet --fleet-config data/follower_agent/fleet.config.json
```

Lancer un agent mono-compte en console :

```powershell
python scripts/run_follower_agent.py --config data/follower_agent/config.json
```

Lancer la flotte multi-comptes en console :

```powershell
python scripts/run_follower_agent.py --fleet --fleet-config data/follower_agent/fleet.config.json
```

Lancer l'interface CustomTkinter multi-comptes :

```powershell
python scripts/run_follower_app.py --config data/follower_agent/fleet.config.json
```

Lancer l'ancienne interface mono-compte :

```powershell
python scripts/run_follower_agent.py --ui --config data/follower_agent/config.json
```

Lancer l'interface multi-comptes depuis le CLI principal :

```powershell
python scripts/run_follower_agent.py --ui --fleet --fleet-config data/follower_agent/fleet.config.json
```

## Fleet Manager

Le fichier `data/follower_agent/fleet.config.json` contient un tableau `accounts[]`.
Chaque compte expose :

- `enabled` : active ou ignore le compte au demarrage global.
- `client_id` : identifiant utilise par le relay pour router les commandes.
- `account_label` : libelle visible dans l'interface.
- `mt5_login`, `mt5_password`, `mt5_server`, `mt5_terminal_path` : connexion locale MT5.
- `allocation_ratio` : multiplicateur de risque manuel applique apres le sizing dynamique.
- `balance_reference` : capital du compte follower, ou `null` pour lire l'equity MT5 locale.
- `master_balance_reference` : capital de reference du maitre, `10000` par defaut pour le master 10K.
- `symbol_map` : traduction des symboles du master vers le broker local.
- `state_path` et `log_path` : fichiers separes pour eviter les collisions entre comptes.

Regle critique : un compte MT5 doit avoir son terminal ou son dossier data separe. Sinon MetaTrader peut melanger les sessions et provoquer des doubles connexions.

## Regles copy-trading

- Ouverture : mapping symbole + volume local `volume_maitre * (capital_follower / capital_master) * allocation_ratio`.
- Si le relay transmet `master_equity` ou `master_balance`, cette valeur remplace `master_balance_reference`.
- Cloture maitre positive : l'agent ferme 70% de la position follower et remonte le SL au break-even.
- Cloture maitre negative ou SL : l'agent ferme 100% de la position follower.
- Idempotence : chaque `command_id` n'est execute qu'une seule fois.
- Etat local : chaque compte conserve ses liens `master_ticket -> local_ticket` dans son propre `state_path`.

## Relay attendu

Le MVP fournit ces endpoints HTTP :

- `POST /api/master/commands`
- `GET /api/follower/commands?client_id=...&after=...`
- `POST /api/follower/ack`
- `POST /api/follower/heartbeat`
- `GET /api/relay/status`

Le relay doit signer ou filtrer les commandes par client. L'agent envoie `Authorization: Bearer <api_token>` si un token est configure.

## Prochaine etape

Le MVP multi-comptes est pret pour le packaging Windows. La prochaine passe doit ajouter la protection des secrets via Windows Credential Manager/DPAPI, une icone tray et un mode service au demarrage.
