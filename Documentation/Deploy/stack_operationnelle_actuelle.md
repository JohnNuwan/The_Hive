# Stack operationnelle actuelle

Ce document remplace les anciennes descriptions trop generales. Il decrit
l'etat reel de la stack au moment ou le depot pilote :

- un `master FTMO 10K` ;
- un `follower FTMO 50K` ;
- followers FTMO/FTUK actifs, avec les challenges passes ou invalides mis de cote ;
- un training distant `MuZero scalp full-7` avec cutover Arena tardif.

## 1. Topologie live locale

### 1.1 Comptes et ports

| Port | Role | Login | Broker | Terminal |
| --- | --- | --- | --- | --- |
| `8100` | Master | `521044924` | FTMO | `John` |
| `8110` | Follower | `531240000` | FTMO | `Robin` |
| `8120` | Follower | `333382300` | FTUK | `John_100K_API_333382300` |
| `8130` | Follower | `333382206` | FTUK | `John_100K_API_333382206` |
| `8170` | Follower | `541264545` | FTMO | `Robin_541264545` |
| `8180` | Follower | `333382439` | FTUK | `John_100K_API_333382439` |

Comptes retires ou mis de cote :

- `333382355` : retire du fleet actif ;
- `333382356` : mis de cote jusqu'a validation des identifiants.

### 1.2 Lanceurs actifs

- `banker.master.bat`
- `banker.ftmo50k.bat`
- `banker.ftuk100k.bat`
- `banker.ftuk100k_333382206.bat`
- `banker.ftmo541264545.bat`
- `banker.ftuk100k_333382439.bat`
- `banker.challenge.all.bat`

## 2. Regles live actuellement voulues

### 2.1 Ouverture

- le `master` decide ;
- les followers copient selon `BANKER_COPY_TARGETS_JSON` ;
- les FTUK appliquent un mapping explicite de symboles ;
- les tailles followers sont calculees a partir de l'equity de la cible.

### 2.2 Fermeture

- cloture gagnante du `master` :
  - `70% close`
  - `SL` du reliquat remonte au `break-even`
  - propagation de la meme logique aux followers
- cloture perdante ou `SL` du `master` :
  - fermeture `100%` des followers

### 2.3 Garde-fous de spread

Les seuils ne sont plus par famille seulement, mais par symbole :

- `EURUSD` et majors forex : `25`
- `XAUUSD` : `60`
- `US100.cash` : `120`
- `US30.cash` : `250`
- `GER40.cash` / `DE40.e` : `300`
- `US500.cash` : `70`
- `BTCUSD` / `BTCUSD.e` : `1500`

## 3. Mapping FTUK

### 3.1 Mappings standards

- `XAUUSD -> XAUUSD.m`
- `EURUSD -> EURUSD.e`
- `GBPUSD -> GBPUSD.e`
- `USDCHF -> USDCHF.e`
- `USDJPY -> USDJPY.e`
- `USDCNH -> USDCNH.e`
- `AUDUSD -> AUDUSD.e`
- `NZDUSD -> NZDUSD.e`
- `USDCAD -> USDCAD.e`
- `USDSEK -> USDSEK.e`
- `US30.cash -> US30.e`
- `US100.cash -> USTEC.m`
- `GER40.cash -> DE40.e`
- `US500.cash -> US500.e`
- `BTCUSD -> BTCUSD.e`

### 3.2 Politique de terminal

- un terminal dedie par compte FTUK ;
- mode `non portable` ;
- compte revendique par verrou local pour empecher les doublons ;
- serveur FTUK de reference : `FTUKMarkets-Live`.

## 4. Training distant

### 4.1 Lancement

Le training distant se pilote via :

```powershell
python scripts/deploy/start_training_proxmox.py --muzero-scalp-full-7 --stop-existing
python scripts/deploy/start_training_proxmox.py --muzero-scalp-arena-cutover-8000
```

### 4.2 Univers canonique scalp

- `XAUUSD`
- `US30.cash`
- `GER40.cash`
- `EURUSD`
- `US100.cash`
- `US500.cash`
- `BTCUSD`

### 4.3 Cutover Arena

Le watcher attend maintenant :

- `ckpt10000`
- `ckpt12000`
- `ckpt14000`

Puis :

- stoppe le run au `ckpt14000` ;
- lance un screen sur les trois checkpoints ;
- ne lance la full Arena que si au moins un checkpoint est `VICTORY`.

## 5. Fichiers a surveiller

### 5.1 Logs locaux

- `logs/ftmo_master_10k.log`
- `logs/ftmo_challenge_50k.log`
- `logs/ftuk_challenge_100k.log`
- `logs/arena_cutover_watch.log`
- `logs/arena_cutover_watch.err.log`

### 5.2 Etat local

- `GET http://127.0.0.1:8100/trading/status`
- `GET http://127.0.0.1:8100/copy-trading/status`
- `GET http://127.0.0.1:8110/health`
- `GET http://127.0.0.1:8120/health`
- `GET http://127.0.0.1:8130/health`
- `GET http://127.0.0.1:8170/health`
- `GET http://127.0.0.1:8180/health`

### 5.3 Etat distant

- `/home/aza/The_Hive/data/checkpoints/training_status.json`
- `/home/aza/The_Hive/hive_nightly_training.log`

## 6. Discipline d'exploitation

- ne pas relancer le `master` en plein panier sans raison critique ;
- relancer les followers seuls quand un patch leur est specifique ;
- archiver les scripts temporaires et launchers obsoletes dans `tmp/archive` ;
- garder la racine reservee aux launchers actifs, au `README`, aux manifests et aux scripts d'entree.
