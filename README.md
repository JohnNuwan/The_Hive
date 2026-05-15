# THE HIVE

THE HIVE est la base de travail locale qui pilote deux plans distincts :

- le live local Windows via les instances Banker + MT5 ;
- l'entrainement distant EVA Lab sur le serveur Proxmox `192.168.1.6`.

Le depot n'est plus un prototype generaliste. Il est maintenant oriente vers
un flux d'exploitation precis :

- `FTMO 10K` comme compte master de decision ;
- `FTMO 50K` comme follower principal ;
- plusieurs `FTUK 100K` comme followers supplementaires ;
- un cycle `MuZero scalp full-7` cote serveur ;
- un cutover Arena tardif avec screen sur `10000 / 12000 / 14000`.

## Etat actuel

### Live local

Le live tourne sur Windows avec MT5 natif. Les API Banker exposent les comptes
suivants :

| Port | Role | Compte | Broker | Mode |
| --- | --- | --- | --- | --- |
| `8100` | Master | `521044924` | FTMO | decision live |
| `8110` | Follower | `531240000` | FTMO | copy trading |
| `8120` | Follower | `333382300` | FTUK | copy trading |
| `8130` | Follower | `333382206` | FTUK | copy trading |
| `8170` | Follower | `541264545` | FTMO | copy trading |
| `8180` | Follower | `333382439` | FTUK | copy trading |

Comptes retires ou mis de cote :

- `333382355` : retire du fleet actif ;
- `333382356` : mis de cote jusqu'a validation des identifiants.

Regles de copie actuellement chargees :

- cloture gagnante du master : `70% close + SL au break-even` ;
- cloture perdante ou `stop loss` du master : fermeture `100%` des followers ;
- les followers FTUK utilisent un mapping de symboles explicite ;
- les garde-fous de spread sont maintenant definis par symbole.

### Training distant

Le training tourne sur le serveur Proxmox via `scripts/deploy/start_training_proxmox.py`.

Le cycle cible actuel est :

- moteur : `MuZero`
- horizon : `scalp`
- univers canonique : `XAUUSD, US30.cash, GER40.cash, EURUSD, US100.cash, US500.cash, BTCUSD`
- mode : `full-7`
- Arena : screen tardif sur `10000 / 12000 / 14000`

Le cutover Arena ne doit plus stopper le run a `9000`. La coupure valide
attend `14000` pour rendre possible le screen complet.

## Arborescence utile

### Racine

- `banker.bat` : wrapper commun de lancement Banker
- `banker.master.bat` : master FTMO 10K
- `banker.ftmo50k.bat` : follower FTMO 50K
- `banker.ftuk100k*.bat` : followers FTUK dedies
- `banker.challenge.all.bat` : demarrage de toute la flotte live locale
- `banker.dashboard.bat` : dashboard Rich local
- `scripts/deploy/start_training_proxmox.py` : sync + lancement distant training/Arena

### Code applicatif

- `src/eva-banker` : execution live, copy trading, risk, dashboard, MT5 bridge
- `src/eva-lab` : entrainement MuZero, Arena, promotion de champion, notifier
- `src/shared` : configuration et modeles communs

### Donnees et logs

- `data/history` : historiques CSV utilises par le training
- `data/shadow_learning/imports` : imports shadow learning
- `logs` : logs locaux Banker, dashboard, watcher Arena
- `tmp` : archive de travail et fichiers temporaires

## Lancement local

### Demarrer toute la flotte live

```bat
banker.challenge.all.bat
```

### Demarrer uniquement le master

```bat
banker.master.bat
```

### Ouvrir le dashboard local

```bat
banker.dashboard.bat
```

## Lancement training + Arena

### Lancer un cycle MuZero scalp full-7

```powershell
python scripts/deploy/start_training_proxmox.py --muzero-scalp-full-7 --stop-existing
```

### Attacher le watcher Arena tardif

```powershell
python scripts/deploy/start_training_proxmox.py --muzero-scalp-arena-cutover-8000
```

Le watcher :

- attend `ckpt14000` ;
- evalue `10000 / 12000 / 14000` ;
- ne lance la full Arena que si au moins un checkpoint sort `VICTORY`.

## Historique de marche

Le refresh d'historique local et distant passe par :

```powershell
python scripts/fetch_history.py
```

Pour un cycle `full-7`, le lanceur distant effectue maintenant un refresh cible
sur :

- `XAUUSD`
- `US30.cash`
- `GER40.cash`
- `EURUSD`
- `US100.cash`
- `US500.cash`
- `BTCUSD`

Timeframes obligatoires :

- `M1`
- `M5`
- `M15`
- `H1`
- `D1`
- `W1`

## Documentation de reference

- `Documentation/Deploy/hybrid_mode_banker_local.md`
- `Documentation/Deploy/stack_operationnelle_actuelle.md`

## Regles de maintenance

- ne pas deplacer les launchers actifs hors racine sans mettre a jour les usages Windows ;
- archiver les fichiers temporaires dans `tmp/archive/...` au lieu de les laisser en racine ;
- ne pas relancer le master en plein panier sans raison precise, car cela casse les liens
  memoire de copie sur les tickets deja ouverts ;
- ne pas promouvoir un challenger en live sans victoire Arena exploitable.
