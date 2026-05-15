# Mode hybride: serveur Proxmox + Bankers locaux Windows

Ce document decrit le mode effectivement utilise aujourd'hui.

## 1. Principe

- le `training`, l'`Arena`, Redis, TimescaleDB et l'inference distante vivent
  sur `192.168.1.6` ;
- les `Bankers` et `MetaTrader 5` vivent sur le PC Windows local ;
- le `master` prend les decisions live ;
- les followers FTMO/FTUK copient avec gestion de risque et mapping broker.

## 2. Ports locaux

| Port | Instance | Usage |
| --- | --- | --- |
| `8100` | FTMO Master 10K | decision live |
| `8110` | FTMO Challenge 50K | follower |
| `8120` | FTUK 333382300 | follower |
| `8130` | FTUK 333382206 | follower |
| `8170` | FTMO 541264545 | follower |
| `8180` | FTUK 333382439 | follower |

Comptes retires ou mis de cote :

- `333382355` : retire du fleet actif ;
- `333382356` : mis de cote jusqu'a validation des identifiants.

## 3. Lancement

### 3.1 Toute la flotte live

```bat
banker.challenge.all.bat
```

### 3.2 Master seul

```bat
banker.master.bat
```

### 3.3 Dashboard

```bat
banker.dashboard.bat
```

## 4. Regles de copie

### 4.1 Ouvertures

- le master ouvre sur son compte FTMO ;
- les followers recoivent les ordres via `COPY_ROUTER` ;
- les FTUK traduisent les symboles via `symbol_map`.

### 4.2 Fermetures

- gain du master :
  - fermeture de `70%`
  - `SL` du reliquat remonte au `break-even`
  - meme logique sur les followers
- perte du master ou `SL` :
  - fermeture `100%` sur les followers

## 5. Verrous de securite

- un compte MT5 ne peut pas etre reclame par deux API Banker en meme temps ;
- chaque follower FTUK utilise son propre terminal dedie ;
- les garde-fous de spread sont calibres par symbole.

## 6. Verification rapide

### 6.1 Sanite locale

```powershell
Invoke-RestMethod http://127.0.0.1:8100/trading/status | ConvertTo-Json -Depth 6
Invoke-RestMethod http://127.0.0.1:8100/copy-trading/status | ConvertTo-Json -Depth 6
```

### 6.2 Health des followers

```powershell
Invoke-RestMethod http://127.0.0.1:8110/health
Invoke-RestMethod http://127.0.0.1:8120/health
Invoke-RestMethod http://127.0.0.1:8130/health
Invoke-RestMethod http://127.0.0.1:8170/health
Invoke-RestMethod http://127.0.0.1:8180/health
```

## 7. Points d'attention

- ne pas redemarrer le master en plein panier sans besoin reel ;
- si un patch ne concerne que les followers, redemarrer uniquement les followers ;
- si un terminal FTUK est duplique par erreur, le garde-fou de claim doit
  maintenant refuser le doublon ;
- si des positions manquent sur un follower apres son redemarrage, utiliser la
  reparation du panier ouvert depuis le master au lieu d'ouvrir manuellement.
