# Profils Reduits MuZero

Ce document fixe les profils de relance cibles pour les horizons `scalp`,
`intraday` et `swing` quand on veut deboguer vite sans relancer tout le
pipeline.

## Principe

Les profils reduits servent a:
- raccourcir les boucles d'entrainement;
- verifier rapidement la qualite d'un horizon;
- valider l'Arena et la promotion sur un univers lisible;
- garder le live en `demo` pendant le debug.

Ils ne servent pas a produire un champion final multi-marche complet.

## Univers Par Defaut

### scalp
- `EURUSD`
- `GBPUSD`
- `USDJPY`
- `XAUUSD`
- `BTCUSD`
- `ETHUSD`
- `US30.cash`
- `US500.cash`
- `GER40.cash`

### intraday
- `EURUSD`
- `GBPUSD`
- `USDJPY`
- `XAUUSD`
- `US30.cash`
- `US500.cash`
- `GER40.cash`

### swing
- `EURUSD`
- `GBPUSD`
- `USDJPY`
- `XAUUSD`
- `US30.cash`
- `US500.cash`
- `GER40.cash`

## Parametres Par Horizon

### scalp
- `MUZERO_TRAINING_STEPS=12000`
- `MUZERO_GAMES_PER_SYMBOL=10`
- `ARENA_GAMES_PER_SYMBOL=4`
- `ARENA_MIN_GAMES=18`
- `ARENA_MIN_SYMBOLS=5`
- `MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS=5`
- `MUZERO_LIVE_TOP_SYMBOLS=5`

### intraday
- `MUZERO_TRAINING_STEPS=8000`
- `MUZERO_GAMES_PER_SYMBOL=8`
- `ARENA_GAMES_PER_SYMBOL=4`
- `ARENA_MIN_GAMES=14`
- `ARENA_MIN_SYMBOLS=4`
- `MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS=4`
- `MUZERO_LIVE_TOP_SYMBOLS=4`

### swing
- `MUZERO_TRAINING_STEPS=6000`
- `MUZERO_GAMES_PER_SYMBOL=6`
- `ARENA_GAMES_PER_SYMBOL=3`
- `ARENA_MIN_GAMES=10`
- `ARENA_MIN_SYMBOLS=4`
- `MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS=4`
- `MUZERO_LIVE_TOP_SYMBOLS=4`

## Commandes

### scalp reduit
```powershell
python scripts\deploy\start_training_proxmox.py --scalp-reduced --stop-existing
```

### intraday reduit
```powershell
python scripts\deploy\start_training_proxmox.py --intraday-reduced --stop-existing
```

### swing reduit
```powershell
python scripts\deploy\start_training_proxmox.py --swing-reduced --stop-existing
```

### socle reduit complet
```powershell
python scripts\deploy\start_training_proxmox.py --all-reduced --stop-existing
```

### Surcharge manuelle des symboles
```powershell
python scripts\deploy\start_training_proxmox.py --intraday-reduced --stop-existing --symbols "EURUSD,GBPUSD,USDJPY,XAUUSD,US30.cash,US500.cash,GER40.cash"
```

## Ordre Recommande

1. `scalp` reduit
2. `intraday` reduit
3. `swing` reduit
4. ensuite seulement, split par familles d'actifs

Quand on veut relancer rapidement tout le socle de validation sans separer les
horizons, utiliser `--all-reduced`.

## Cible Long Terme

La cible finale n'est ni:
- un modele unique sur 83 actifs;
- ni un modele par actif.

La cible finale est:
- un modele par horizon;
- puis un modele par famille d'actifs.

Familles recommandees:
- `FX`
- `indices`
- `metaux`
- `crypto`
