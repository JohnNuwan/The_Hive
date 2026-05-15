# Ingestion historique MT5 multi-comptes

Ce pipeline transforme les historiques des comptes MT5 en donnees exploitables
par Shadow Learning et Nemesis.

## Principe

- Le pipeline interroge les instances Banker via HTTP.
- Il n'ouvre pas directement les terminaux MT5 afin d'eviter les conflits de
  claim avec le live.
- Le master expose la flotte via `/copy-trading/status`.
- Chaque Banker expose ses deals via `/history/deals`.

## Commande

```powershell
$env:PYTHONPATH='src/eva-lab;src/eva-banker;src/shared'
.\venv\Scripts\python.exe scripts\ingest_mt5_fleet_history.py --days 30
```

Options utiles :

- `--include-disabled` inclut les comptes desactives dans la decouverte.
- `--force` reimporte les positions deja vues.
- `--max-deals 5000` borne le volume lu par compte.
- `--master-url http://127.0.0.1:8100` force le master a interroger.

## Sorties

- `data/mt5_history_ingestion/raw_deals/` : deals MT5 bruts decores.
- `data/mt5_history_ingestion/positions/` : positions fermees normalisees.
- `data/mt5_history_ingestion/open_positions/` : positions ouvertes observees.
- `data/shadow_learning/mt5_fleet/` : transitions Shadow Learning.
- `data/mt5_history_ingestion/nemesis/` : slices Nemesis issues des pertes.
- `data/mt5_history_ingestion/reports/latest.json` : rapport du dernier batch.

## Exploitation

Les followers ne sont pas traites comme des strategies independantes. Le
pipeline conserve le role du compte, le groupe de copie, le symbole canonique et
les anomalies de copie pour separer :

- la decision strategique du master ;
- l'execution broker/follower ;
- les close partielles `EVA Close` ;
- les runners `HOLD` ;
- les pertes transformables en stress tests Nemesis.

## Precondition

Les bankers doivent etre redemarres apres le patch qui ajoute `/history/deals`.
Sans redemarrage, le pipeline verra une erreur HTTP sur cet endpoint.
