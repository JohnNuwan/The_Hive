# Exploitation CPU Sure Pendant Un Run Trading Actif

Ce document decrit les charges CPU qui peuvent etre exploitees sans perturber
le run trading actif.

## Principe

Le CPU peut servir:
- la connaissance;
- l'OSINT et la securite;
- les usines business;
- la consolidation lecture seule de `core`.

En revanche, il ne faut pas lancer en parallele:
- un second training `MuZero`;
- `Dreamer offline`;
- `GNN market`;
- un second `eva-trainer`;
- des ecritures dans `data/muzero`, `data/checkpoints`, `data/history`.

## CLI de pilotage

La CLI dediee est:

```powershell
python scripts/cpu_assist.py list
```

Une facade plus courte existe aussi:

```powershell
python scripts/evactl.py cpu list
```

Elle expose 4 groupes:
- `knowledge`
- `ops`
- `business`
- `core-readonly`

Par defaut, le mode `run` reste en dry-run:

```powershell
python scripts/cpu_assist.py run --group knowledge --group ops
```

Pour executer reellement les jobs lecture seule:

```powershell
python scripts/cpu_assist.py run --group knowledge --group core-readonly --execute
```

Equivalent via `evactl`:

```powershell
python scripts/evactl.py cpu run --group knowledge --group core-readonly --execute
```

Pour inclure aussi les jobs actifs non trading:

```powershell
python scripts/cpu_assist.py run --group knowledge --group business --execute --include-active
```

Le controle du run trading peut etre force:

```powershell
python scripts/cpu_assist.py run --group knowledge --execute --strict-training-check
```

## Groupes Disponibles

### knowledge
- `researcher.health`
- `researcher.sync_sources`
- `researcher.ingest_status`
- `researcher.review_pending`
- `researcher.approved`
- `researcher.sources`
- `core.memory_fragments`
- `core.memory_graph`

### ops
- `shadow.health`
- `shadow.alerts`
- `shadow.monitors`
- `shadow.personas`
- `shadow.threat_history`
- `sentinel.health`
- `sentinel.metrics`
- `sentinel.alerts`
- `sentinel.audit_logs`
- `sentinel.compliance_check`
- `sentinel.security_scan`
- `sentinel.integrity_check`
- `substrate.health`
- `substrate.metrics`
- `substrate.mode`
- `substrate.alerts`
- `substrate.thresholds`
- `substrate.metrics_history`

### business
- `builder.health`
- `builder.docgen`
- `builder.log_analysis`
- `builder.catalog_sync`
- `builder.pipeline_status`
- `builder.build_history`
- `builder.deploy_history`
- `muse.health`
- `muse.stats`
- `muse.niches`
- `muse.niche_scores`
- `muse.templates`
- `accountant.health`
- `accountant.report`
- `accountant.dashboard`
- `accountant.expenses`
- `accountant.export`
- `compliance.health`
- `compliance.ledger`
- `compliance.identity`
- `compliance.history`
- `compliance.urssaf_report`
- `compliance.alerts`
- `rwa.health`
- `rwa.portfolio`
- `rwa.strategy`
- `rwa.recommendations`
- `rwa.telemetry`
- `rwa.energy_history`

### core-readonly
- `core.health`
- `core.agents_status`
- `core.intelligence_status`
- `core.autonomy_context`
- `core.system_status`
- `core.docker_containers`
- `core.telemetry`
- `core.circuit_breaker`

## Regles D'Execution

- utiliser `--execute` seulement pour les services non trading;
- n'activer `--include-active` que si la charge CPU additionnelle est voulue;
- garder un `parallelism` faible;
- ne jamais toucher `lab`, `trainer`, `banker`, `vllm`;
- verifier que le `run_id` du training ne change pas.

## Rapport JSON

La CLI peut produire un rapport JSON hors des artefacts trading:

```powershell
python scripts/cpu_assist.py run --group knowledge --execute --report data/builder/cpu_assist/latest.json
```

Le rapport contient:
- les jobs selectionnes;
- les statuts HTTP;
- un apercu de reponse;
- le `run_id` training avant/apres;
- la confirmation que le trainer reste actif.

## Statuts De Resultat

Les jobs peuvent renvoyer:
- `ok`: execution terminee;
- `unavailable`: endpoint optionnel absent sur la version serveur;
- `degraded`: endpoint optionnel lent ou temporairement instable;
- `http_error` ou `error`: echec bloquant sur un job requis.

Les endpoints optionnels ne font pas echouer le run CPU tant que le training
reste coherent.
