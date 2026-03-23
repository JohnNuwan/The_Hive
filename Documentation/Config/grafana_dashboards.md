# Configuration Dashboards Grafana - THE HIVE

> **Version**: 1.0.0  
> **Source de données**: Prometheus, TimescaleDB

---

##  Vue d'Ensemble

Ce document définit les dashboards Grafana pour le monitoring de THE HIVE.

---

##  Dashboard Principal - Panopticon

### Panneau 1: Statut Système Global

```json
{
  "title": " Statut THE HIVE",
  "type": "stat",
  "datasource": "Prometheus",
  "targets": [
    {
      "expr": "up{job=~\"eva-core|eva-banker|eva-sentinel\"}",
      "legendFormat": "{{job}}"
    }
  ],
  "options": {
    "colorMode": "background",
    "graphMode": "none",
    "justifyMode": "center"
  },
  "fieldConfig": {
    "defaults": {
      "mappings": [
        {"type": "value", "value": 1, "text": " EN LIGNE"},
        {"type": "value", "value": 0, "text": " HORS LIGNE"}
      ],
      "thresholds": {
        "steps": [
          {"color": "red", "value": 0},
          {"color": "green", "value": 1}
        ]
      }
    }
  }
}
```

### Panneau 2: Température GPU

```json
{
  "title": " Température GPU",
  "type": "gauge",
  "datasource": "Prometheus",
  "targets": [
    {
      "expr": "nvidia_smi_temperature_gpu",
      "legendFormat": "GPU Temp"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "celsius",
      "min": 0,
      "max": 100,
      "thresholds": {
        "steps": [
          {"color": "green", "value": 0},
          {"color": "yellow", "value": 70},
          {"color": "orange", "value": 80},
          {"color": "red", "value": 90}
        ]
      }
    }
  }
}
```

### Panneau 3: Utilisation Mémoire GPU

```json
{
  "title": " VRAM GPU",
  "type": "timeseries",
  "datasource": "Prometheus",
  "targets": [
    {
      "expr": "nvidia_smi_memory_used_bytes / 1073741824",
      "legendFormat": "Utilisé (GB)"
    },
    {
      "expr": "nvidia_smi_memory_total_bytes / 1073741824",
      "legendFormat": "Total (GB)"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "decgbytes",
      "custom": {
        "fillOpacity": 20,
        "lineWidth": 2
      }
    }
  }
}
```

---

##  Dashboard Trading

### Panneau 1: P&L Journalier

```json
{
  "title": " P&L Journalier",
  "type": "stat",
  "datasource": "TimescaleDB",
  "targets": [
    {
      "rawSql": "SELECT SUM(profit) as pnl FROM trade_orders WHERE closed_at >= NOW() - INTERVAL '1 day' AND status = 'closed'",
      "format": "table"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "currencyUSD",
      "thresholds": {
        "steps": [
          {"color": "red", "value": -1000},
          {"color": "yellow", "value": 0},
          {"color": "green", "value": 100}
        ]
      }
    }
  }
}
```

### Panneau 2: Drawdown en Temps Réel

```json
{
  "title": " Drawdown",
  "type": "gauge",
  "datasource": "TimescaleDB",
  "targets": [
    {
      "rawSql": "SELECT daily_drawdown_percent FROM v_current_risk ORDER BY timestamp DESC LIMIT 1",
      "format": "table"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "percent",
      "min": 0,
      "max": 5,
      "thresholds": {
        "steps": [
          {"color": "green", "value": 0},
          {"color": "yellow", "value": 2},
          {"color": "orange", "value": 3},
          {"color": "red", "value": 4}
        ]
      }
    }
  },
  "options": {
    "showThresholdLabels": true,
    "showThresholdMarkers": true
  }
}
```

### Panneau 3: Positions Ouvertes

```json
{
  "title": " Positions Ouvertes",
  "type": "table",
  "datasource": "TimescaleDB",
  "targets": [
    {
      "rawSql": "SELECT symbol, action, volume, open_price, current_price, profit, open_time FROM v_open_positions ORDER BY open_time DESC",
      "format": "table"
    }
  ],
  "fieldConfig": {
    "overrides": [
      {
        "matcher": {"id": "byName", "options": "profit"},
        "properties": [
          {
            "id": "thresholds",
            "value": {
              "steps": [
                {"color": "red", "value": -100},
                {"color": "green", "value": 0}
              ]
            }
          }
        ]
      }
    ]
  }
}
```

### Panneau 4: Anti-Tilt & Kill-Switch

```json
{
  "title": " Protections Actives",
  "type": "stat",
  "datasource": "TimescaleDB",
  "targets": [
    {
      "rawSql": "SELECT CASE WHEN anti_tilt_active THEN 1 ELSE 0 END as anti_tilt FROM risk_snapshots ORDER BY timestamp DESC LIMIT 1",
      "legendFormat": "Anti-Tilt"
    },
    {
      "rawSql": "SELECT CASE WHEN trading_allowed THEN 0 ELSE 1 END as kill_switch FROM risk_snapshots ORDER BY timestamp DESC LIMIT 1",
      "legendFormat": "Kill-Switch"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "mappings": [
        {"type": "value", "value": 0, "text": " OK"},
        {"type": "value", "value": 1, "text": " ACTIF"}
      ]
    }
  }
}
```

### Panneau 5: Historique Trades

```json
{
  "title": " Historique Trades",
  "type": "timeseries",
  "datasource": "TimescaleDB",
  "targets": [
    {
      "rawSql": "SELECT time_bucket('1 hour', closed_at) as time, SUM(profit) as profit FROM trade_orders WHERE closed_at IS NOT NULL AND closed_at >= NOW() - INTERVAL '7 days' GROUP BY time ORDER BY time",
      "format": "time_series"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "currencyUSD",
      "custom": {
        "drawStyle": "bars",
        "barAlignment": 0
      },
      "thresholds": {
        "mode": "absolute",
        "steps": [
          {"color": "red", "value": 0},
          {"color": "green", "value": 0}
        ]
      }
    }
  }
}
```

---

##  Dashboard Sécurité

### Panneau 1: Alertes Récentes

```json
{
  "title": " Alertes Sécurité (24h)",
  "type": "table",
  "datasource": "TimescaleDB",
  "targets": [
    {
      "rawSql": "SELECT timestamp, event_type, source_ip, severity, action_taken FROM security_events WHERE timestamp >= NOW() - INTERVAL '24 hours' ORDER BY timestamp DESC LIMIT 20",
      "format": "table"
    }
  ],
  "fieldConfig": {
    "overrides": [
      {
        "matcher": {"id": "byName", "options": "severity"},
        "properties": [
          {
            "id": "mappings",
            "value": [
              {"type": "value", "value": "critical", "text": " CRITIQUE"},
              {"type": "value", "value": "high", "text": " HAUTE"},
              {"type": "value", "value": "medium", "text": " MOYENNE"},
              {"type": "value", "value": "low", "text": " BASSE"}
            ]
          }
        ]
      }
    ]
  }
}
```

### Panneau 2: IPs Bloquées

```json
{
  "title": " IPs Bloquées Actives",
  "type": "stat",
  "datasource": "TimescaleDB",
  "targets": [
    {
      "rawSql": "SELECT COUNT(DISTINCT source_ip) FROM security_events WHERE action_taken = 'IP_BLOCKED' AND timestamp >= NOW() - INTERVAL '1 hour'",
      "format": "table"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "thresholds": {
        "steps": [
          {"color": "green", "value": 0},
          {"color": "yellow", "value": 3},
          {"color": "red", "value": 10}
        ]
      }
    }
  }
}
```

### Panneau 3: Intégrité Kernel

```json
{
  "title": " Intégrité Kernel",
  "type": "stat",
  "datasource": "Prometheus",
  "targets": [
    {
      "expr": "eva_kernel_integrity_check",
      "legendFormat": "Kernel"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "mappings": [
        {"type": "value", "value": 1, "text": " VÉRIFIÉ"},
        {"type": "value", "value": 0, "text": " COMPROMIS"}
      ],
      "thresholds": {
        "steps": [
          {"color": "red", "value": 0},
          {"color": "green", "value": 1}
        ]
      }
    }
  }
}
```

---

##  Dashboard EVA Core

### Panneau 1: Requêtes par Minute

```json
{
  "title": " Requêtes API/min",
  "type": "timeseries",
  "datasource": "Prometheus",
  "targets": [
    {
      "expr": "rate(eva_core_requests_total[1m]) * 60",
      "legendFormat": "Requêtes"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "reqpm"
    }
  }
}
```

### Panneau 2: Latence Inférence LLM

```json
{
  "title": "⏱ Latence LLM",
  "type": "timeseries",
  "datasource": "Prometheus",
  "targets": [
    {
      "expr": "histogram_quantile(0.50, rate(eva_llm_inference_seconds_bucket[5m]))",
      "legendFormat": "p50"
    },
    {
      "expr": "histogram_quantile(0.95, rate(eva_llm_inference_seconds_bucket[5m]))",
      "legendFormat": "p95"
    },
    {
      "expr": "histogram_quantile(0.99, rate(eva_llm_inference_seconds_bucket[5m]))",
      "legendFormat": "p99"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "s",
      "thresholds": {
        "steps": [
          {"color": "green", "value": 0},
          {"color": "yellow", "value": 2},
          {"color": "red", "value": 5}
        ]
      }
    }
  }
}
```

### Panneau 3: Classification Intent

```json
{
  "title": " Distribution Intents",
  "type": "piechart",
  "datasource": "Prometheus",
  "targets": [
    {
      "expr": "sum by (intent) (eva_intent_classification_total)",
      "legendFormat": "{{intent}}"
    }
  ]
}
```

---

##  Provisioning Grafana

### Fichier datasources.yaml

```yaml
# config/grafana/provisioning/datasources/datasources.yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
    
  - name: TimescaleDB
    type: postgres
    url: timescaledb:5432
    database: thehive
    user: eva
    secureJsonData:
      password: ${TIMESCALE_PASSWORD}
    jsonData:
      sslmode: disable
      maxOpenConns: 10
      postgresVersion: 1500
      timescaledb: true
```

### Fichier dashboards.yaml

```yaml
# config/grafana/provisioning/dashboards/dashboards.yaml
apiVersion: 1
providers:
  - name: 'THE HIVE'
    orgId: 1
    folder: 'THE HIVE'
    folderUid: 'thehive'
    type: file
    disableDeletion: true
    updateIntervalSeconds: 30
    options:
      path: /etc/grafana/provisioning/dashboards/json
```

---

##  Alertes Grafana

### Alerte GPU Température

```yaml
# Alerte: GPU surchauffe
- alert: GPUTemperatureCritique
  expr: nvidia_smi_temperature_gpu > 90
  for: 30s
  labels:
    severity: critical
    service: system
  annotations:
    summary: " GPU température critique: {{ $value }}°C"
    description: "La température GPU dépasse 90°C depuis 30s. Loi 0 - Intégrité Systémique"
```

### Alerte Trading Drawdown

```yaml
# Alerte: Drawdown journalier proche limite
- alert: DrawdownJournalierWarning
  expr: eva_daily_drawdown_percent > 3.5
  for: 1m
  labels:
    severity: warning
    service: trading
  annotations:
    summary: " Drawdown journalier: {{ $value }}%"
    description: "Proche de la limite de 4%. Loi 2 - Protection du Capital"

- alert: DrawdownJournalierCritique
  expr: eva_daily_drawdown_percent >= 4.0
  for: 0s
  labels:
    severity: critical
    service: trading
  annotations:
    summary: " KILL-SWITCH DÉCLENCHÉ: {{ $value }}%"
    description: "Limite 4% atteinte. Toutes positions fermées. Trading désactivé."
```
