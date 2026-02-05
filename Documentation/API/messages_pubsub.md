# Format des Messages Inter-Agents (Redis Pub/Sub)

> **Version**: 1.0.0  
> **Transport**: Redis Pub/Sub (Channels)  
> **Encoding**: JSON (UTF-8)

---

## 🔄 Architecture de Communication

```
┌─────────────┐        ┌───────────────┐        ┌─────────────┐
│   EVA Core  │───────▶│  Redis Broker │◀───────│  The Banker │
│  (Producer) │        │  (Pub/Sub)    │        │  (Consumer) │
└─────────────┘        └───────────────┘        └─────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  The Sentinel   │
                    │  (Subscriber)   │
                    └─────────────────┘
```

### Channels (Topics)

| Channel | Description | Producers | Consumers |
|---------|-------------|-----------|-----------|
| `eva.core.requests` | Requêtes vers le Core | Tous agents | Core |
| `eva.banker.requests` | Requêtes trading | Core | Banker |
| `eva.banker.responses` | Réponses trading | Banker | Core |
| `eva.shadow.requests` | Requêtes OSINT | Core | Shadow |
| `eva.sentinel.alerts` | Alertes sécurité | Sentinel | Tous |
| `eva.system.metrics` | Métriques système | Keeper | Tous |
| `eva.audit` | Audit trail | Tous | BlackBox |

---

## 📨 Format des Messages

### Structure de Base

Tous les messages suivent cette structure :

```json
{
  "id": "uuid-v4",
  "type": "request | response | event | alert",
  "source": "agent_name",
  "target": "agent_name | broadcast",
  "action": "action_name",
  "payload": { ... },
  "metadata": {
    "correlation_id": "uuid-v4",
    "priority": 1-5,
    "timestamp": "ISO 8601",
    "ttl_seconds": 30,
    "requires_response": true
  }
}
```

### Champs Obligatoires

| Champ | Type | Description |
|-------|------|-------------|
| `id` | UUID v4 | Identifiant unique du message |
| `type` | string | Type de message |
| `source` | string | Agent émetteur |
| `action` | string | Action à effectuer |
| `payload` | object | Données du message |
| `metadata.timestamp` | string | Horodatage ISO 8601 |

---

## 🏦 Messages The Banker

### TRADE_EXECUTE (Core → Banker)

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "request",
  "source": "core",
  "target": "banker",
  "action": "TRADE_EXECUTE",
  "payload": {
    "symbol": "XAUUSD",
    "action": "BUY",
    "volume": 0.5,
    "stop_loss_price": 2050.00,
    "take_profit_price": 2100.00,
    "order_type": "MARKET",
    "source": "manual",
    "user_context": {
      "mood": "focused",
      "sleep_hours": 8.0
    }
  },
  "metadata": {
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "priority": 1,
    "timestamp": "2026-02-05T10:30:00Z",
    "ttl_seconds": 10,
    "requires_response": true
  }
}
```

### TRADE_RESULT (Banker → Core)

```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "type": "response",
  "source": "banker",
  "target": "core",
  "action": "TRADE_RESULT",
  "payload": {
    "success": true,
    "ticket": 1234567890,
    "symbol": "XAUUSD",
    "action": "BUY",
    "volume": 0.5,
    "entry_price": 2075.50,
    "stop_loss": 2050.00,
    "take_profit": 2100.00,
    "slippage_points": 2,
    "execution_time_ms": 45
  },
  "metadata": {
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "priority": 1,
    "timestamp": "2026-02-05T10:30:00.050Z",
    "requires_response": false
  }
}
```

### TRADE_REJECTED (Banker → Core)

```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "type": "response",
  "source": "banker",
  "target": "core",
  "action": "TRADE_REJECTED",
  "payload": {
    "success": false,
    "rejection_reason": "DAILY_LOSS_LIMIT",
    "details": "Daily drawdown at 3.95%, limit is 4%",
    "constitution_reference": "Law 2 - Protection du Capital",
    "current_risk_status": {
      "daily_drawdown_percent": 3.95,
      "total_drawdown_percent": 5.2,
      "trading_allowed": false
    }
  },
  "metadata": {
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "priority": 1,
    "timestamp": "2026-02-05T10:30:00.020Z",
    "requires_response": false
  }
}
```

### RISK_CHECK (Core → Banker)

```json
{
  "id": "880e8400-e29b-41d4-a716-446655440003",
  "type": "request",
  "source": "core",
  "target": "banker",
  "action": "RISK_CHECK",
  "payload": {
    "symbol": "XAUUSD",
    "stop_loss_distance_pips": 50,
    "risk_percent": 1.0
  },
  "metadata": {
    "correlation_id": "880e8400-e29b-41d4-a716-446655440003",
    "priority": 2,
    "timestamp": "2026-02-05T10:29:00Z",
    "ttl_seconds": 5,
    "requires_response": true
  }
}
```

---

## 🔍 Messages The Shadow (OSINT)

### OSINT_SEARCH (Core → Shadow)

```json
{
  "id": "990e8400-e29b-41d4-a716-446655440004",
  "type": "request",
  "source": "core",
  "target": "shadow",
  "action": "OSINT_SEARCH",
  "payload": {
    "query": "Jean Dupont Paris",
    "search_type": "person",
    "sources": ["google", "linkedin", "social"],
    "depth": "shallow",
    "max_results": 10
  },
  "metadata": {
    "correlation_id": "990e8400-e29b-41d4-a716-446655440004",
    "priority": 3,
    "timestamp": "2026-02-05T10:35:00Z",
    "ttl_seconds": 60,
    "requires_response": true
  }
}
```

### OSINT_RESULT (Shadow → Core)

```json
{
  "id": "aa0e8400-e29b-41d4-a716-446655440005",
  "type": "response",
  "source": "shadow",
  "target": "core",
  "action": "OSINT_RESULT",
  "payload": {
    "query": "Jean Dupont Paris",
    "results": [
      {
        "source": "linkedin",
        "content": "Jean Dupont - CEO at TechCorp",
        "url": "https://linkedin.com/in/jeandupont",
        "credibility_score": 0.95
      }
    ],
    "total_found": 15,
    "search_duration_ms": 2500
  },
  "metadata": {
    "correlation_id": "990e8400-e29b-41d4-a716-446655440004",
    "priority": 3,
    "timestamp": "2026-02-05T10:35:02.500Z",
    "requires_response": false
  }
}
```

---

## 🛡️ Messages The Sentinel (Security)

### SECURITY_ALERT (Sentinel → Broadcast)

```json
{
  "id": "bb0e8400-e29b-41d4-a716-446655440006",
  "type": "alert",
  "source": "sentinel",
  "target": "broadcast",
  "action": "SECURITY_ALERT",
  "payload": {
    "event_type": "intrusion_attempt",
    "severity": "high",
    "source_ip": "192.168.1.100",
    "target_service": "ssh",
    "description": "5 failed login attempts in 60 seconds",
    "action_taken": "blocked",
    "block_duration_seconds": 3600
  },
  "metadata": {
    "priority": 1,
    "timestamp": "2026-02-05T10:40:00Z",
    "requires_response": false
  }
}
```

### KILL_SWITCH_TRIGGERED (Sentinel → Broadcast)

```json
{
  "id": "cc0e8400-e29b-41d4-a716-446655440007",
  "type": "alert",
  "source": "sentinel",
  "target": "broadcast",
  "action": "KILL_SWITCH_TRIGGERED",
  "payload": {
    "reason": "DAILY_LOSS_LIMIT",
    "account_state": {
      "login": 12345678,
      "balance": 10000.00,
      "equity": 9600.00,
      "loss_percent": 4.0
    },
    "actions_taken": [
      "CLOSE_ALL_POSITIONS",
      "DISABLE_TRADING",
      "NOTIFY_ADMIN"
    ]
  },
  "metadata": {
    "priority": 1,
    "timestamp": "2026-02-05T10:45:00Z",
    "requires_response": false
  }
}
```

---

## 📊 Messages System (Keeper)

### SYSTEM_METRICS (Keeper → Broadcast)

```json
{
  "id": "dd0e8400-e29b-41d4-a716-446655440008",
  "type": "event",
  "source": "keeper",
  "target": "broadcast",
  "action": "SYSTEM_METRICS",
  "payload": {
    "gpu": {
      "temperature_celsius": 72.0,
      "power_watts": 280,
      "memory_used_mb": 18500,
      "memory_total_mb": 24576,
      "utilization_percent": 85
    },
    "cpu_load_percent": 45.0,
    "ram_used_mb": 98000,
    "ram_total_mb": 131072,
    "disk_used_gb": 450,
    "disk_total_gb": 1000
  },
  "metadata": {
    "priority": 5,
    "timestamp": "2026-02-05T10:50:00Z",
    "requires_response": false
  }
}
```

### THERMAL_ALERT (Keeper → Broadcast)

```json
{
  "id": "ee0e8400-e29b-41d4-a716-446655440009",
  "type": "alert",
  "source": "keeper",
  "target": "broadcast",
  "action": "THERMAL_ALERT",
  "payload": {
    "level": "warning",
    "current_temp_celsius": 85.0,
    "threshold_celsius": 80.0,
    "duration_above_threshold_secs": 30,
    "action_recommended": "REDUCE_LOAD"
  },
  "metadata": {
    "priority": 2,
    "timestamp": "2026-02-05T10:55:00Z",
    "requires_response": false
  }
}
```

---

## 📝 Messages Audit (Black Box)

### AUDIT_RECORD (Any → Audit)

```json
{
  "id": "ff0e8400-e29b-41d4-a716-446655440010",
  "type": "event",
  "source": "banker",
  "target": "audit",
  "action": "AUDIT_RECORD",
  "payload": {
    "event_type": "TRADE_EXECUTED",
    "actor": "banker",
    "action": "execute_market_order",
    "target": "XAUUSD",
    "old_value": null,
    "new_value": {
      "ticket": 1234567890,
      "volume": 0.5,
      "price": 2075.50
    },
    "context": {
      "account_equity": 10250.00,
      "daily_pnl": 125.50
    }
  },
  "metadata": {
    "priority": 3,
    "timestamp": "2026-02-05T11:00:00Z",
    "requires_response": false
  }
}
```

---

## ⚠️ Gestion des Erreurs

### ERROR Message

```json
{
  "id": "error-uuid",
  "type": "response",
  "source": "banker",
  "target": "core",
  "action": "ERROR",
  "payload": {
    "error_code": "MT5_CONNECTION_LOST",
    "error_message": "Unable to connect to MT5 terminal",
    "original_request_id": "550e8400-e29b-41d4-a716-446655440000",
    "recoverable": true,
    "retry_after_seconds": 5
  },
  "metadata": {
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "priority": 1,
    "timestamp": "2026-02-05T11:05:00Z",
    "requires_response": false
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `MT5_CONNECTION_LOST` | Connexion MT5 perdue |
| `INSUFFICIENT_MARGIN` | Marge insuffisante |
| `MARKET_CLOSED` | Marché fermé |
| `SYMBOL_NOT_FOUND` | Symbole inconnu |
| `TIMEOUT` | Timeout de requête |
| `VALIDATION_FAILED` | Validation échouée |
| `KERNEL_REJECTION` | Rejeté par le Kernel |

---

## 🔧 Implémentation Python

```python
import json
import redis
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel

class MessageMetadata(BaseModel):
    correlation_id: str
    priority: int = 3
    timestamp: str
    ttl_seconds: int = 30
    requires_response: bool = True

class AgentMessage(BaseModel):
    id: str
    type: str  # request, response, event, alert
    source: str
    target: str
    action: str
    payload: dict
    metadata: MessageMetadata

class MessageBus:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()
    
    def publish(self, channel: str, message: AgentMessage) -> None:
        self.redis.publish(channel, message.model_dump_json())
    
    def subscribe(self, channel: str, callback) -> None:
        self.pubsub.subscribe(**{channel: callback})
        self.pubsub.run_in_thread(sleep_time=0.01)
```
