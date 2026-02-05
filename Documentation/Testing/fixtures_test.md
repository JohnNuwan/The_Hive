# Fixtures de Test - THE HIVE

> **Version**: 1.0.0  
> **Usage**: Données de test pour développement et CI/CD

---

## 📋 Vue d'Ensemble

Ce fichier contient les données de test standardisées pour tous les composants.

---

## 💰 Fixtures Trading

### Comptes de Test

```python
# fixtures/trading/accounts.py
from decimal import Decimal
from uuid import UUID
from datetime import datetime

COMPTE_DEMO_FTMO = {
    "id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
    "nom": "FTMO Challenge Demo",
    "login": 12345678,
    "serveur": "FTMO-Demo",
    "broker": "FTMO",
    "phase": "challenge",
    "balance_initiale": Decimal("100000.00"),
    "balance_actuelle": Decimal("102500.00"),
    "equity": Decimal("102800.00"),
    "drawdown_journalier_percent": Decimal("1.5"),
    "drawdown_total_percent": Decimal("2.5"),
    "copy_enabled": True,
    "actif": True,
    "date_creation": datetime(2026, 2, 1, 9, 0, 0),
}

COMPTE_DEMO_FUNDED = {
    "id": UUID("b2c3d4e5-f6a7-8901-bcde-f23456789012"),
    "nom": "FundedNext Funded",
    "login": 87654321,
    "serveur": "FundedNext-Live",
    "broker": "FundedNext",
    "phase": "funded",
    "balance_initiale": Decimal("50000.00"),
    "balance_actuelle": Decimal("51200.00"),
    "equity": Decimal("51350.00"),
    "drawdown_journalier_percent": Decimal("0.8"),
    "drawdown_total_percent": Decimal("1.2"),
    "copy_enabled": True,
    "actif": True,
    "date_creation": datetime(2026, 1, 15, 10, 30, 0),
}

# Compte en violation pour tests d'erreur
COMPTE_VIOLATION_DRAWDOWN = {
    "id": UUID("c3d4e5f6-a7b8-9012-cdef-345678901234"),
    "nom": "Test Violation",
    "login": 11111111,
    "serveur": "Test-Server",
    "broker": "Test",
    "phase": "challenge",
    "balance_initiale": Decimal("100000.00"),
    "balance_actuelle": Decimal("96000.00"),  # -4% = limite
    "equity": Decimal("95800.00"),
    "drawdown_journalier_percent": Decimal("4.0"),  # LIMITE!
    "drawdown_total_percent": Decimal("4.2"),
    "copy_enabled": False,
    "actif": False,  # Désactivé par Kill-Switch
    "date_creation": datetime(2026, 1, 1, 8, 0, 0),
}
```

### Ordres de Test

```python
# fixtures/trading/orders.py
from decimal import Decimal
from uuid import UUID
from datetime import datetime

ORDRE_VALIDE_BUY_GOLD = {
    "id": UUID("d4e5f6a7-b8c9-0123-def0-456789012345"),
    "symbol": "XAUUSD",
    "action": "BUY",
    "volume": Decimal("0.50"),
    "stop_loss_price": Decimal("2050.00"),
    "take_profit_price": Decimal("2120.00"),
    "order_type": "MARKET",
    "source": "VOICE",
    "account_id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
    "risque_calcule_percent": Decimal("0.85"),
    "created_at": datetime.now(),
}

ORDRE_RISQUE_EXCESSIF = {
    "id": UUID("e5f6a7b8-c9d0-1234-ef01-567890123456"),
    "symbol": "XAUUSD",
    "action": "BUY",
    "volume": Decimal("2.00"),  # Trop gros
    "stop_loss_price": Decimal("2000.00"),  # SL loin
    "take_profit_price": None,
    "order_type": "MARKET",
    "source": "CHAT",
    "account_id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
    "risque_calcule_percent": Decimal("3.5"),  # > 1% = REJET
    "created_at": datetime.now(),
}

ORDRE_SANS_STOP_LOSS = {
    "id": UUID("f6a7b8c9-d0e1-2345-f012-678901234567"),
    "symbol": "EURUSD",
    "action": "SELL",
    "volume": Decimal("0.10"),
    "stop_loss_price": None,  # INTERDIT par ROE
    "take_profit_price": Decimal("1.0800"),
    "order_type": "MARKET",
    "source": "API",
    "account_id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
    "created_at": datetime.now(),
}
```

### Positions Ouvertes

```python
# fixtures/trading/positions.py

POSITION_GOLD_EN_PROFIT = {
    "ticket": 12345678,
    "symbol": "XAUUSD",
    "action": "BUY",
    "volume": Decimal("0.50"),
    "open_price": Decimal("2080.00"),
    "current_price": Decimal("2095.00"),
    "stop_loss": Decimal("2050.00"),
    "take_profit": Decimal("2120.00"),
    "profit": Decimal("750.00"),
    "swap": Decimal("-12.50"),
    "commission": Decimal("-3.50"),
    "magic_number": 12345,
    "open_time": datetime(2026, 2, 5, 10, 30, 0),
}

POSITION_EURO_EN_PERTE = {
    "ticket": 12345679,
    "symbol": "EURUSD",
    "action": "SELL",
    "volume": Decimal("0.20"),
    "open_price": Decimal("1.0850"),
    "current_price": Decimal("1.0875"),
    "stop_loss": Decimal("1.0900"),
    "take_profit": Decimal("1.0750"),
    "profit": Decimal("-50.00"),
    "swap": Decimal("-2.00"),
    "commission": Decimal("-1.40"),
    "magic_number": 12345,
    "open_time": datetime(2026, 2, 5, 9, 15, 0),
}
```

---

## 🧠 Fixtures EVA Core

### Messages Chat

```python
# fixtures/core/messages.py

MESSAGE_INTENT_TRADING = {
    "id": UUID("a1a1a1a1-b2b2-c3c3-d4d4-e5e5e5e5e5e5"),
    "session_id": UUID("11111111-2222-3333-4444-555555555555"),
    "content": "Achète 0.5 lot de Gold avec un stop loss à 2050",
    "role": "user",
    "timestamp": datetime.now(),
    "metadata": {
        "source": "nexus_mobile",
        "language": "fr"
    }
}

MESSAGE_INTENT_CHAT = {
    "id": UUID("b2b2b2b2-c3c3-d4d4-e5e5-f6f6f6f6f6f6"),
    "session_id": UUID("11111111-2222-3333-4444-555555555555"),
    "content": "Comment ça va aujourd'hui EVA ?",
    "role": "user",
    "timestamp": datetime.now(),
    "metadata": {
        "source": "nexus_desktop",
        "language": "fr"
    }
}

MESSAGE_INTENT_MEMOIRE = {
    "id": UUID("c3c3c3c3-d4d4-e5e5-f6f6-a7a7a7a7a7a7"),
    "session_id": UUID("11111111-2222-3333-4444-555555555555"),
    "content": "Rappelle-moi notre discussion sur le Gold de la semaine dernière",
    "role": "user",
    "timestamp": datetime.now(),
    "metadata": {
        "source": "voice",
        "language": "fr"
    }
}

REPONSE_EVA_TRADING = {
    "id": UUID("d4d4d4d4-e5e5-f6f6-a7a7-b8b8b8b8b8b8"),
    "session_id": UUID("11111111-2222-3333-4444-555555555555"),
    "content": "J'ai transmis l'ordre à The Banker. L'ordre d'achat de 0.5 lot de XAUUSD avec SL à 2050 a été exécuté avec succès. Ticket #12345678.",
    "role": "assistant",
    "timestamp": datetime.now(),
    "metadata": {
        "expert": "banker",
        "intent": "TRADING_ORDER",
        "confidence": 0.95
    }
}
```

### Intents Classifiés

```python
# fixtures/core/intents.py

INTENT_TRADING_ORDER = {
    "intent": "TRADING_ORDER",
    "confidence": 0.92,
    "entities": {
        "action": "BUY",
        "symbol": "XAUUSD",
        "volume": 0.5,
        "stop_loss": 2050.0
    },
    "target_expert": "banker"
}

INTENT_POSITION_STATUS = {
    "intent": "POSITION_STATUS",
    "confidence": 0.88,
    "entities": {},
    "target_expert": "banker"
}

INTENT_GENERAL_CHAT = {
    "intent": "GENERAL_CHAT",
    "confidence": 0.95,
    "entities": {},
    "target_expert": "core"
}

INTENT_MEMORY_RECALL = {
    "intent": "MEMORY_RECALL",
    "confidence": 0.85,
    "entities": {
        "topic": "Gold",
        "timeframe": "semaine dernière"
    },
    "target_expert": "core"
}
```

---

## 🛡️ Fixtures Sécurité

### Événements de Sécurité

```python
# fixtures/security/events.py

EVENEMENT_BRUTE_FORCE = {
    "id": UUID("sec11111-2222-3333-4444-555555555555"),
    "timestamp": datetime.now(),
    "event_type": "BRUTE_FORCE_DETECTED",
    "source_ip": "192.168.100.50",
    "target_service": "ssh",
    "severity": "high",
    "details": {
        "attempts": 5,
        "window_seconds": 60,
        "username_tried": ["root", "admin", "eva"]
    },
    "action_taken": "IP_BLOCKED",
    "resolved": True
}

EVENEMENT_PORT_SCAN = {
    "id": UUID("sec22222-3333-4444-5555-666666666666"),
    "timestamp": datetime.now(),
    "event_type": "PORT_SCAN_DETECTED",
    "source_ip": "10.0.2.100",
    "target_service": "multiple",
    "severity": "medium",
    "details": {
        "ports_scanned": [22, 80, 443, 3389, 5432, 6379],
        "scan_type": "SYN"
    },
    "action_taken": "LOGGED",
    "resolved": False
}

EVENEMENT_KERNEL_INTEGRITY_OK = {
    "id": UUID("sec33333-4444-5555-6666-777777777777"),
    "timestamp": datetime.now(),
    "event_type": "KERNEL_INTEGRITY_CHECK",
    "source_ip": "localhost",
    "target_service": "kernel",
    "severity": "info",
    "details": {
        "hash_expected": "abc123...",
        "hash_actual": "abc123...",
        "match": True
    },
    "action_taken": "NONE",
    "resolved": True
}
```

---

## ⚙️ Fixtures Système

### Métriques Hardware

```python
# fixtures/system/metrics.py

METRIQUES_GPU_NORMAL = {
    "timestamp": datetime.now(),
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "temperature_celsius": 65.0,
    "utilization_percent": 45.0,
    "memory_used_mb": 8500,
    "memory_total_mb": 24576,
    "power_draw_watts": 180.0,
    "fan_speed_percent": 40
}

METRIQUES_GPU_CHAUD = {
    "timestamp": datetime.now(),
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "temperature_celsius": 85.0,  # Warning!
    "utilization_percent": 98.0,
    "memory_used_mb": 22000,
    "memory_total_mb": 24576,
    "power_draw_watts": 350.0,
    "fan_speed_percent": 100
}

METRIQUES_CPU_RAM = {
    "timestamp": datetime.now(),
    "cpu_percent": 35.0,
    "cpu_freq_mhz": 3800,
    "ram_used_gb": 48.5,
    "ram_total_gb": 128.0,
    "swap_used_gb": 0.0,
    "load_average_1m": 2.5
}
```

---

## 🔧 Utilisation dans les Tests

### Exemple pytest

```python
# tests/test_trading_risk.py
import pytest
from fixtures.trading.orders import ORDRE_VALIDE_BUY_GOLD, ORDRE_RISQUE_EXCESSIF
from fixtures.trading.accounts import COMPTE_DEMO_FTMO
from eva_banker.risk import RiskValidator

@pytest.fixture
def validateur_risque():
    return RiskValidator(max_risque_percent=1.0)

@pytest.fixture
def compte_demo():
    return COMPTE_DEMO_FTMO

def test_ordre_valide_approuve(validateur_risque, compte_demo):
    """Un ordre avec risque < 1% doit être approuvé"""
    resultat = validateur_risque.valider(ORDRE_VALIDE_BUY_GOLD, compte_demo)
    
    assert resultat.approuve == True
    assert resultat.raison is None

def test_ordre_risque_excessif_rejete(validateur_risque, compte_demo):
    """Un ordre avec risque > 1% doit être rejeté (Loi 2)"""
    resultat = validateur_risque.valider(ORDRE_RISQUE_EXCESSIF, compte_demo)
    
    assert resultat.approuve == False
    assert resultat.code_erreur == "EVA-TRD-002"
    assert "Loi 2" in resultat.reference_constitution
```

### Fichier conftest.py

```python
# tests/conftest.py
import pytest
from fixtures.trading.accounts import COMPTE_DEMO_FTMO, COMPTE_DEMO_FUNDED
from fixtures.core.messages import MESSAGE_INTENT_TRADING

@pytest.fixture
def tous_comptes():
    """Retourne tous les comptes de test"""
    return [COMPTE_DEMO_FTMO, COMPTE_DEMO_FUNDED]

@pytest.fixture
def message_trading():
    """Message utilisateur pour intent trading"""
    return MESSAGE_INTENT_TRADING.copy()
```
