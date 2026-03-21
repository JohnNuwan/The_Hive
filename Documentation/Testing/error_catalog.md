# Catalogue des Erreurs - THE HIVE

> **Version**: 1.0.0  
> **Format**: Code, Message, Sévérité, Action Corrective

---

##  Structure des Erreurs

```json
{
  "error_code": "EVA-XXX-NNN",
  "error_message": "Message lisible par un humain",
  "severity": "info|warning|error|critical",
  "category": "CORE|TRADING|SECURITY|SYSTEM|NETWORK",
  "recoverable": true,
  "retry_strategy": "none|immediate|exponential",
  "action": "Description de l'action corrective"
}
```

---

##  Erreurs EVA Core (EVA-COR-XXX)

| Code | Message | Sévérité | Récupérable | Action |
|------|---------|----------|-------------|--------|
| `EVA-COR-001` | Serveur LLM non répondant | critical |  | Redémarrer vLLM, vérifier GPU |
| `EVA-COR-002` | Échec classification intent | warning |  | Fallback vers GENERAL_CHAT |
| `EVA-COR-003` | Timeout recherche mémoire | warning |  | Retry avec cache local |
| `EVA-COR-004` | ID de session invalide | error |  | Créer nouvelle session |
| `EVA-COR-005` | Échec routage agent | error |  | Retry direct vers Core |
| `EVA-COR-006` | Limite de tokens dépassée | warning |  | Tronquer le contexte |
| `EVA-COR-007` | Connexion Qdrant perdue | critical |  | Retry + alerte admin |
| `EVA-COR-008` | Échec pub/sub Redis | critical |  | Reconnexion auto |

### Détails EVA-COR-001
```json
{
  "error_code": "EVA-COR-001",
  "error_message": "Serveur LLM non répondant après {timeout}ms",
  "severity": "critical",
  "category": "CORE",
  "recoverable": true,
  "retry_strategy": "exponential",
  "retry_max_attempts": 3,
  "retry_base_delay_ms": 1000,
  "action": "1. Vérifier nvidia-smi pour statut GPU\n2. Redémarrer vLLM: systemctl restart vllm\n3. Si persiste, reboot VM eva-core",
  "escalation": "Si non résolu après 5 min, notifier admin via Discord"
}
```

---

##  Erreurs Trading (EVA-TRD-XXX)

| Code | Message | Sévérité | Récupérable | Action |
|------|---------|----------|-------------|--------|
| `EVA-TRD-001` | Connexion MT5 perdue | critical |  | Reconnexion auto, pause trading |
| `EVA-TRD-002` | Limite de risque dépassée | error |  | Rejeter ordre, log Constitution |
| `EVA-TRD-003` | Limite drawdown journalier | critical |  | Kill-Switch, fermer tout |
| `EVA-TRD-004` | Limite drawdown total | critical |  | Kill-Switch, désactiver trading |
| `EVA-TRD-005` | Anti-tilt déclenché | warning |  | Pause 24h |
| `EVA-TRD-006` | Filtre news actif | info |  | Attendre fin période |
| `EVA-TRD-007` | Marge insuffisante | error |  | Réduire taille position |
| `EVA-TRD-008` | Marché fermé | info |  | Ordre en attente |
| `EVA-TRD-009` | Échec exécution ordre | error |  | Retry (max 2x) |
| `EVA-TRD-010` | Symbole non autorisé | error |  | Ajouter à whitelist si légitime |
| `EVA-TRD-011` | Stop loss manquant | error |  | Rejeter (ROE: SL obligatoire) |
| `EVA-TRD-012` | Slippage excessif | warning |  | Log, ajuster EA |
| `EVA-TRD-013` | Échec copie Hydra | error |  | Retry sur compte spécifique |
| `EVA-TRD-014` | Violation règle Prop Firm | critical |  | Alerte immédiate, review trade |

### Détails EVA-TRD-003
```json
{
  "error_code": "EVA-TRD-003",
  "error_message": "Limite drawdown journalier atteinte: {current}% >= {limit}%",
  "severity": "critical",
  "category": "TRADING",
  "recoverable": false,
  "retry_strategy": "none",
  "action": "1. Kill-Switch: Fermer toutes les positions\n2. Désactiver le trading pour la journée\n3. Notifier admin via tous les canaux\n4. Logger dans Black Box audit trail",
  "constitution_reference": "Loi 2 - Protection du Capital",
  "auto_actions": ["CLOSE_ALL_POSITIONS", "DISABLE_TRADING", "NOTIFY_ADMIN"]
}
```

---

##  Erreurs Sécurité (EVA-SEC-XXX)

| Code | Message | Sévérité | Récupérable | Action |
|------|---------|----------|-------------|--------|
| `EVA-SEC-001` | Brute force détecté | high |  | Bloquer IP, alerter |
| `EVA-SEC-002` | Échec intégrité Kernel | critical |  | ARRÊT système |
| `EVA-SEC-003` | Constitution altérée | critical |  | ARRÊT système |
| `EVA-SEC-004` | Accès non autorisé | high |  | Bloquer + enquêter |
| `EVA-SEC-005` | HSM non répondant | critical |  | Retry, alerter admin |
| `EVA-SEC-006` | Tablet non montée | critical |  | Refuser démarrage |
| `EVA-SEC-007` | Scan de ports détecté | medium |  | Logger, bloquer optionnel |
| `EVA-SEC-008` | Signature malware | critical |  | Quarantaine, alerter |
| `EVA-SEC-009` | Certificat SSL expiré | warning |  | Renouveler certificat |
| `EVA-SEC-010` | Tailscale déconnecté | warning |  | Reconnexion auto |

### Détails EVA-SEC-002
```json
{
  "error_code": "EVA-SEC-002",
  "error_message": "Hash binaire Kernel non conforme: attendu {expected}, obtenu {actual}",
  "severity": "critical",
  "category": "SECURITY",
  "recoverable": false,
  "retry_strategy": "none",
  "action": "1. ARRÊTER toutes les opérations immédiatement\n2. Ne PAS rebooter automatiquement\n3. Intervention physique admin requise\n4. Booter depuis backup connu",
  "constitution_reference": "Loi 0 - Intégrité Systémique",
  "requires_physical_access": true
}
```

---

##  Erreurs Système (EVA-SYS-XXX)

| Code | Message | Sévérité | Récupérable | Action |
|------|---------|----------|-------------|--------|
| `EVA-SYS-001` | Avertissement température GPU | warning |  | Réduire charge |
| `EVA-SYS-002` | Température GPU critique | critical |  | Arrêt gracieux |
| `EVA-SYS-003` | Espace disque faible | warning |  | Nettoyage, alerter |
| `EVA-SYS-004` | Pression mémoire | warning |  | Tuer tâches basse priorité |
| `EVA-SYS-005` | Pool connexions DB épuisé | error |  | Étendre pool, retry |
| `EVA-SYS-006` | Timeout watchdog | critical |  | Reset par ESP32 |
| `EVA-SYS-007` | Échec sauvegarde | error |  | Retry, alerter si 3x échec |
| `EVA-SYS-008` | Container OOM killed | error |  | Restart, augmenter limites |

### Détails EVA-SYS-002
```json
{
  "error_code": "EVA-SYS-002",
  "error_message": "Température GPU critique: {temp}°C > {threshold}°C pendant {duration}s",
  "severity": "critical",
  "category": "SYSTEM",
  "recoverable": true,
  "retry_strategy": "none",
  "action": "1. Arrêter toutes les charges GPU immédiatement\n2. Si temp > 95°C, initier arrêt d'urgence\n3. Attendre refroidissement avant redémarrage\n4. Vérifier ventilateurs et pâte thermique",
  "constitution_reference": "Loi 0 - Intégrité Systémique",
  "thresholds": {
    "warning": 80,
    "critical": 90,
    "emergency": 95
  }
}
```

---

##  Erreurs Réseau (EVA-NET-XXX)

| Code | Message | Sévérité | Récupérable | Action |
|------|---------|----------|-------------|--------|
| `EVA-NET-001` | Timeout API | warning |  | Retry avec backoff |
| `EVA-NET-002` | Échec résolution DNS | error |  | Utiliser DNS fallback |
| `EVA-NET-003` | Rate limit atteint | warning |  | Attendre, retry |
| `EVA-NET-004` | Échec handshake TLS | error |  | Vérifier certificats |
| `EVA-NET-005` | API externe indisponible | warning |  | Utiliser cache si dispo |

---

##  Mapping Codes HTTP

| Code HTTP | Code Erreur | Signification |
|-----------|-------------|---------------|
| 400 | `EVA-*-0XX` | Requête invalide - Erreur de validation |
| 401 | `EVA-SEC-004` | Non autorisé |
| 403 | `EVA-TRD-002` | Interdit - Violation Risque/Constitution |
| 404 | - | Ressource non trouvée |
| 429 | `EVA-NET-003` | Limite de requêtes atteinte |
| 500 | `EVA-COR-*` | Erreur serveur interne |
| 503 | `EVA-SYS-*` | Service indisponible |

---

##  Matrice d'Escalation

| Sévérité | Notification | Délai | Canaux |
|----------|--------------|-------|--------|
| info | Aucune | - | Logs uniquement |
| warning | Optionnelle | 5 min agrégé | Discord (système) |
| error | Requise | Immédiat | Discord (alertes) |
| critical | Requise + Appel | Immédiat | Discord + SMS + Sirène |

---

##  Implémentation Python

```python
from enum import Enum
from pydantic import BaseModel

class SeveriteErreur(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErreurHive(BaseModel):
    code: str
    message: str
    severite: SeveriteErreur
    categorie: str
    recuperable: bool
    details: dict = {}
    
class ErreurRisqueTrading(ErreurHive):
    code: str = "EVA-TRD-002"
    categorie: str = "TRADING"
    recuperable: bool = False
    reference_constitution: str = "Loi 2 - Protection du Capital"
    
    def __init__(self, risque_demande: float, risque_max: float):
        super().__init__(
            message=f"Risque {risque_demande}% dépasse limite {risque_max}%",
            severite=SeveriteErreur.ERROR,
            details={
                "risque_demande": risque_demande,
                "risque_max": risque_max
            }
        )
```
