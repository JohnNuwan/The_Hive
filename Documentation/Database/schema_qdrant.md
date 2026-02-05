# Qdrant Vector Database Schema

> **Version**: 1.0.0  
> **Instance**: `http://10.0.1.100:6333`  
> **Purpose**: Mémoire long-terme vectorielle pour le RAG (Retrieval Augmented Generation)

---

## 📦 Collections

### 1. `conversations` - Mémoire Conversationnelle

Stocke les messages et contextes des conversations pour le rappel contextuel.

```json
{
  "collection_name": "conversations",
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "on_disk_payload": true,
  "optimizers_config": {
    "indexing_threshold": 10000
  }
}
```

#### Payload Schema
```json
{
  "message_id": "uuid",
  "session_id": "uuid",
  "role": "user | assistant | system",
  "content": "string - texte original du message",
  "timestamp": "ISO 8601 datetime",
  "intent": "TRADING_ORDER | GENERAL_CHAT | ...",
  "routed_to": "core | banker | shadow | ...",
  "user_mood": "focused | relaxed | stressed | unknown",
  "summary": "string - résumé généré par LLM si message long"
}
```

#### Index Fields
- `session_id` (keyword) - Pour filtrer par session
- `role` (keyword) - Pour filtrer user/assistant
- `timestamp` (integer) - Pour tri chronologique
- `intent` (keyword) - Pour filtrer par type d'intent

---

### 2. `osint` - Intelligence Gathering

Stocke les résultats de recherche OSINT de The Shadow.

```json
{
  "collection_name": "osint",
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "on_disk_payload": true
}
```

#### Payload Schema
```json
{
  "query_id": "uuid",
  "source": "google | shodan | intelligence_x | dehashed | social",
  "target": "string - cible de la recherche",
  "content": "string - contenu extrait",
  "url": "string - URL source (si applicable)",
  "credibility_score": 0.0-1.0,
  "tags": ["tag1", "tag2"],
  "collected_at": "ISO 8601 datetime",
  "expires_at": "ISO 8601 datetime - pour RGPD/purge",
  "classification": "public | confidential | sensitive"
}
```

#### Index Fields
- `source` (keyword)
- `target` (keyword) 
- `classification` (keyword)
- `collected_at` (integer)
- `expires_at` (integer) - Pour GC automatique

---

### 3. `trading_knowledge` - Savoir Trading

Base de connaissances pour The Banker (patterns, règles, historique d'analyse).

```json
{
  "collection_name": "trading_knowledge",
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  }
}
```

#### Payload Schema
```json
{
  "knowledge_id": "uuid",
  "category": "pattern | rule | analysis | lesson",
  "symbol": "XAUUSD | EURUSD | ...",
  "timeframe": "M1 | M5 | H1 | H4 | D1",
  "content": "string - description de la connaissance",
  "success_rate": 0.0-1.0,
  "sample_size": "integer - nombre de trades",
  "conditions": {
    "market_condition": "trending | ranging",
    "session": "london | newyork | asian"
  },
  "created_by": "algo | admin | researcher",
  "last_validated": "ISO 8601 datetime"
}
```

#### Index Fields
- `category` (keyword)
- `symbol` (keyword)
- `timeframe` (keyword)
- `success_rate` (float)

---

### 4. `documents` - Documents & RAG Général

Stockage de documents techniques, documentation, et ressources textuelles.

```json
{
  "collection_name": "documents",
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "on_disk_payload": true
}
```

#### Payload Schema
```json
{
  "doc_id": "uuid",
  "title": "string",
  "content": "string - chunk de texte",
  "source_file": "string - chemin du fichier original",
  "chunk_index": "integer",
  "total_chunks": "integer",
  "doc_type": "constitution | tech_spec | user_manual | research_paper",
  "tags": ["security", "trading", "ai"],
  "version": "string",
  "indexed_at": "ISO 8601 datetime"
}
```

#### Index Fields
- `doc_type` (keyword)
- `source_file` (keyword)
- `tags` (keyword[])

---

### 5. `security_intel` - Intelligence Sécurité

Base de connaissances de The Sentinel pour la threat intelligence.

```json
{
  "collection_name": "security_intel",
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  }
}
```

#### Payload Schema
```json
{
  "intel_id": "uuid",
  "type": "ioc | technique | vulnerability | actor",
  "name": "string",
  "description": "string",
  "indicators": {
    "ips": ["ip1", "ip2"],
    "domains": ["domain1"],
    "hashes": ["sha256_hash"]
  },
  "severity": "low | medium | high | critical",
  "source": "internal | abuseipdb | virustotal | osint",
  "first_seen": "ISO 8601 datetime",
  "last_seen": "ISO 8601 datetime",
  "ttl_days": "integer - durée de vie avant purge"
}
```

#### Index Fields
- `type` (keyword)
- `severity` (keyword)
- `source` (keyword)
- `last_seen` (integer)

---

## 🔧 Configuration de l'Embedding Model

### Model: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Dimension**: 384 (Note: upgrade vers 1024 avec Llama embedding recommandé)
- **Max Sequence Length**: 256 tokens
- **Performance**: ~4000 embeddings/sec sur CPU

### Alternative recommandée (Production)
- **Model**: `nomic-embed-text-v1` (via Ollama)
- **Dimension**: 768 ou 1024
- **Avantage**: Local, pas de dépendance API externe

---

## 📊 Politiques de Rétention (RGPD Compliance)

| Collection | Rétention | Trigger |
|------------|-----------|---------|
| `conversations` | 6 mois | `expires_at` ou inactivité |
| `osint` | Variable | Champ `expires_at` obligatoire |
| `trading_knowledge` | Permanent | Review annuelle |
| `documents` | Permanent | Versioning |
| `security_intel` | 90 jours default | Champ `ttl_days` |

### GC Script (The Keeper lance ceci quotidiennement)
```python
from qdrant_client import QdrantClient
from datetime import datetime

client = QdrantClient("localhost", port=6333)

# Delete expired OSINT data
client.delete(
    collection_name="osint",
    points_selector=Filter(
        must=[
            FieldCondition(key="expires_at", range=Range(lt=datetime.now().timestamp()))
        ]
    )
)
```

---

## 🔒 Sécurité

1. **Accès**: Uniquement depuis `vmbr1` (réseau interne)
2. **Auth**: API Key configurée dans `.env`
3. **Encryption**: TLS pour les connexions (via reverse proxy)
4. **Backup**: Snapshot quotidien vers cold storage

---

## 📋 Scripts d'Initialisation

### Créer toutes les collections
```bash
curl -X PUT 'http://localhost:6333/collections/conversations' \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {"size": 1024, "distance": "Cosine"},
    "on_disk_payload": true
  }'
  
# Répéter pour chaque collection...
```

### Python Setup
```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient("localhost", port=6333)

collections = [
    ("conversations", 1024),
    ("osint", 1024),
    ("trading_knowledge", 1024),
    ("documents", 1024),
    ("security_intel", 1024),
]

for name, size in collections:
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE)
    )
```
