# Configuration des Ports Réseau - THE HIVE

> **Version**: 1.0.0  
> **Dernière mise à jour**: 2026-02-05

---

##  Vue d'ensemble

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PROXMOX HOST (10.0.0.1)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  vmbr0 (WAN/DMZ)          │  vmbr1 (Internal: 10.0.1.0/24)                  │
│  ├─ :443  → Nginx         │  ├─ EVA Core VM (10.0.1.100)                    │
│  └─ :22   → SSH (Tailsc.) │  │   ├─ :8000  - Core API                       │
│                           │  │   ├─ :8080  - vLLM Server                    │
│                           │  │   ├─ :6333  - Qdrant                         │
│                           │  │   └─ :6379  - Redis                          │
│                           │  │                                              │
│                           │  ├─ Trading VM (10.0.1.200)                     │
│                           │  │   ├─ :8100  - Banker API                     │
│                           │  │   └─ :5432  - TimescaleDB                    │
│                           │  │                                              │
│                           │  └─ Sentinel VM (10.0.1.150)                    │
│                           │      ├─ :8200  - Sentinel API                   │
│                           │      ├─ :1514  - Wazuh (UDP)                    │
│                           │      └─ :55000 - Wazuh API                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

##  VM EVA-Core (10.0.1.100)

| Port | Service | Protocole | Description |
|------|---------|-----------|-------------|
| 8000 | Core API | TCP/HTTP | API FastAPI principale |
| 8080 | vLLM | TCP/HTTP | Serveur d'inférence LLM |
| 11434 | Ollama | TCP/HTTP | Alternative à vLLM |
| 6333 | Qdrant REST | TCP/HTTP | API REST Vector DB |
| 6334 | Qdrant gRPC | TCP/gRPC | API gRPC Vector DB |
| 6379 | Redis | TCP | Message Broker & Cache |

### Règles Firewall (iptables)
```bash
# Autoriser uniquement le réseau interne
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -s 10.0.1.0/24 -j ACCEPT
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -p icmp -j ACCEPT
iptables -A INPUT -j DROP
```

---

##  VM Trading Floor (10.0.1.200)

| Port | Service | Protocole | Description |
|------|---------|-----------|-------------|
| 8100 | Banker API | TCP/HTTP | API FastAPI Trading |
| 5432 | TimescaleDB | TCP | PostgreSQL avec extension time-series |
| 3389 | RDP | TCP | Remote Desktop (Windows only) |

### Notes
- MT5 communique via son propre protocole interne (pas de port exposé)
- RDP uniquement accessible via Tailscale

---

##  VM Sentinel (10.0.1.150)

| Port | Service | Protocole | Description |
|------|---------|-----------|-------------|
| 8200 | Sentinel API | TCP/HTTP | API FastAPI Security |
| 1514 | Wazuh Agent | UDP | Réception logs agents |
| 1515 | Wazuh Agent | TCP | Réception logs agents (TCP) |
| 55000 | Wazuh API | TCP/HTTPS | API de management |

---

##  Services Exposés (WAN via Nginx)

| Port Externe | Port Interne | Service | Notes |
|--------------|--------------|---------|-------|
| 443 | 8000 | Core API | Reverse proxy HTTPS |

### Configuration Nginx
```nginx
server {
    listen 443 ssl http2;
    server_name the-hive.example.com;
    
    ssl_certificate /etc/letsencrypt/live/the-hive.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/the-hive.example.com/privkey.pem;
    
    location /api/ {
        proxy_pass http://10.0.1.100:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

##  Tailscale VPN

Tous les accès administratifs passent par Tailscale (100.x.x.x):

| Port | Service | Usage |
|------|---------|-------|
| 22 | SSH | Administration système |
| 3389 | RDP | Windows VM (Trading) |
| 3000 | Grafana | Dashboards |
| 9090 | Prometheus | Métriques |

---

##  Communication Inter-Services

### Redis Pub/Sub Topics

| Topic | Publishers | Subscribers |
|-------|------------|-------------|
| `eva.core.requests` | All agents | Core |
| `eva.banker.requests` | Core | Banker |
| `eva.banker.responses` | Banker | Core |
| `eva.shadow.requests` | Core | Shadow |
| `eva.sentinel.alerts` | Sentinel | All |
| `eva.system.metrics` | Keeper | All |
| `eva.audit` | All | Black Box |

---

##  Ports à NE JAMAIS Exposer

| Port | Service | Raison |
|------|---------|--------|
| 5432 | PostgreSQL | Données sensibles |
| 6379 | Redis | Pas d'auth forte |
| 8080 | vLLM | Coûteux en ressources |
| 55000 | Wazuh API | Administration critique |
