# Diagrammes d'Architecture THE HIVE

> **Version**: 1.0.0  
> **Format**: Mermaid

---

## 📐 C4 - Context Diagram

Vue d'ensemble du système et ses interactions externes.

```mermaid
C4Context
    title System Context - THE HIVE

    Person(admin, "Admin / John", "Propriétaire du système")
    
    System_Boundary(hive, "THE HIVE") {
        System(eva, "E.V.A.", "IA personnelle autonome")
    }
    
    System_Ext(mt5, "MetaTrader 5", "Plateforme de trading")
    System_Ext(propfirm, "Prop Firms", "FTMO, FundedNext...")
    System_Ext(osint_sources, "OSINT Sources", "Google, Shodan, etc.")
    System_Ext(comms, "Communications", "Telegram, Discord, Email")
    
    Rel(admin, eva, "Interagit via", "The Nexus / Voix")
    Rel(eva, mt5, "Exécute trades", "MT5 Python API")
    Rel(mt5, propfirm, "Connecté à", "Broker API")
    Rel(eva, osint_sources, "Collecte données", "HTTPS")
    Rel(eva, comms, "Notifie", "Bot API")
```

---

## 📦 C4 - Container Diagram

Architecture des conteneurs/VMs du système.

```mermaid
C4Container
    title Container Diagram - THE HIVE Infrastructure

    Person(admin, "Admin")
    
    System_Boundary(proxmox, "Proxmox VE - Hypervisor") {
        
        Container_Boundary(vm_core, "VM EVA-Core - Ubuntu 22.04") {
            Container(core_api, "Core API", "FastAPI + LangGraph", "Orchestrateur principal")
            Container(llm_server, "vLLM Server", "Python", "Inference LLM")
            Container(qdrant, "Qdrant", "Vector DB", "Mémoire long-terme")
            Container(redis, "Redis", "Message Broker", "Pub/Sub + Cache")
        }
        
        Container_Boundary(vm_trading, "VM Trading Floor - Windows 11") {
            Container(mt5, "MetaTrader 5", "Terminal", "Exécution trades")
            Container(banker_api, "Banker API", "FastAPI", "Gestion trading")
            Container(timescale, "TimescaleDB", "PostgreSQL", "Données marché")
        }
        
        Container_Boundary(vm_security, "VM Sentinel - Ubuntu 22.04") {
            Container(wazuh, "Wazuh", "SIEM/XDR", "Monitoring sécurité")
            Container(sentinel_api, "Sentinel API", "FastAPI", "Alertes")
        }
        
        Container(kernel, "EVA Kernel", "Rust Binary", "Sécurité / Risk Mgmt")
    }
    
    System_Ext(tablet, "The Tablet", "USB Key", "Lois immuables")
    System_Ext(vault, "The Vault", "YubiKey HSM", "Secrets")
    
    Rel(admin, core_api, "API REST", "HTTPS/Tailscale")
    Rel(core_api, redis, "Messages", "TCP 6379")
    Rel(core_api, llm_server, "Inference", "HTTP 8080")
    Rel(core_api, qdrant, "Vectors", "HTTP 6333")
    Rel(redis, banker_api, "Pub/Sub", "TCP 6379")
    Rel(banker_api, mt5, "Trades", "MT5 Python")
    Rel(banker_api, timescale, "Market Data", "TCP 5432")
    Rel(kernel, tablet, "Vérifie", "USB")
    Rel(kernel, vault, "Signe", "PIV")
```

---

## 🔄 Sequence - Trade Execution Flow

Flux complet d'exécution d'un ordre de trading.

```mermaid
sequenceDiagram
    autonumber
    participant User as 👤 Admin
    participant Nexus as 📱 The Nexus
    participant Core as 🧠 EVA Core
    participant Kernel as 🔒 Rust Kernel
    participant Banker as 💰 The Banker
    participant MT5 as 📊 MetaTrader 5
    participant DB as 🗄️ TimescaleDB
    
    User->>Nexus: "Achète 0.5 lot Gold avec SL à 2050"
    Nexus->>Core: POST /chat/message
    
    Core->>Core: Intent Detection (TRADING_ORDER)
    Core->>Core: Extract entities (XAUUSD, BUY, 0.5, 2050)
    
    Core->>Kernel: ValidateTrade(request)
    
    Kernel->>Kernel: Check Law 2 (Risk < 1%)
    Kernel->>Kernel: Check Daily Drawdown (< 4%)
    Kernel->>Kernel: Check Anti-Tilt
    Kernel->>Kernel: Verify Biometrics
    
    alt Risk Check PASSED
        Kernel-->>Core: TradeValidationResult(approved=true)
        
        Core->>Banker: TRADE_EXECUTE via Redis
        
        Banker->>Banker: Calculate Lot Size
        Banker->>MT5: order_send(symbol, action, volume, sl, tp)
        MT5-->>Banker: OrderResult(ticket=12345)
        
        Banker->>DB: INSERT trade_orders
        Banker->>Core: TRADE_RESULT via Redis
        
        Core->>Core: Generate Response
        Core-->>Nexus: ChatResponse("Ordre exécuté: Ticket #12345...")
        Nexus-->>User: Display Response
        
    else Risk Check FAILED
        Kernel-->>Core: TradeValidationResult(rejected, reason)
        Core-->>Nexus: ChatResponse("Ordre refusé: Loi 2 - Daily DD limit")
        Nexus-->>User: Display Rejection
    end
```

---

## 🛡️ Sequence - Security Alert Flow

Flux de détection et réponse à une alerte sécurité.

```mermaid
sequenceDiagram
    autonumber
    participant Attack as 🔴 Attacker
    participant FW as 🧱 Firewall
    participant Wazuh as 🛡️ Wazuh SIEM
    participant Sentinel as 🔍 The Sentinel
    participant Kernel as 🔒 Kernel
    participant Discord as 💬 Discord
    participant Admin as 👤 Admin
    
    Attack->>FW: Brute Force SSH (5 attempts)
    FW->>Wazuh: Log Events
    
    Wazuh->>Wazuh: Correlation Rule Match
    Wazuh->>Sentinel: Alert JSON
    
    Sentinel->>Sentinel: AI Analysis (Cyber-Llama)
    Sentinel->>Sentinel: Severity: HIGH
    
    Sentinel->>Kernel: Block IP Request
    Kernel->>FW: iptables -A INPUT -s 192.168.1.100 -j DROP
    
    Sentinel->>Discord: POST /webhooks (Alert + Screenshot)
    Discord-->>Admin: 🚨 Security Alert Notification
    
    Sentinel->>Sentinel: Log to audit_trail (Black Box)
```

---

## 🌐 Network Topology

Architecture réseau avec VLANs et flux.

```mermaid
flowchart TB
    subgraph WAN ["🌍 Internet"]
        ISP[ISP Router]
        Tailscale[Tailscale VPN]
    end
    
    subgraph DMZ ["DMZ - vmbr0"]
        Nginx[Nginx Reverse Proxy]
    end
    
    subgraph Internal ["Internal - vmbr1 (10.0.1.0/24)"]
        subgraph Core ["EVA Core VM"]
            CoreAPI[Core API :8000]
            LLM[vLLM :8080]
            Qdrant[Qdrant :6333]
            Redis[Redis :6379]
        end
        
        subgraph Trading ["Trading Floor VM"]
            BankerAPI[Banker API :8100]
            MT5[MT5 Terminal]
            TimescaleDB[TimescaleDB :5432]
        end
        
        subgraph Security ["Sentinel VM"]
            WazuhManager[Wazuh :1514/1515]
            SentinelAPI[Sentinel API :8200]
        end
        
        Kernel[Rust Kernel]
    end
    
    subgraph Hardware ["Hardware Security"]
        Tablet[(The Tablet USB)]
        Vault[(YubiKey HSM)]
        Watchdog[(ESP32 Watchdog)]
    end
    
    ISP --> |Public IP| Nginx
    Tailscale --> |VPN Tunnel| CoreAPI
    
    Nginx --> |443 → 8000| CoreAPI
    
    CoreAPI <--> Redis
    CoreAPI --> LLM
    CoreAPI --> Qdrant
    
    Redis <-.-> BankerAPI
    Redis <-.-> SentinelAPI
    
    BankerAPI --> MT5
    BankerAPI --> TimescaleDB
    
    WazuhManager --> SentinelAPI
    
    Kernel --> Tablet
    Kernel --> Vault
    Watchdog --> |GPIO| Kernel
    
    style WAN fill:#fdd,stroke:#f00
    style DMZ fill:#ffd,stroke:#ff0
    style Internal fill:#dfd,stroke:#0f0
    style Hardware fill:#ddf,stroke:#00f
```

---

## 🤖 Agent Communication

Flux de communication entre agents EVA.

```mermaid
flowchart LR
    subgraph Core ["🧠 The Core"]
        Router[Intent Router]
        Memory[Memory Manager]
    end
    
    subgraph Experts ["Experts"]
        Banker[💰 Banker]
        Shadow[🔍 Shadow]
        Sentinel[🛡️ Sentinel]
        Builder[🔧 Builder]
        Muse[🎨 Muse]
    end
    
    subgraph Infra ["Infrastructure"]
        Redis[(Redis Pub/Sub)]
        Kernel[🔒 Kernel]
        Keeper[⚡ Keeper]
    end
    
    Router --> |eva.banker.requests| Redis
    Redis --> |subscribe| Banker
    Banker --> |eva.banker.responses| Redis
    Redis --> |subscribe| Router
    
    Router --> |eva.shadow.requests| Redis
    Router --> |eva.sentinel.requests| Redis
    
    Keeper --> |eva.system.metrics| Redis
    Sentinel --> |eva.sentinel.alerts| Redis
    
    Banker --> |validate| Kernel
    Sentinel --> |audit| Kernel
```

---

## 🗃️ Data Flow

Flux des données à travers le système.

```mermaid
flowchart TD
    subgraph Input ["📥 Input Sources"]
        Voice[🎤 Voice Command]
        Text[⌨️ Text Message]
        Market[📈 Market Data]
        Logs[📋 System Logs]
    end
    
    subgraph Processing ["⚙️ Processing"]
        LLM[🧠 LLM Inference]
        Intent[Intent Classification]
        RAG[RAG Retrieval]
        Risk[Risk Calculation]
    end
    
    subgraph Storage ["💾 Storage"]
        Qdrant[(Qdrant Vectors)]
        Timescale[(TimescaleDB)]
        Redis[(Redis Cache)]
        Audit[(Audit Trail)]
    end
    
    subgraph Output ["📤 Output"]
        Response[💬 Chat Response]
        Trade[📊 Trade Execution]
        Alert[🚨 Security Alert]
        Metrics[📉 Metrics]
    end
    
    Voice --> LLM
    Text --> LLM
    LLM --> Intent
    Intent --> RAG
    RAG --> Qdrant
    
    Market --> Timescale
    Market --> Risk
    Risk --> Trade
    
    Logs --> Redis
    Redis --> Alert
    
    Trade --> Audit
    Alert --> Audit
    
    LLM --> Response
    Risk --> Trade
    Response --> Text
```

---

## 🏗️ Class Diagram - Trading Models

Structure des classes principales du module Trading.

```mermaid
classDiagram
    class TradeOrder {
        +UUID id
        +String symbol
        +TradeAction action
        +Decimal volume
        +Decimal stop_loss_price
        +Decimal take_profit_price
        +OrderType order_type
        +OrderSource source
        +DateTime created_at
    }
    
    class Position {
        +int ticket
        +String symbol
        +TradeAction action
        +Decimal volume
        +Decimal open_price
        +Decimal current_price
        +Decimal profit
        +DateTime open_time
    }
    
    class RiskStatus {
        +UUID account_id
        +Decimal daily_drawdown_percent
        +Decimal total_drawdown_percent
        +bool anti_tilt_active
        +bool trading_allowed
        +check_trading_allowed()
    }
    
    class AccountBalance {
        +int login
        +String server
        +Decimal balance
        +Decimal equity
        +Decimal margin
        +Decimal free_margin
    }
    
    class PropFirmAccount {
        +UUID id
        +String name
        +String broker
        +String phase
        +Decimal initial_balance
        +Decimal current_balance
        +bool copy_enabled
    }
    
    TradeOrder --> Position : executes
    RiskStatus --> TradeOrder : validates
    AccountBalance --> RiskStatus : provides data
    PropFirmAccount --> AccountBalance : contains
    PropFirmAccount --> TradeOrder : copies
```

---

## 📊 State Machine - Trade Lifecycle

États d'un ordre de trading.

```mermaid
stateDiagram-v2
    [*] --> Created: User submits order
    
    Created --> Validating: Send to Kernel
    
    Validating --> RiskCheck: Kernel validates
    
    RiskCheck --> Approved: Risk OK
    RiskCheck --> Rejected: Risk NOK
    
    Rejected --> [*]: Log & Notify
    
    Approved --> Executing: Send to MT5
    
    Executing --> Filled: Order executed
    Executing --> PartialFill: Partial execution
    Executing --> Failed: Execution error
    
    PartialFill --> Filled: Complete fill
    PartialFill --> Cancelled: User cancels
    
    Failed --> [*]: Log error
    
    Filled --> Open: Position active
    
    Open --> Closed_SL: Stop Loss hit
    Open --> Closed_TP: Take Profit hit
    Open --> Closed_Manual: User closes
    Open --> Modified: SL/TP modified
    
    Modified --> Open: Resume
    
    Closed_SL --> [*]: Record P&L
    Closed_TP --> [*]: Record P&L
    Closed_Manual --> [*]: Record P&L
    Cancelled --> [*]: Log
```
