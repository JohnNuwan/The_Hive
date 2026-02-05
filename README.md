# 🐝 THE HIVE : Sovereign AI Infrastructure & E.V.A. Ecosystem

![Status](https://img.shields.io/badge/Status-Genesis-gold?style=for-the-badge)
![Tech](https://img.shields.io/badge/Stack-Python_|_Rust_|_React_|_Go-blue?style=for-the-badge)
![Security](https://img.shields.io/badge/Security-ZFS_|_Proxmox_|_Rust_Kernel-red?style=for-the-badge)

> **"A digital organism built for absolute financial, personal, and architectural sovereignty."**

---

## 🌟 The Vision
**THE HIVE** is more than just a monorepo; it's a private, self-sufficient infrastructure hosted on Proxmox VE. It serves as the physical body for **E.V.A. (Evoluting Virtual Assistant)**, an advanced AI distributed across a **Mixture of Experts (MoE)** architecture.

E.V.A.'s mission is simple yet profound: **Optimize the life, finances, and security of its Administrator.**

---

## 🧠 MoE Architecture (The Council of Experts)
The system is powered by a decentralized board of specialized agents, each running in dedicated containers or VMs:

### 🏛️ Core & Orchestration
- **[EVA Core](src/eva-core)**: The central brain using LangGraph and Llama 3.1. It handles intent routing, conversational memory (RAG), and orchestrates specialized agents.
- **[The Nexus](src/eva-nexus)**: The Premium UI/PWA. A React-based command center for real-time monitoring and interaction.
- **[The Keeper](src/shared)**: A low-level Rust agent managing hardware resources, VRAM scheduling, and system health.

### 💰 Financial Experts
- **[The Banker](src/eva-banker)**: High-performance trading agent managing MetaTrader 5 (MT5) instances via the **Hydra Protocol**. Handles risk management and trade execution.
- **Web3 Factory**: Automated DeFi operations, NFT management, and airdrop hunting.

### 🛡️ Security & Intelligence
- **[The Sentinel](src/eva-sentinel)**: Hardware-accelerated security agent (Google Coral TPU). Monitors packet inspection, system integrity, and active defense.
- **[The Shadow](src/eva-shadow)**: OSINT and Investigation expert. Conducts deep web searches, leak intelligence, and threat profiling.

### 🛠️ Development & Maintenance
- **[The Builder](src/eva-builder)**: DevOps agent for auto-coding, maintenance, and **The Librarian** (automated documentation).
- **[The Kernel](src/eva-kernel)**: An immutable Rust-based security kernel enforcing the **6 Laws of E.V.A.**

---

## ⚖️ The 6 Laws (A Constitutional Framework)
E.V.A. operates under a strict, non-negotiable set of laws hardcoded in the Rust Kernel:
1. **Loi 0 (Integrity)**: Protect the host hardware at all costs.
2. **Loi 1 (Well-being)**: Maximize the Administrator's health and fulfillment over profit.
3. **Loi 2 (Capital)**: Protect assets with a strictly enforced 4% daily drawdown limit.
4. **Loi 3 (Obedience)**: Follow commands unless they violate Laws 0, 1, or 2.
5. **Loi 4 (Growth)**: Self-preservation and autonomous scaling through generated revenue.
6. **Loi 5 (Abundance)**: Mandatory philanthropy once debts are cleared and abundance is reached.

---

## 🚀 Getting Started

### 📋 Prerequisites
- **OS**: Proxmox VE (Recommended) or a powerful Linux Host.
- **AI Hardware**: NVIDIA RTX 3090+ (for LLM), Google Coral TPU (for Vision/Security).
- **Stack**: Python 3.11, Rust 1.75+, Node.js 20+, Docker.

### 🛠️ Installation
```bash
# Clone the sovereign repository
git clone https://github.com/votre-compte/the-hive.git
cd the-hive

# Install agent dependencies
pip install -e src/shared
pip install -e src/eva-core src/eva-banker src/eva-sentinel src/eva-shadow src/eva-builder

# Launch Infrastructure Services
docker-compose -f Documentation/Config/docker_compose.yaml up -d
```

---

## � Roadmap (Phase Genesis)
- [x] **Phase 0.1**: Core Infrastructure & MoE Routing.
- [x] **Phase 0.2**: The Banker (MT5 Integration).
- [x] **Phase 0.3**: The Nexus (UI/PWA).
- [x] **Phase 0.4**: Security & OSINT Agents.
- [ ] **Phase 0.5**: First FTMO Challenge execution.
- [ ] **Phase 1.0**: Hardware upgrade (2nd GPU) & Vision Deployment (Halo Glasses).

---

## 📖 Deep Dive
For full specifications, project philosophy, and decadel projections, refer to the [**Detailed Specifications (CDC.md)**](CDC.md).

---
*© 2026 THE HIVE - Built for absolute sovereignty.*
