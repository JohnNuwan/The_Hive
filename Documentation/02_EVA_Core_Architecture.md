# 02 - Architecture Logicielle E.V.A. (The Core & Experts)

## 1. Vue d'Ensemble Senior Architect
E.V.A. repose sur une architecture **Mixture of Experts (MoE)** distribuée. Contrairement aux MoE monolithiques (comme Mixtral), E.V.A. est un MoE "Agentique" : chaque expert est un service indépendant (Container/Processus) orchestré par un graphe d'états (LangGraph). Cela permet une **modularité totale** et une **résilience** (si l'Expert "Muse" crash, "Banker" continue de trader).

## 2. Stack Technologique & Choix Structurants

*   **Orchestration** : **LangGraph** (Python). C'est le standard industriel actuel pour les workflows cycliques multi-agents.
*   **Communication** :
    *   **Interne (Rapide)** : **Redis Pub/Sub**. Pour l'échange de messages temps réel entre agents.
    *   **Mémoire (Contexte)** : **Qdrant** (Vectoriel) + **Redis** (Short-term state).
    *   **Externe (API)** : **FastAPI**.
*   **Modèles (Inférence)** :
    *   **Engine** : **vLLM** ou **SGLang** (plus performant que Ollama pour la prod) sur la VM 100.
    *   **Quantization** : Utilisation exclusive de modèles **AWQ/GPTQ** (4-bit) pour faire tenir Llama-3-70B (ou équivalent DeepSeek) dans 24GB VRAM, ou fallback sur Llama-3-8B optimisé si 70B trop lent.

## 3. Roadmap de Développement des Experts

### Phase 0: Le Triumvirat Vital (Core, Banker, Builder)
1.  **Expert A: E.V.A. Core (L'Orchestrateur)**
    *   *Dev*: Implémenter le "Router" LangGraph qui classifie l'intention utilisateur (Question ? Ordre Trading ? Code ?).
    *   *Tech*: Modèle léger (Llama-3-8B-Instruct) pour latence minimale (<500ms).
2.  **Expert B: The Banker (La Priorité)**
    *   *Dev*: Créer le pont Python <-> MT5 (ZeroMQ ou API Windows).
    *   *Sécurité*: Implémenter le "Risk Check" en dur DANS le code de l'agent avant tout appel API.
3.  **Expert E: The Builder (L'Auto-Maintenance)**
    *   *Dev*: Scripts de surveillance système. Capacité à lire les logs d'erreur et proposer des fixs.

### Phase 1: Les Sens (Sentinel, Wraith) (Post-TPU)
*   Intégration de la vision et de la surveillance réseau une fois le hardware Coral installé.

## 4. Design Patterns & Standards (Senior Guidelines)

### 🛡️ Pattern: "The Airgap Logic" (Sécurité Financière)
*   *Principe* : L'Expert "Banker" ne doit JAMAIS avoir accès direct à Internet pour naviguer ou télécharger.
*   *Flux* : Banker reçoit des infos de *Shadow* (qui a accès au web), analyse, et envoie un ordre à MT5.
*   *Isolation* : Banker tourne dans un container sans route par défaut vers le WAN, seulement vers le LAN interne et l'API Broker IP whitelisted.

### 🛡️ Pattern: "Thinking Fast and Slow"
*   *Fast (System 1)* : Routage par mots-clés ou modèle Zero-Shot classification (DistilBERT). Coût ~0ms.
*   *Slow (System 2)* : Pour des décisions complexes (Stratégie Trading, Architecture Code), activation de la boucle de "Débat" (Section 7.1 CDC).

### 🛡️ Gestion des Erreurs (Self-Healing)
*   Chaque outil (Tool) appelé par un Agent doit retourner un résultat typé (Success/Error).
*   En cas d'erreur, l'Agent a le droit à **2 retries** avec une modification de son prompt (Reflection) pour corriger l'erreur. Au-delà, escalade vers l'Admin (Notification).

## 5. Structure du Code (Monorepo vs Polyrepo)
Recommandation : **Monorepo** pour la Phase Genesis.
```
/src
  /core (LangGraph logic, Shared Utilities)
  /agents
    /banker (Finance logic, MT5 connectors)
    /builder (System scripts)
    /shadow (OSINT tools)
  /kernel (Rust security modules, FFI bindings)
  /interfaces (FastAPI, Streamlit Dashboard)
  /shared (Data Models Pydantic, Prompts)
```

## 6. Next Steps
1.  Initialiser l'environnement Python (Poetry/Uv).
2.  Mettre en place le serveur d'inférence (Docker vLLM) et valider le chargement d'un modèle sur le GPU unique.
3.  Créer le "Hello World" de l'orchestrateur : Utilisateur -> Core -> Builder -> "System Status OK".
