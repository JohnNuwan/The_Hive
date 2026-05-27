# 🏭 HERMES CHALLENGE FACTORY & MASTER LLM OPENROUTER

Ce document détaille les modules de certification prop firm d'Hermès, le Daemon d'analyse technique proactive (Chartist) et l'intégration du client LLM OpenRouter pour le Cortex.

---

## 1. Hermes Challenge Factory (`hermes_challenge_factory.py`)

La **Challenge Factory** est le module autonome de certification des champions. Elle simule un challenge prop firm binaire sur l'historique réel en ticks ou bougies de MetaTrader 5 avant de promouvoir un modèle JAX (MuZero / Dreamer) comme Champion.

### 🛡️ Le Profil Strict `custom_strict`
Afin de répondre à des exigences de sécurité renforcées, un profil de règles ultra-conservateur et performant a été injecté :
*   **Drawdown Quotidien Maximal (`daily_dd_pct`)** : **2.0 %** (au lieu de 5.0 %).
*   **Drawdown Global Maximal (`total_dd_pct`)** : **5.0 %** (au lieu de 10.0 %).
*   **Objectif de Profit Mensuel (`profit_target_pct`)** : **8.0 %** (soit environ 0.36 % par jour ouvré), un ratio réaliste et mathématiquement sécurisé.

### ⏱️ Évaluation Multi-Timeframe
Le script supporte désormais l'évaluation sur n'importe quel timeframe via la ligne de commande (`--timeframe`) :
```bash
# Évaluation d'un challenge sur les bougies M15 réelles
python scripts/hermes_challenge_factory.py --firm custom_strict --timeframe M15 --dry-run
```

---

## 2. Hermes Chartist Daemon (`hermes.chartist.bat`)

Le Daemon Chartist extrait l'historique des prix MT5, calcule les indicateurs techniques (Fibonacci Golden Pocket, points pivots, ADX, RSI, ATR) et interroge l'expert Hermès pour rédiger des bulletins sémantiques complets avec graphiques.

### 🔄 Cycle Multi-Timeframe Séquentiel
Le daemon de lancement a été modifié pour envoyer successivement, toutes les 4 heures, des briefs techniques d'une richesse absolue sur le salon Discord `#analyse-technique` :
1.  **Timeframe H4** : Analyse de structure majeure et tendance de fond (Swing).
2.  **Timeframe H1** : Analyse de dynamique de tendance et points de confluences (Intraday).
3.  **Timeframe M15** : Analyse court-terme et micro-configurations (Scalping).

---

## 3. Support OpenRouter dans `LLMClient` (`llm_client.py`)

Le client LLM partagé du Banker dispose désormais d'un support de premier ordre pour le backend **OpenRouter** en plus de vLLM et Ollama.

### 🧠 Biais Cortex Actif sans dépendance locale
*   **Raison d'être** : Éviter le blocage systématique `Cortex: NEUTRAL` et les messages d'avertissement `vLLM indisponible` lorsque le conteneur GPU vLLM local est éteint (notamment pour libérer la VRAM pendant les entraînements nocturnes).
*   **Fonctionnement** : Si `LLM_BACKEND="openrouter"` est configuré dans le `.env` local, `LLMClient` interroge directement l'API externe d'OpenRouter en injectant les headers OpenAI-compatibles et la clé `OPENROUTER_API_KEY`.
*   **Bénéfice** : Le Cortex reste pleinement actif et émet des biais directionnels ou consultatifs fins en direct tout en préservant 100 % de tes ressources matérielles GPU locales.

---

## 🚀 Commandes Utiles de Lancement
*   **Lancement du Chartist** :
    `call hermes.chartist.bat`
*   **Exécution manuelle de la Factory** :
    `python scripts/hermes_challenge_factory.py --firm custom_strict --timeframe M15`
