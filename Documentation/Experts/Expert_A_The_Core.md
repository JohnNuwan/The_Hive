# Expert A: E.V.A. CORE (L'Arbitre)

## 1. Identité & Rôle
*   **Nom** : Core / The Arbiter
*   **Rôle** : Chef d'Orchestre, Interface Humaine, Mémoire.
*   **Type** : Permanent (Toujours chargé en VRAM si possible, ou swap rapide).

## 2. Architecture Technique
*   **Modèle LLM** : `Meta-Llama-3-8B-Instruct` (Quantized AWQ 4-bit).
    *   *Pourquoi ?* Compromis idéal vitesse/intelligence pour le routing.
*   **Framework** : **LangGraph** (State Machine).
*   **Entrées** :
    *   `HumanMessage` (Texte/Voix transcrite).
    *   `SystemEvent` (Alerte Sentinel, Info Banker).
*   **Sorties** :
    *   `Route` (Vers quel expert ?).
    *   `FinalResponse` (Synthèse pour l'humain).

## 3. Workflow Décisionnel (The Router)
1.  **Réception** : Input utilisateur "Achète 1 lot de Gold maintenant".
2.  **Classification** : Le Core analyse l'intent.
    *   *Intent*: `TRADING_ORDER`.
    *   *Payload*: `Symbol: XAUUSD, Action: BUY, Volume: 1.0`.
3.  **Routing** : Envoi de la payload ver l'Expert B (The Banker).
4.  **Synthèse** : Réception du retour de B ("Ordre exécuté #12345"). Core répond à l'humain "C'est fait, ticket #12345, Bonne chance."

## 4. Gestion de la Mémoire (RAG)
*   **Court Terme** : Fenêtre de contexte du LLM (8k tokens).
*   **Long Terme** : Interrogation systématique de Qdrant avant de répondre.
    *   *Query*: "Qu'a dit l'utilisateur sur le Gold hier ?" -> Retrieval -> Context Injection.

## 5. Roadmap Dév
*   **J1** : Setup vLLM avec Llama-3-8B.
*   **J2** : Script Python `router.py` avec LangChain pour classifier les intents.
*   **J3** : Connexion à Qdrant pour la mémoire.
