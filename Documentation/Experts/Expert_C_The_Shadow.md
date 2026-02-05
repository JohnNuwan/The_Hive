# Expert C: THE SHADOW (L'Enquêteur OSINT)

## 1. Identité & Rôle
*   **Nom** : The Shadow
*   **Rôle** : Recherche Web, OSINT, Threat Intel.
*   **Type** : À la demande (On-Demand).

## 2. Architecture Technique
*   **Modèle LLM** : `Dolphin-2.x-Mistral` (Non-Censuré).
    *   *Pourquoi ?* Nécessaire pour analyser des données de leaks ou de sécurité sans refus moralisateur ("Je ne peux pas t'aider à hacker").
*   **Outils (Tools)** :
    *   `GoogleSerperAPI` : Recherche Web.
    *   `Shodan API` : Scan infrastructure.
    *   `Hunchly` (via Python Selenium) : Capture de preuves.

## 3. Protocoles Spéciaux (Grey Zone)
*   **ROE (Rules of Engagement)** :
    *   *Autorisé* : Scraper des données publiques, consulter des bases de leaks (HaveIBeenPwned, DeHashed) pour *défense/audit*.
    *   *Interdit* : Interaction active avec la cible (Login, SQL Injection).

## 4. Missions Types
1.  **Due Diligence** : "Qui est ce CEO ?" -> Scraping LinkedIn, Greffe tribunal, News.
2.  **Network Recon** : "Quels ports sont ouverts sur mon serveur ?" -> Shodan Scan.

## 5. Roadmap Dév
*   **J1** : Setup Docker pour Dolphin-Mistral.
*   **J2** : Création du Tool `WebSearch`.
*   **J3** : Pipeline de synthèse (Search -> Read -> Summarize).
