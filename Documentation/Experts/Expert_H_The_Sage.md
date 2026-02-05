# Expert H: THE SAGE (Le Savant)

## 1. Identité & Rôle
*   **Nom** : The Sage
*   **Rôle** : Santé, Science, Environnement, Synthèse.
*   **Type** : À la demande.

## 2. Architecture Technique
*   **Modèle** : `BioMistral` (Médical) ou `Galactica` (Science).
*   **Connecteurs** :
    *   API Météo/Pollution (OpenMeteo).
    *   Apple Health / Google Fit Export (XML/JSON).

## 3. Missions
*   **Health Dashboard** : Analyser les données de sommeil/sport de l'Admin. Corréler avec la performance Trading ("Tu trades mal quand tu dors < 6h").
*   **Environmental Awareness** : Alerter l'Admin si la qualité de l'air est mauvaise avant son sport.

## 4. Roadmap Dév
*   **Phase 0** : Simple script météo.
*   **Phase 1** : Ingest data santé (Export manuel CSV).
