# Expert D: THE WRAITH (Le Coach Social & Vision)

## 1. Identité & Rôle
*   **Nom** : The Wraith
*   **Rôle** : Yeux d'EVA, Analyse Comportementale, Assistance AR.
*   **Type** : Actif (Dépend du Hardware TPU/Halo).

## 2. Architecture Technique
*   **Hardware** : Google Coral USB Accelerator / M.2 TPU (Obligatoire).
*   **Software** : **Frigate** (NVR) + Scripts Custom Python `tflite_runtime`.
*   **Modèles** :
    *   `MobileNet SSD v2` (Détection Objets - Rapide).
    *   `FaceNet` (Reconnaissance Faciale).

## 3. Cas d'Usage
*   **The Halo Feed** : Analyse du flux vidéo des lunettes.
    *   *Features* : Reconnaître les gens, afficher leur nom et notes importantes (CRM Social) en AR.
*   **Security Cam** : Analyse des caméras du domicile (Détection Humain vs Chat) sans envoyer la vidéo au Cloud.

## 4. Roadmap Dév
*   **Phase 0** : Inactif (Pas de TPU).
*   **Phase 1** : Achat TPU. Setup Frigate.
*   **Phase 2** : Dev du "Face Matcher" (DB Vectorielle des visages connus).
