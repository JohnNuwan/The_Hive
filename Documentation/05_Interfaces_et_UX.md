# 05 - Interfaces & Expérience Utilisateur (UI/UX)

## 1. Vue d'Ensemble
L'interface est le lien symbiotique entre l'Admin et E.V.A. Elle doit être **fluide**, **sécurisée** (Souveraine), et **multi-modale** (Texte, Voix, Vision).

## 2. Phase 0 : The Nexus (Mobile & Desktop)
C'est l'outil de communication principal en phase de survie.

### Architecture ("WhatsApp Souverain")
*   **Backend** : Matrix (Protocol standard E2EE) ou solution custom légère en Go/Rust (WebSocket).
    *   *Recommandation Senior* : Commencer par une WebApp (PWA) React connectée via WebSocket sécurisé (WSS) à l'API Gateway FastAPI. Simple à dev, compatible partout. Matrix est lourd à maintenir pour un seul utilisateur au début.
*   **Fonctionnalités MVP** :
    *   Chat Streamé (Tokens s'affichent en temps réel).
    *   Upload Photo (Pour analyse Vision).
    *   Voice Notes (Whisper STT processing côté serveur).
    *   Notifications Push (via services WebPush ou App Compagnon simple).

### Sécurité (Tailscale)
*   L'interface n'est PAS exposée sur l'internet public.
*   L'accès se fait via **Tailscale** (VPN Mesh). L'utilisateur active le VPN sur son téléphone, puis accède à `http://the-hive:8000`. C'est Zero-Trust et évite de gérer des certificats SSL complexes et des failles de sécu au début.

## 3. The Panopticon (Admin Dashboard)
Le tableau de bord de contrôle de la Ruche.

### Tech Stack
*   **Frontend** : Next.js ou React (Vite). Design "Cyberpunk/Sci-Fi" (Conformément aux directives esthétiques).
*   **Viz** : Grafana (iframe) pour les métriques techniques (T°, CPU) + Recharts/D3.js pour les graphiques financiers custom.
*   **Backend** : FastAPI endpoints.

### Widgets Critiques
1.  **Finance Widget** : PnL du jour, Drawdown actuel (Barre de vie), État du Challenge.
2.  **Security Widget** : Tentatives d'intrusion, Statut des processus Sentinel.
3.  **Thought Stream** : Log en temps réel de ce que E.V.A. "pense" (Chain of Thought).

## 4. Phase 1 & 2 : Interfaces Immersives (Futures)
*   **Halo (AR)** : Lunettes connectées.
    *   *Proto* : Application mobile utilisant la caméra et ARCore/ARKit pour superposer des infos. Pont vers le serveur Python via WebSocket (Flux vidéo compressé).
*   **The District (VR - Unreal Engine)** :
    *   Jumeau numérique. Ne pas prioriser en Phase 0.

## 5. Roadmap UI
1.  **Semaine 1** : CLI (Terminal) seulement.
2.  **Semaine 2** : Web Dashboard basique (Streamlit ou React simple) pour voir les logs et le PnL.
3.  **Mois 1** : PWA Mobile "Nexus" fonctionnelle pour le chat avec E.V.A. en déplacement.
