# Tech Architect: GO (The Network)

##  Rôle
Go (Golang) est utilisé pour la **Connectivité Temps Réel**, les **Microservices légers**, et l'**Infrastructure Réseau**. C'est le "Système Circulatoire".

##  Composants Go

### 1. The Nexus Backend (`eva-nexus`)
*   **Type** : Serveur WebSocket / HTTP.
*   **Responsabilité** : Gérer les connexions persistantes avec:
    *   L'App Mobile Admin.
    *   Les Lunettes Halo.
    *   Le Dashboard Web.
*   **Performance** : Goroutines (Green Threads) pour gérer 10k connexions simultanées avec peu de RAM.

### 2. The Message Bus (Optionnel)
*   Si Redis devient un goulot d'étranglement, implémentation d'un bus NATS ou gRPC en Go pour router les messages entre les Experts Python.

### 3. Reverse Proxy & Auth
*   Un petit service Go devant l'API Python pour gérer:
    *   Le Rate Limiting.
    *   La vérification des Tokens JWT/Macaroons.
    *   La compression Gzip.

##  Règles de Dév Go
*   **Simplicité** : Code idiomatique ("Effective Go"). Pas d'abstraction inutile.
*   **Channels** : Utilisation des channels pour la synchronisation, pas de Mutex si possible ("Share memory by communicating").

##  Interaction
*   Go sert de "Buffer" entre le monde extérieur (Internet sale) et le monde intérieur (Python/Rust). Il absorbe les attaques DDoS légères.
