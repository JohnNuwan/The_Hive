# Critères d'Acceptation BDD (Given/When/Then)

> **Format**: Gherkin (Étant donné/Quand/Alors)  
> **Version**: 1.0.0

Ce document enrichit les User Stories avec des critères d'acceptation testables.

---

## 📋 Story 00 : Infrastructure

### TASK-00-01 : Installation Proxmox

```gherkin
Fonctionnalité: Installation Proxmox
  En tant qu'Admin Système
  Je veux installer Proxmox VE sur le serveur
  Afin d'avoir un hyperviseur pour THE HIVE

  Scénario: Accès à l'interface Web Proxmox
    Étant donné que Proxmox VE 8.x est installé
    Quand je navigue vers https://IP_SERVEUR:8006
    Alors je dois voir la page de connexion Proxmox
    Et je peux m'authentifier avec les identifiants root

  Scénario: Pool de stockage ZFS
    Étant donné que Proxmox est installé
    Quand je lance "zpool status" sur l'hôte
    Alors je dois voir un pool ZFS nommé "local-lvm"
    Et le pool doit afficher le statut "ONLINE"

  Scénario: Point de montage The Tablet
    Étant donné que la clé USB The Tablet est connectée
    Quand je lance "mount | grep THE_LAW"
    Alors je dois voir "/mnt/THE_LAW type ext4 (ro,noexec)"
    Et le fichier Constitution.toml doit être lisible
```

### TASK-00-02 : Topologie Réseau

```gherkin
Fonctionnalité: Segmentation Réseau
  En tant qu'Ingénieur Réseau
  Je veux des bridges réseau isolés
  Afin que les VMs soient correctement segmentées

  Scénario: Accès WAN via vmbr0
    Étant donné une VM connectée à vmbr0
    Quand je lance "ping 8.8.8.8" depuis la VM
    Alors le ping doit réussir avec <100ms de latence

  Scénario: Isolation du réseau interne
    Étant donné une VM dans vmbr1 (10.0.1.x)
    Quand un hôte externe tente de se connecter directement
    Alors la connexion doit être refusée
    Et seul le VPN Tailscale doit permettre l'accès

  Scénario: Isolation DMZ Arena
    Étant donné une VM dans vmbr2 (The Arena)
    Quand elle tente d'atteindre vmbr1 (Interne)
    Alors la connexion doit être bloquée par les règles firewall
```

### TASK-00-03 : Templates VM

```gherkin
Fonctionnalité: Templates VM
  En tant qu'Ingénieur DevOps
  Je veux un déploiement VM rapide
  Afin que le scaling soit rapide et cohérent

  Scénario: Cloner Template Ubuntu
    Étant donné que Template-Ubuntu-AI existe
    Quand je le clone avec "qm clone 9000 101 --name test-vm"
    Alors le clone doit se terminer en moins de 120 secondes
    Et la nouvelle VM doit démarrer avec succès

  Scénario: Contenu Template Ubuntu
    Étant donné une VM clonée depuis Template-Ubuntu-AI
    Quand je vérifie les paquets installés
    Alors Python 3.10+ doit être installé
    Et les drivers NVIDIA doivent être pré-installés (si GPU)
    Et Docker doit être installé
```

### TASK-00-04 : GPU Passthrough

```gherkin
Fonctionnalité: Passthrough GPU
  En tant qu'Admin Système
  Je veux l'isolation GPU pour les charges IA
  Afin que la RTX 3090 soit pleinement disponible pour les VMs

  Scénario: Binding VFIO
    Étant donné que IOMMU est activé dans GRUB
    Quand je lance "lspci -nnk | grep -A3 NVIDIA"
    Alors je dois voir "Kernel driver in use: vfio-pci"
    Et PAS "nvidia" ou "nouveau"

  Scénario: Détection GPU dans la VM
    Étant donné que la VM 100 (eva-core) a le GPU passthrough configuré
    Quand je lance "nvidia-smi" dans la VM
    Alors je dois voir "NVIDIA RTX 3090" avec 24GB de mémoire
    Et la version CUDA doit s'afficher
```

### TASK-00-05 : Git & CI/CD

```gherkin
Fonctionnalité: Configuration Repository
  En tant que Lead Développeur
  Je veux une structure monorepo propre
  Afin que le code soit organisé et la qualité assurée

  Scénario: Hooks Pre-commit
    Étant donné que le repo est cloné localement
    Quand je modifie un fichier Python avec mauvais formatage
    Et que je lance "git commit"
    Alors le commit doit échouer
    Et Ruff/Black doit signaler les problèmes de formatage

  Scénario: Build Docker
    Étant donné que les Dockerfiles existent pour Core et Banker
    Quand je lance "docker build -f src/eva-core/Dockerfile ."
    Alors le build doit réussir
    Et l'image doit faire moins de 2GB
```

---

## 📋 Story 01 : EVA Core

### TASK-01-01 : Routeur LangGraph

```gherkin
Fonctionnalité: Classification d'Intent
  En tant qu'EVA Core
  Je veux classifier les intentions utilisateur
  Afin de router les requêtes vers le bon expert

  Scénario: Détection Intent Trading
    Étant donné qu'un utilisateur envoie "Achète 0.5 lot de Gold"
    Quand le Core traite le message
    Alors l'intent doit être classifié comme "TRADING_ORDER"
    Et la confiance doit être > 0.85
    Et le message doit être routé vers "banker"

  Scénario: Fallback Chat Général
    Étant donné qu'un utilisateur envoie "Comment ça va aujourd'hui ?"
    Quand le Core traite le message
    Alors l'intent doit être classifié comme "GENERAL_CHAT"
    Et la réponse doit venir directement du Core
```

### TASK-01-02 : Intégration Mémoire

```gherkin
Fonctionnalité: Mémoire RAG
  En tant qu'EVA Core
  Je veux me souvenir des conversations passées
  Afin que le contexte soit maintenu

  Scénario: Stocker Conversation
    Étant donné une conversation terminée sur le trading
    Quand la session se termine
    Alors la conversation doit être vectorisée
    Et stockée dans la collection Qdrant "conversations"

  Scénario: Récupérer Mémoire Pertinente
    Étant donné que je demande "Rappelle-moi notre discussion sur le Gold"
    Quand le Core recherche en mémoire
    Alors il doit retourner les messages passés pertinents
    Et le score de similarité doit être > 0.7
```

---

## 📋 Story 02 : The Banker

### TASK-02-01 : Validation Risque

```gherkin
Fonctionnalité: Validation Risque Trade
  En tant que The Banker
  Je veux valider les trades selon la Loi 2 de la Constitution
  Afin que le capital soit protégé

  Scénario: Risque sous la limite
    Étant donné une équité compte de 10000 USD
    Et un drawdown journalier de 2%
    Quand un trade avec 0.8% de risque est soumis
    Alors le trade doit être approuvé
    Et aucun avertissement ne doit être levé

  Scénario: Risque au-dessus de la limite
    Étant donné une équité compte de 10000 USD
    Quand un trade avec 1.5% de risque est soumis
    Alors le trade doit être rejeté
    Et la raison doit être "RISK_TOO_HIGH"
    Et constitution_reference doit mentionner "Loi 2"

  Scénario: Limite Drawdown Journalier
    Étant donné un drawdown journalier déjà à 3.95%
    Quand n'importe quel nouveau trade est soumis
    Alors le trade doit être rejeté
    Et la raison doit être "DAILY_LOSS_LIMIT"
    Et le Kill-Switch doit être déclenché
```

### TASK-02-02 : Protection Anti-Tilt

```gherkin
Fonctionnalité: Règle Trading Anti-Tilt
  En tant que Système de Gestion des Risques
  Je veux mettre le trading en pause après des pertes consécutives
  Afin de prévenir le trading émotionnel

  Scénario: Deux Pertes Consécutives
    Étant donné que les 2 derniers trades clôturés sont des pertes
    Quand un nouveau trade est soumis
    Alors le trade doit être rejeté
    Et la raison doit être "ANTI_TILT_ACTIVE"
    Et le trading doit reprendre après 24 heures

  Scénario: Un Gain Casse la Série
    Étant donné que le dernier trade était une perte
    Et que le trade précédent était un gain
    Quand un nouveau trade est soumis
    Alors l'anti-tilt ne doit PAS être déclenché
    Et le trade doit passer à la validation risque
```

### TASK-02-03 : Exécution MT5

```gherkin
Fonctionnalité: Exécution de Trade
  En tant que The Banker
  Je veux exécuter des trades sur MT5
  Afin que les ordres soient placés sur le marché

  Scénario: Ordre Market Réussi
    Étant donné que MT5 est connecté
    Et que le marché est ouvert
    Quand je soumets un ordre BUY pour XAUUSD avec 0.5 lots
    Alors l'ordre doit être exécuté
    Et je dois recevoir un numéro de ticket
    Et le temps d'exécution doit être < 100ms

  Scénario: Stop Loss Manquant
    Étant donné un ordre de trade sans stop_loss_price
    Quand l'ordre est validé
    Alors l'ordre doit être rejeté immédiatement
    Et la raison doit citer "ROE Trading: SL Obligatoire"
```

---

## 📋 Story 03 : The Sentinel

### TASK-03-01 : Détection d'Intrusion

```gherkin
Fonctionnalité: Détection Brute Force
  En tant que The Sentinel
  Je veux détecter et bloquer les attaques brute force
  Afin que le système soit protégé

  Scénario: Brute Force SSH
    Étant donné 5 tentatives de connexion SSH échouées d'une même IP en 60 secondes
    Quand le Sentinel analyse les logs
    Alors l'IP source doit être bloquée pendant 3600 secondes
    Et un SECURITY_ALERT doit être publié sur Redis
    Et une notification Discord doit être envoyée

  Scénario: Échec de Connexion Légitime
    Étant donné 2 tentatives de connexion SSH échouées d'une même IP
    Quand le Sentinel analyse les logs
    Alors l'IP ne doit PAS être bloquée
    Mais l'événement doit être loggé avec sévérité "low"
```

---

## 📋 Story 04 : The Nexus (UI)

### TASK-04-01 : Commande Vocale

```gherkin
Fonctionnalité: Entrée Vocale
  En tant qu'Admin
  Je veux parler à EVA
  Afin de pouvoir interagir mains libres

  Scénario: Reconnaissance Vocale
    Étant donné que le microphone est actif
    Quand je dis "EVA, quel est le statut de mes positions ?"
    Alors l'audio doit être transcrit
    Et le texte doit être envoyé à l'API Core
    Et je dois recevoir une réponse audio

  Scénario: Authentification Vocale (Future)
    Étant donné que l'authentification vocale est activée
    Quand une voix inconnue donne un ordre de trade
    Alors l'ordre doit être rejeté
    Et la raison doit être "VOICE_NOT_RECOGNIZED"
```
