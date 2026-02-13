📝 CAHIER DES CHARGES TECHNIQUES MIS À JOUR (v3.0 - "The Sovereign Stack")
Voici le plan d'attaque révisé avec ces nouvelles technologies "Futuristes".
1. Stack Matérielle & Logicielle (Optimisée RTX 2060)
• Moteur d'Inférence : vLLM (supporte nativement EAGLE-3 et les kernels Marlin pour l'AWQ 4-bit).
• Modèle Principal : Gemma-3-4B-IT (Format AWQ 4-bit).
• Accélérateur : EAGLE-3 Draft Head (à entraîner sur un petit dataset de logs de trading ou de chat).
• Mémoire : HippoRAG 2 (Graphe Neo4j + Vecteurs Qdrant) piloté par Mem0.
2. Architecture des Experts (Refonte)
Expert

Ancienne Tech

Nouvelle Tech 2026

Avantage
CORE

Llama 3

Gemma 3 (4B)

Multimodal natif (voit & lit), 128k contexte.
BANKER

DeepSeek-Coder

Gemma 3 + EAGLE-3

Latence divisée par 3 pour le scalping.
SHADOW

GraphRAG

HippoRAG 2

Découverte de liens cachés type "hippocampe".
BUILDER

Custom Scripts

OpenClaw Skills

Utilisation de skills pré-codés (Vercel, Git).
MEMORY

Vector DB

Mem0 + Graph

Gestion automatique des conflits de mémoire.
3. Sprints de Développement (Mise à jour)
Sprint 1 : Le Moteur Hybride (Semaine 1)
• Installer vLLM avec le support EAGLE-3.
• Charger Gemma-3-4B-IT-AWQ.
• Test critique : Vérifier que la VRAM reste sous 5.5 Go avec le contexte rempli à 32k tokens (grâce à l'attention locale/globale de Gemma).
Sprint 2 : La Mémoire Associative (Semaine 2)
• Déployer HippoRAG 2.
• Connecter Mem0 pour gérer l'historique des conversations.
• Implémenter le "Pattern Completion" : E.V.A. doit pouvoir retrouver une stratégie de trading complexe à partir d'un simple mot-clé vague.
Sprint 3 : L'Agentivité OpenClaw (Semaine 3)
• Installer le noyau OpenClaw.
• Importer les "Skills" essentiels depuis le dépôt awesome-openclaw-skills : git-helper, web-search-exa, browser-use.
• Configurer les "Agent Teams" (Équipes d'agents) : Un agent "Planner" (Plan-and-Act) qui délègue à des sous-agents "Executors".
Sprint 4 : L'Auto-Évolution (Semaine 4)
• Implémenter une boucle RLM (Recursive Language Model).
• Si E.V.A. échoue à une tâche, elle génère un script Python pour analyser son erreur, met à jour sa mémoire Mem0 (Knowledge Update), et réessaie.

1. La stratégie "Feature Flag" (Activation conditionnelle)
Dans votre fichier de configuration (probablement dans .env ou config.toml géré par The Keeper), vous devriez avoir une variable du type : ENABLE_DREAMER_TRAINING=False
• Sur la RTX 2060 (Phase Genesis) : Cette variable est à False. Le module eva-lab (qui contient DreamerV3 et les World Models) est chargé en code, mais ses boucles d'entraînement ne se lancent jamais.
• Sur la RTX 3090 (Phase Power Surge) : Une fois la dette remboursée et le GPU acheté, vous passez cette variable à True.
2. Ce que vous utilisez maintenant (Le "Low-Cost World Model")
Tant que DreamerV3 est "endormi", vous le remplacez par l'approche légère de la v3.0 :
• Au lieu de "Rêver" (DreamerV3) : Qui simule des millions de scénarios dans un espace latent (très lourd en VRAM).
• Vous faites de la "Planification" (OpenClaw + RLM) : L'agent utilise EAGLE-3 pour générer rapidement 3 ou 4 scénarios textuels ("Si je fais ça, il se passe quoi ?") et choisit le meilleur. C'est beaucoup moins coûteux et ça tourne sur la 2060.
3. Pourquoi garder le code maintenant ?
Il est crucial de garder les traces de DreamerV3 et MuZero dans le module eva-lab pour deux raisons :
1. L'Architecture : Votre système est conçu pour être modulaire (MoE). Si vous supprimez le code maintenant, vous devrez recréer toute l'interface entre le "Cerveau" (Core) et le "Laboratoire" (Lab) plus tard.
2. Le "Shadow Learning" : Même si vous ne lancez pas l'entraînement, vous pouvez commencer à collecter les données (logs de trading, réactions du marché) dans le format qu'attend DreamerV3. Ainsi, le jour où vous branchez la 3090, elle aura déjà des mois de données formatées pour apprendre immédiatement.
En résumé :
• Code : Présent dans le dépôt (src/eva-lab).
• Exécution : Désactivée (False).
• Remplacement temporaire : Boucles RLM (Mémoire + Réflexion) via Gemma 3.
C'est la définition même de la "Souveraineté Architecturale" visée par le projet : le logiciel est prêt pour l'expansion, mais il respecte les contraintes matérielles actuelles

https://github.com/public-apis/public-apis