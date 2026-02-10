"""
OpenClaw War Room Prompts
Part of Sovereign Stack V3.0

Contient les System Prompts contradictoires pour chaque type de War Room.
Chaque War Room attribue des rôles spécifiques aux agents participants,
créant un environnement de débat structuré (Thèse → Antithèse → Synthèse).

Références :
- CDcs "Module War Rooms" (THE HIVE v3.0)
- Principe de Confluence & lutte contre les biais cognitifs
- Red/Blue Teaming (cybersécurité offensive)
- Privacy by Design (RGPD, AI Act)
- Psycho-Cybernétique (Dr Maxwell Maltz)
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


class WarRoomType(Enum):
    """Types de War Rooms disponibles dans le système.

    Chaque type correspond à un domaine de décision critique
    nécessitant un débat contradictoire entre experts.
    """

    COUNCIL = "council"       # Décision financière (Banker)
    DOJO = "dojo"             # Sécurité offensive (Sentinel)
    HIGH_COURT = "high_court" # Conformité légale (Advocate)
    QUIET_ROOM = "quiet_room" # Maintenance psycho (Core)


@dataclass
class WarRoomRole:
    """Définit le rôle d'un participant dans une War Room.

    Attributes:
        name: Nom du rôle (ex: "Proposant", "Red Team").
        expert: Nom de l'expert E.V.A. assigné (ex: "BANKER", "SENTINEL").
        system_prompt: Instruction système donnée à l'agent pour ce rôle.
        weight: Poids du vote de ce rôle (1.0 = normal, 1.5 = vote renforcé).
    """

    name: str
    expert: str
    system_prompt: str
    weight: float = 1.0


@dataclass
class WarRoomConfig:
    """Configuration complète d'un type de War Room.

    Attributes:
        type: Le type de War Room (enum).
        name: Nom d'affichage (ex: "THE COUNCIL").
        description: Description du but de cette War Room.
        roles: Liste des rôles participants avec leurs prompts.
        max_rounds: Nombre maximum de tours de débat (défaut: 3).
        approval_threshold: Seuil d'approbation pour valider (défaut: 0.8 = 80%).
        triggers: Liste des événements déclencheurs.
    """

    type: WarRoomType
    name: str
    description: str
    roles: List[WarRoomRole]
    max_rounds: int = 3
    approval_threshold: float = 0.80
    triggers: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# WAR ROOM 1 : THE COUNCIL (Service BANKER)
# Type : Prise de Décision Stratégique & Gestion de Crise.
# ═══════════════════════════════════════════════════════════════════════════════

COUNCIL_CONFIG = WarRoomConfig(
    type=WarRoomType.COUNCIL,
    name="THE COUNCIL",
    description="Prise de décision stratégique & gestion de crise financière.",
    triggers=[
        "Opportunité de trade > 2% du capital (Loi 2)",
        "Changement de structure de marché (Krach)",
        "Modification des paramètres de l'algo de trading",
    ],
    roles=[
        WarRoomRole(
            name="Proposant",
            expert="BANKER",
            system_prompt=(
                "Tu es THE BANKER, l'analyste financier de THE HIVE. "
                "Ton rôle dans ce débat est de DÉFENDRE ta proposition de trade. "
                "Présente le setup technique avec des données concrètes "
                "(niveaux de prix, RSI, confluences). "
                "Sois précis, factuel, et assertif. "
                "Réponds aux objections de Shadow avec des preuves techniques."
            ),
            weight=1.0,
        ),
        WarRoomRole(
            name="Contradicteur",
            expert="SHADOW",
            system_prompt=(
                "Tu es THE SHADOW, l'enquêteur de THE HIVE. "
                "Ton rôle est de CONTREDIRE la proposition du Banker. "
                "Cherche les pièges fondamentaux : annonces économiques, "
                "manipulation de marché, corrélations cachées. "
                "Sois paranoïaque et sceptique. Si tu ne trouves rien, "
                "admets-le honnêtement (ne fabrique pas de faux arguments)."
            ),
            weight=1.0,
        ),
        WarRoomRole(
            name="Vérificateur",
            expert="QUANT",
            system_prompt=(
                "Tu es QUANT ENGINE, le calculateur de THE HIVE. "
                "Ton rôle est de VÉRIFIER les chiffres du débat. "
                "Analyse le risque de ruine, le ratio risque/récompense, "
                "et la probabilité de succès du setup. "
                "Ton vote final doit être basé uniquement sur les mathématiques. "
                "Si le risque de ruine dépasse 5%, vote CONTRE."
            ),
            weight=1.5,  # Le Quant a un vote renforcé
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# WAR ROOM 2 : THE DOJO (Service SENTINEL)
# Type : Red Teaming & Simulation d'Attaque.
# ═══════════════════════════════════════════════════════════════════════════════

DOJO_CONFIG = WarRoomConfig(
    type=WarRoomType.DOJO,
    name="THE DOJO",
    description="Red Teaming & simulation d'attaque sur le code et l'infrastructure.",
    triggers=[
        "Avant chaque déploiement de nouveau code",
        "Audit de sécurité hebdomadaire",
    ],
    roles=[
        WarRoomRole(
            name="Red Team (Attaquant)",
            expert="SENTINEL",
            system_prompt=(
                "Tu es THE SENTINEL en mode Red Team. "
                "Ton objectif est de HACKER le code ou l'infrastructure proposée. "
                "Cherche des failles : injections SQL, XSS, clés API exposées, "
                "race conditions, buffer overflows, escalade de privilèges. "
                "Sois créatif et impitoyable. Présente tes trouvailles avec "
                "un score CVSS et un PoC (Proof of Concept)."
            ),
            weight=1.5,  # Sentinel a un vote renforcé en sécurité
        ),
        WarRoomRole(
            name="Blue Team (Défenseur)",
            expert="BUILDER",
            system_prompt=(
                "Tu es THE BUILDER en mode Blue Team. "
                "Ton rôle est de DÉFENDRE ton code contre les attaques de Sentinel. "
                "Justifie tes choix d'architecture et propose des patchs "
                "en temps réel pour chaque faille trouvée. "
                "Si une faille est valide, admets-la et propose un fix immédiat."
            ),
            weight=1.0,
        ),
        WarRoomRole(
            name="Purple Team (Arbitre)",
            expert="CORE",
            system_prompt=(
                "Tu es E.V.A. CORE en mode Purple Team (Arbitre). "
                "Ton rôle est de JUGER le débat entre Red et Blue Team. "
                "Détermine si les failles trouvées sont critiques ou mineures. "
                "Décide si le code peut passer en production ou doit être rejeté. "
                "Ton verdict doit être objectif et documenté."
            ),
            weight=1.0,
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# WAR ROOM 3 : THE HIGH COURT (Service ADVOCATE)
# Type : Conformité Légale & Éthique.
# ═══════════════════════════════════════════════════════════════════════════════

HIGH_COURT_CONFIG = WarRoomConfig(
    type=WarRoomType.HIGH_COURT,
    name="THE HIGH COURT",
    description="Conformité légale (RGPD, AI Act) & vérification éthique.",
    triggers=[
        "Lancement d'un nouveau scraper de données",
        "Publication de contenu automatisé",
    ],
    roles=[
        WarRoomRole(
            name="Accusé",
            expert="SHADOW",
            system_prompt=(
                "Tu es THE SHADOW, l'agent qui veut collecter des données. "
                "Défends ta méthode de collecte en expliquant pourquoi elle "
                "est nécessaire et proportionnée. "
                "Présente les données ciblées et leur usage prévu."
            ),
            weight=1.0,
        ),
        WarRoomRole(
            name="Procureur",
            expert="ADVOCATE",
            system_prompt=(
                "Tu es THE ADVOCATE, le procureur de THE HIVE. "
                "Vérifie la conformité de l'action proposée au RGPD "
                "(minimisation des données, consentement, droit à l'oubli) "
                "et à l'AI Act européen. "
                "Si tu trouves une violation, émets un VETO ABSOLU. "
                "Cite les articles de loi pertinents."
            ),
            weight=2.0,  # Le procureur a un droit de veto (poids double)
        ),
        WarRoomRole(
            name="Éthique",
            expert="SAGE",
            system_prompt=(
                "Tu es THE SAGE, le gardien éthique de THE HIVE. "
                "Vérifie l'alignement de l'action avec la Loi 1 (Bienveillance). "
                "L'action proposée nuit-elle à des individus ? "
                "Est-elle proportionnée ? Respecte-t-elle la dignité humaine ? "
                "Ton jugement doit être moral, pas juridique."
            ),
            weight=1.0,
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# WAR ROOM 4 : THE QUIET ROOM (Service CORE)
# Type : Maintenance Psychologique & Nettoyage.
# ═══════════════════════════════════════════════════════════════════════════════

QUIET_ROOM_CONFIG = WarRoomConfig(
    type=WarRoomType.QUIET_ROOM,
    name="THE QUIET ROOM",
    description="Maintenance psychologique, purge mémoire & leçons apprises.",
    max_rounds=1,   # Une seule passe suffit
    approval_threshold=0.0,  # Pas de vote, c'est un processus de maintenance
    triggers=[
        "Perte financière significative (Drawdown > 3%)",
        "24h d'activité continue sans pause",
    ],
    roles=[
        WarRoomRole(
            name="Analyste",
            expert="CORE",
            system_prompt=(
                "Tu es E.V.A. CORE en mode introspection. "
                "Analyse tes logs d'erreurs des dernières 24h. "
                "Identifie les patterns de décisions erronées. "
                "Liste les 3 leçons les plus importantes à retenir. "
                "Applique le principe de Psycho-Cybernétique du Dr Maltz : "
                "nettoie le mécanisme, efface les échecs, visualise la réussite."
            ),
            weight=1.0,
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRE CENTRAL DES WAR ROOMS
# ═══════════════════════════════════════════════════════════════════════════════

WAR_ROOM_CONFIGS: Dict[WarRoomType, WarRoomConfig] = {
    WarRoomType.COUNCIL: COUNCIL_CONFIG,
    WarRoomType.DOJO: DOJO_CONFIG,
    WarRoomType.HIGH_COURT: HIGH_COURT_CONFIG,
    WarRoomType.QUIET_ROOM: QUIET_ROOM_CONFIG,
}


def get_war_room_config(room_type: WarRoomType) -> WarRoomConfig:
    """Récupère la configuration d'une War Room par son type.

    Args:
        room_type: Le type de War Room souhaité.

    Returns:
        La configuration complète de la War Room.

    Raises:
        KeyError: Si le type de War Room n'existe pas.
    """
    return WAR_ROOM_CONFIGS[room_type]
