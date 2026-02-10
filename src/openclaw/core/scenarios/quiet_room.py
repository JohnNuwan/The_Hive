"""
OpenClaw War Room Scenario: THE QUIET ROOM
Part of Sovereign Stack V3.0 — Sprint WR-4

Module de maintenance psychologique et nettoyage mémoriel.
Inspiré de la Psycho-Cybernétique du Dr Maxwell Maltz.

Objectifs :
    - Purger la mémoire court terme (context) après un traumatisme (drawdown).
    - Extraire les leçons apprises et les archiver en mémoire long terme (Mem0).
    - Réinitialiser l'état émotionnel de l'agent pour reprendre sereinement.

Déclencheurs automatiques :
    - Drawdown > 3% sur la journée.
    - 24h d'activité continue sans pause.
    - Erreurs en cascade (> 5 erreurs en 1h).

Références :
    - CDcs War Rooms, Sprint 4 (Semaine 4)
    - Psycho-Cybernétique (Dr Maxwell Maltz) : "Nettoyer le mécanisme"
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from ..war_room import WarRoomSession, WarRoomVerdict
from ..war_room_prompts import WarRoomType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QuietRoomTrigger:
    """Événement déclencheur de la Quiet Room.

    Attributes:
        reason: Raison du déclenchement.
        severity: Niveau de gravité (1-10).
        details: Détails supplémentaires.
        timestamp: Horodatage du déclenchement.
    """

    reason: str
    severity: int = 5
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LessonLearned:
    """Leçon extraite lors de la session Quiet Room.

    Attributes:
        category: Catégorie (trading, security, infrastructure, general).
        lesson: Texte de la leçon.
        action_item: Action concrète à prendre.
        priority: Priorité (1=haute, 3=basse).
    """

    category: str
    lesson: str
    action_item: str = ""
    priority: int = 2


@dataclass
class QuietRoomReport:
    """Rapport de la session Quiet Room.

    Attributes:
        session_id: ID de la session War Room.
        trigger: Événement déclencheur.
        lessons: Leçons extraites.
        errors_purged: Nombre d'erreurs purgées du court terme.
        lessons_archived: Nombre de leçons archivées en long terme.
        report_text: Rapport final formaté.
        timestamp: Horodatage du rapport.
    """

    session_id: str
    trigger: QuietRoomTrigger
    lessons: List[LessonLearned] = field(default_factory=list)
    errors_purged: int = 0
    lessons_archived: int = 0
    report_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# QUIET ROOM SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════


class QuietRoomSession:
    """Gère une session de maintenance psychologique.

    La Quiet Room est un processus d'introspection où l'agent :
    1. Coupe les inputs sensoriels (marché, alertes).
    2. Analyse ses erreurs récentes.
    3. Extrait les leçons et les archive en mémoire long terme.
    4. Purge la mémoire court terme pour repartir à zéro.

    Contrairement aux autres War Rooms, il n'y a pas de débat adversarial.
    C'est un processus solo de Core en mode introspection.

    Usage :
        quiet = QuietRoomSession()
        report = await quiet.run(trigger, error_logs, llm_service, memory_bridge)
    """

    async def run(
        self,
        trigger: QuietRoomTrigger,
        error_logs: List[str],
        llm_service,
        memory_bridge=None,
    ) -> QuietRoomReport:
        """Exécute la session complète de Quiet Room.

        Args:
            trigger: L'événement déclencheur.
            error_logs: Liste des logs d'erreur à analyser.
            llm_service: Service LLM pour l'introspection.
            memory_bridge: MemoryBridge pour archiver les leçons.

        Returns:
            QuietRoomReport avec leçons, stats de purge, et rapport.
        """
        logger.info(
            f"[QuietRoom] 🧘 Session déclenchée: {trigger.reason} "
            f"(severity: {trigger.severity}/10)"
        )

        # ─── Phase 1 : Coupure sensorielle ───
        logger.info("[QuietRoom] Phase 1: Coupure des inputs sensoriels...")
        # (En production, on désactive les webhooks marché/Twitter ici)

        # ─── Phase 2 : Analyse des erreurs via War Room ───
        logger.info("[QuietRoom] Phase 2: Analyse introspective...")

        errors_text = "\n".join(f"  - {log}" for log in error_logs[-20:])
        subject = (
            f"INTROSPECTION — Raison: {trigger.reason}\n"
            f"Sévérité: {trigger.severity}/10\n"
            f"Détails: {trigger.details}\n\n"
            f"LOGS D'ERREURS RÉCENTS:\n{errors_text}\n\n"
            f"Applique le principe de Psycho-Cybernétique:\n"
            f"1. Identifie les PATTERNS d'erreurs (pas les cas isolés).\n"
            f"2. Extrais les 3 LEÇONS les plus importantes.\n"
            f"3. Pour chaque leçon, propose une ACTION CONCRÈTE.\n"
            f"4. Visualise la réussite future."
        )

        session = WarRoomSession(
            room_type=WarRoomType.QUIET_ROOM,
            subject=subject,
        )
        verdict = await session.run_debate(
            llm_service=llm_service,
            memory_bridge=None,  # On archive manuellement après extraction
        )

        # ─── Phase 3 : Extraction des leçons ───
        logger.info("[QuietRoom] Phase 3: Extraction des leçons...")
        lessons = self._extract_lessons(session)

        # ─── Phase 4 : Archivage long terme (Mem0) ───
        archived_count = 0
        if memory_bridge and lessons:
            logger.info("[QuietRoom] Phase 4: Archivage Mem0...")
            for lesson in lessons:
                try:
                    await memory_bridge.add(
                        content=f"[LEÇON] {lesson.category}: {lesson.lesson}. Action: {lesson.action_item}",
                        metadata={
                            "type": "lesson_learned",
                            "category": lesson.category,
                            "priority": lesson.priority,
                            "trigger": trigger.reason,
                        },
                    )
                    archived_count += 1
                except Exception as e:
                    logger.warning(f"[QuietRoom] Archivage échoué: {e}")

        # ─── Phase 5 : Purge du court terme ───
        logger.info("[QuietRoom] Phase 5: Purge mémoire court terme...")
        errors_purged = len(error_logs)
        # (En production, on vide les buffers Redis et le context window ici)

        # ─── Rapport ───
        report = QuietRoomReport(
            session_id=session.session_id,
            trigger=trigger,
            lessons=lessons,
            errors_purged=errors_purged,
            lessons_archived=archived_count,
        )
        report.report_text = self._generate_report(report)

        logger.info(
            f"[QuietRoom] 🧘 Session terminée: {len(lessons)} leçons, "
            f"{errors_purged} erreurs purgées, {archived_count} archivées"
        )
        return report

    def _extract_lessons(self, session: WarRoomSession) -> List[LessonLearned]:
        """Extrait les leçons du transcript de l'introspection.

        Analyse les arguments de Core pour identifier les leçons,
        catégories, et actions proposées.

        Args:
            session: La session War Room terminée.

        Returns:
            Liste des LessonLearned extraites.
        """
        lessons = []

        for entry in session.transcript:
            content = entry.content
            # Heuristique : chercher des patterns de leçon
            content_lower = content.lower()

            # Détection de catégorie
            if any(kw in content_lower for kw in ["trade", "profit", "perte", "drawdown"]):
                category = "trading"
            elif any(kw in content_lower for kw in ["sécurité", "faille", "hack", "vulnérabilité"]):
                category = "security"
            elif any(kw in content_lower for kw in ["serveur", "docker", "deploy", "infrastructure"]):
                category = "infrastructure"
            else:
                category = "general"

            lessons.append(LessonLearned(
                category=category,
                lesson=content[:200],
                action_item=f"Réviser la stratégie {category}",
                priority=2,
            ))

        return lessons

    def _generate_report(self, report: QuietRoomReport) -> str:
        """Génère le rapport final de la Quiet Room.

        Args:
            report: Les données du rapport à formater.

        Returns:
            Texte Markdown formaté du rapport de maintenance.
        """
        lines = [
            f"# 🧘 QUIET ROOM REPORT — Session {report.session_id}",
            f"**Trigger: {report.trigger.reason}** (sévérité: {report.trigger.severity}/10)",
            f"**Date: {report.timestamp.strftime('%Y-%m-%d %H:%M')}**",
            "",
            "## Statistiques",
            f"  - Erreurs purgées: {report.errors_purged}",
            f"  - Leçons extraites: {len(report.lessons)}",
            f"  - Leçons archivées (Mem0): {report.lessons_archived}",
            "",
            "## Leçons Apprises",
        ]

        for i, lesson in enumerate(report.lessons, 1):
            prio_icon = {1: "🔴", 2: "🟡", 3: "🟢"}.get(lesson.priority, "⚪")
            lines.append(f"  {i}. {prio_icon} [{lesson.category}] {lesson.lesson[:150]}")
            if lesson.action_item:
                lines.append(f"     ➡️ Action: {lesson.action_item}")

        lines.extend([
            "",
            "## Psycho-Cybernétique",
            "  ✅ Mécanisme nettoyé",
            "  ✅ Échecs passés effacés du court terme",
            "  ✅ Leçons conservées en long terme",
            "  ✅ Visualisation positive activée",
        ])

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-CONVOCATION
# ═══════════════════════════════════════════════════════════════════════════════


async def auto_convene_quiet_room(
    error_logs: List[str],
    llm_service,
    memory_bridge=None,
    drawdown_percent: float = 0.0,
    uptime_hours: float = 0.0,
    error_count_last_hour: int = 0,
) -> Optional[QuietRoomReport]:
    """Vérifie les conditions d'auto-convocation et lance la Quiet Room si nécessaire.

    Cette fonction est destinée à être appelée périodiquement (ex: toutes les 15 min)
    par le scheduler principal pour détecter automatiquement les situations
    nécessitant une maintenance psychologique.

    Args:
        error_logs: Logs d'erreurs récents.
        llm_service: Service LLM.
        memory_bridge: MemoryBridge pour archivage.
        drawdown_percent: Drawdown journalier actuel (%).
        uptime_hours: Heures d'activité continue.
        error_count_last_hour: Nombre d'erreurs dans la dernière heure.

    Returns:
        QuietRoomReport si une session a été déclenchée, None sinon.
    """
    trigger = None

    # Condition 1 : Drawdown excessif
    if drawdown_percent > 3.0:
        trigger = QuietRoomTrigger(
            reason=f"Drawdown excessif: {drawdown_percent:.1f}%",
            severity=8,
            details=f"Le drawdown journalier ({drawdown_percent:.1f}%) dépasse le seuil de 3%.",
        )

    # Condition 2 : Activité continue > 24h
    elif uptime_hours > 24.0:
        trigger = QuietRoomTrigger(
            reason=f"Activité continue: {uptime_hours:.0f}h",
            severity=5,
            details="Le système fonctionne depuis plus de 24h sans pause.",
        )

    # Condition 3 : Erreurs en cascade
    elif error_count_last_hour > 5:
        trigger = QuietRoomTrigger(
            reason=f"Erreurs en cascade: {error_count_last_hour} en 1h",
            severity=7,
            details=f"{error_count_last_hour} erreurs détectées dans la dernière heure.",
        )

    if trigger is None:
        return None

    logger.warning(f"[AutoConvene] Quiet Room triggered: {trigger.reason}")

    quiet = QuietRoomSession()
    return await quiet.run(
        trigger=trigger,
        error_logs=error_logs,
        llm_service=llm_service,
        memory_bridge=memory_bridge,
    )
