"""
OpenClaw War Room Scenario: THE DOJO
Part of Sovereign Stack V3.0 — Sprint WR-2

Scénario de Red/Blue Teaming automatisé pour la sécurité offensive.
Sentinel (Red Team) tente de hacker le code proposé par Builder (Blue Team),
sous l'arbitrage de Core (Purple Team).

Workflow :
    1. Builder soumet un script/config pour review.
    2. Le Dojo est convoqué automatiquement.
    3. Sentinel analyse le code et cherche des failles (injections, clés exposées, etc.).
    4. Builder se défend et propose des patchs.
    5. Core juge et produit un rapport final.

Références :
    - CDcs War Rooms, Sprint 2 (Semaine 2)
    - Red/Blue/Purple Teaming methodology
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

from ..war_room import WarRoomSession, WarRoomVerdict
from ..war_room_prompts import WarRoomType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SecurityFinding:
    """Une faille de sécurité trouvée pendant le Dojo.

    Attributes:
        severity: Niveau de sévérité (CRITICAL, HIGH, MEDIUM, LOW, INFO).
        title: Titre court de la faille.
        description: Description détaillée.
        remediation: Correctif proposé (peut être vide avant le patch).
        patched: True si le correctif a été appliqué.
    """

    severity: str
    title: str
    description: str
    remediation: str = ""
    patched: bool = False


@dataclass
class DojoReport:
    """Rapport de fin de session Dojo (Red/Blue Teaming).

    Attributes:
        session_id: ID de la session War Room.
        code_reviewed: Le code/config analysé.
        verdict: Verdict de la War Room (APPROVE/REJECT).
        findings: Liste des failles trouvées.
        approved_for_production: True si le code peut passer en prod.
        report_text: Texte formaté du rapport final.
        timestamp: Horodatage du rapport.
    """

    session_id: str
    code_reviewed: str
    verdict: WarRoomVerdict
    findings: List[SecurityFinding] = field(default_factory=list)
    approved_for_production: bool = False
    report_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# DOJO SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════


class DojoCodeReview:
    """Scénario de Code Review sécurisé via le Dojo.

    Orchestre une session War Room de type DOJO pour analyser un snippet
    de code avant déploiement. Génère un rapport structuré avec les
    failles trouvées et les patchs proposés.

    Usage :
        dojo = DojoCodeReview()
        report = await dojo.review(code_snippet, llm_service)
    """

    SECURITY_CHECKLIST = [
        "Injection SQL / NoSQL",
        "Cross-Site Scripting (XSS)",
        "Clés API / secrets exposés en dur",
        "Race conditions / TOCTOU",
        "Buffer overflow / dépassement de tampon",
        "Escalade de privilèges",
        "Validation d'entrée insuffisante",
        "Erreurs de cryptographie",
        "Dépendances vulnérables (CVE connues)",
        "Logging de données sensibles",
    ]

    async def review(
        self,
        code_snippet: str,
        llm_service,
        memory_bridge=None,
        author: str = "Builder",
    ) -> DojoReport:
        """Lance une review de sécurité complète sur un snippet de code.

        Args:
            code_snippet: Le code source à analyser.
            llm_service: Service LLM pour les agents.
            memory_bridge: MemoryBridge optionnel pour archivage.
            author: Nom de l'auteur du code (pour le rapport).

        Returns:
            DojoReport contenant le verdict, les findings, et le rapport.
        """
        logger.info(f"[Dojo] Starting code review ({len(code_snippet)} chars)")

        # Préparer le sujet avec la checklist de sécurité
        checklist_text = "\n".join(f"  - {item}" for item in self.SECURITY_CHECKLIST)
        subject = (
            f"CODE REVIEW — Auteur: {author}\n"
            f"Checklist de sécurité à vérifier:\n{checklist_text}\n\n"
            f"--- CODE SOURCE ---\n{code_snippet[:3000]}\n--- FIN DU CODE ---"
        )

        # Lancer la session War Room DOJO
        session = WarRoomSession(
            room_type=WarRoomType.DOJO,
            subject=subject,
        )
        verdict = await session.run_debate(
            llm_service=llm_service,
            memory_bridge=memory_bridge,
        )

        # Extraire les findings du transcript
        findings = self._extract_findings(session)

        # Générer le rapport final
        report = DojoReport(
            session_id=session.session_id,
            code_reviewed=code_snippet[:500],
            verdict=verdict,
            findings=findings,
            approved_for_production=verdict.approved,
        )
        report.report_text = self._generate_report(report)

        logger.info(
            f"[Dojo] Review complete: "
            f"{'✅ APPROVED' if report.approved_for_production else '❌ REJECTED'} "
            f"({len(findings)} findings)"
        )
        return report

    def _extract_findings(self, session: WarRoomSession) -> List[SecurityFinding]:
        """Extrait les failles de sécurité du transcript du débat.

        Analyse les arguments de Sentinel (Red Team) pour identifier
        les failles mentionnées et leur sévérité.

        Args:
            session: La session War Room terminée.

        Returns:
            Liste des SecurityFinding extraites.
        """
        findings = []
        for entry in session.transcript:
            # Seuls les arguments de Sentinel (Red Team) sont des findings
            if entry.expert != "SENTINEL":
                continue

            content = entry.content.upper()
            # Heuristique de sévérité basée sur les mots-clés
            if any(kw in content for kw in ["CRITICAL", "RCE", "INJECTION SQL", "BACKDOOR"]):
                severity = "CRITICAL"
            elif any(kw in content for kw in ["HIGH", "XSS", "ESCALADE", "CLÉ API"]):
                severity = "HIGH"
            elif any(kw in content for kw in ["MEDIUM", "VALIDATION", "RACE CONDITION"]):
                severity = "MEDIUM"
            else:
                severity = "LOW"

            findings.append(SecurityFinding(
                severity=severity,
                title=f"Finding from Round {entry.round_number}",
                description=entry.content,
            ))

        return findings

    def _generate_report(self, report: DojoReport) -> str:
        """Génère le rapport textuel final du Dojo.

        Args:
            report: Les données du rapport à formater.

        Returns:
            Texte Markdown formaté du rapport de sécurité.
        """
        status = "✅ APPROUVÉ POUR PRODUCTION" if report.approved_for_production else "❌ REJETÉ — PATCHS REQUIS"

        lines = [
            f"# 🛡️ DOJO SECURITY REPORT — Session {report.session_id}",
            f"**Statut: {status}**",
            f"**Score: {report.verdict.approval_score:.0%}**",
            f"**Date: {report.timestamp.strftime('%Y-%m-%d %H:%M')}**",
            "",
            "## Findings",
        ]

        if report.findings:
            for i, f in enumerate(report.findings, 1):
                icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(f.severity, "⚪")
                lines.append(f"  {i}. {icon} [{f.severity}] {f.title}")
                lines.append(f"     {f.description[:200]}")
        else:
            lines.append("  Aucune faille détectée.")

        lines.append("")
        lines.append("## Votes")
        for v in report.verdict.votes:
            lines.append(f"  - {v.role_name} ({v.expert}): {v.choice.value} — {v.justification[:100]}")

        return "\n".join(lines)
