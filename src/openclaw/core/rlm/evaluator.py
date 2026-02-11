"""
OpenClaw RLM Evaluator — Module de Diagnostic Autonome
Part of Sovereign Stack V3.0 — Sprint 4

Analyse les logs, probes système et métriques pour détecter :
    - Tracebacks Python et erreurs non-gérées.
    - Latence anormale (> seuil configurable).
    - Containers Docker unhealthy / crashed.
    - Patterns d'erreurs récurrents.

Chaque anomalie détectée produit un `Diagnosis` structuré contenant
le fichier suspect, la cause probable, et un indice de correction.

Références :
    - CDcs Sprint 4 : "Auto-Réparation via Gemma 3"
    - ROADMAP : "Boucle RLM pour l'auto-amélioration"
"""

import logging
import re
import os
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class IssueSeverity(Enum):
    """Niveau de sévérité d'une anomalie détectée."""

    CRITICAL = "critical"   # Crash, data loss, security breach
    HIGH = "high"           # Service down, fonctionnalité cassée
    MEDIUM = "medium"       # Dégradation performance, warning récurrent
    LOW = "low"             # Cosmétique, log verbeux, optimisation


class IssueCategory(Enum):
    """Catégorie d'anomalie."""

    TRACEBACK = "traceback"           # Exception Python non-gérée
    LATENCY = "latency"               # Réponse > seuil (ex: >5s)
    CONTAINER_CRASH = "container"     # Docker container unhealthy/exited
    PATTERN_ERROR = "pattern_error"   # Même erreur > N fois
    IMPORT_ERROR = "import_error"     # Module manquant
    RESOURCE_EXHAUSTION = "resource"  # OOM, disk full, connection pool


@dataclass
class Diagnosis:
    """Résultat structuré de l'analyse d'une anomalie.

    Attributes:
        id: Identifiant unique du diagnostic.
        category: Type d'anomalie (traceback, latence, crash).
        severity: Niveau de sévérité.
        file_path: Chemin du fichier suspect (si identifiable).
        line_number: Numéro de ligne (si identifiable).
        error_message: Message d'erreur brut.
        cause: Cause probable identifiée par l'évaluateur.
        fix_hint: Indication de correction pour le Patcher.
        raw_log: Log brut associé.
        timestamp: Horodatage de la détection.
    """

    id: str = ""
    category: IssueCategory = IssueCategory.TRACEBACK
    severity: IssueSeverity = IssueSeverity.MEDIUM
    file_path: str = ""
    line_number: int = 0
    error_message: str = ""
    cause: str = ""
    fix_hint: str = ""
    raw_log: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

# Regex patterns pour la détection
RE_TRACEBACK = re.compile(
    r'File "([^"]+)", line (\d+), in (\w+)\n\s*(.*)',
    re.MULTILINE,
)
RE_IMPORT_ERROR = re.compile(r"(ModuleNotFoundError|ImportError): (.+)")
RE_OOM = re.compile(r"(MemoryError|Out of memory|OOM|CUDA out of memory)", re.IGNORECASE)
RE_CONNECTION = re.compile(r"(ConnectionRefusedError|ConnectionResetError|ConnectionError|TimeoutError)")


class RLMEvaluator:
    """Évaluateur autonome pour la détection d'anomalies.

    Scanne les logs applicatifs, les probes Docker, et les métriques
    système pour produire une liste de diagnostics structurés.

    Ce module est le "capteur" de la boucle RLM : il ne corrige rien,
    il détecte et diagnostique.

    Usage :
        evaluator = RLMEvaluator()
        diagnoses = evaluator.scan_logs(log_lines)
    """

    def __init__(self, latency_threshold_ms: float = 5000.0):
        """Initialise l'évaluateur.

        Args:
            latency_threshold_ms: Seuil de latence en millisecondes.
                Au-delà, une anomalie LATENCY est créée.
        """
        self.latency_threshold_ms = latency_threshold_ms
        self._diagnosis_counter = 0

    def scan_logs(self, log_lines: List[str]) -> List[Diagnosis]:
        """Scanne une liste de lignes de log et extrait les diagnostics.

        Analyse séquentielle : cherche les tracebacks, import errors,
        OOM, et erreurs de connexion dans les logs.

        Args:
            log_lines: Liste brute de lignes de log.

        Returns:
            Liste des Diagnosis détectés, triés par sévérité.
        """
        diagnoses: List[Diagnosis] = []
        full_text = "\n".join(log_lines)

        # ─── Tracebacks Python ───
        for match in RE_TRACEBACK.finditer(full_text):
            file_path, line_num, func_name, error_line = match.groups()
            self._diagnosis_counter += 1
            diagnoses.append(Diagnosis(
                id=f"DIAG-{self._diagnosis_counter:04d}",
                category=IssueCategory.TRACEBACK,
                severity=IssueSeverity.HIGH,
                file_path=file_path,
                line_number=int(line_num),
                error_message=error_line.strip(),
                cause=f"Exception non-gérée dans {func_name}()",
                fix_hint=f"Ajouter un try/except dans {func_name}() à la ligne {line_num}",
                raw_log=match.group(0),
            ))

        # ─── Import Errors ───
        for match in RE_IMPORT_ERROR.finditer(full_text):
            error_type, module_name = match.groups()
            self._diagnosis_counter += 1
            diagnoses.append(Diagnosis(
                id=f"DIAG-{self._diagnosis_counter:04d}",
                category=IssueCategory.IMPORT_ERROR,
                severity=IssueSeverity.HIGH,
                error_message=f"{error_type}: {module_name}",
                cause=f"Module '{module_name}' manquant ou mal installé",
                fix_hint=f"pip install {module_name.split('.')[0]} ou vérifier le PYTHONPATH",
                raw_log=match.group(0),
            ))

        # ─── OOM / Resource Exhaustion ───
        for match in RE_OOM.finditer(full_text):
            self._diagnosis_counter += 1
            diagnoses.append(Diagnosis(
                id=f"DIAG-{self._diagnosis_counter:04d}",
                category=IssueCategory.RESOURCE_EXHAUSTION,
                severity=IssueSeverity.CRITICAL,
                error_message=match.group(0),
                cause="Mémoire insuffisante (RAM ou VRAM)",
                fix_hint="Réduire le batch size, libérer la VRAM, ou ajouter du swap",
                raw_log=match.group(0),
            ))

        # ─── Connection Errors ───
        for match in RE_CONNECTION.finditer(full_text):
            self._diagnosis_counter += 1
            diagnoses.append(Diagnosis(
                id=f"DIAG-{self._diagnosis_counter:04d}",
                category=IssueCategory.TRACEBACK,
                severity=IssueSeverity.MEDIUM,
                error_message=match.group(0),
                cause="Service distant non-joignable (Redis, Qdrant, Neo4j ?)",
                fix_hint="Vérifier que le service est lancé et joignable",
                raw_log=match.group(0),
            ))

        # Trier par sévérité (CRITICAL > HIGH > MEDIUM > LOW)
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
        }
        diagnoses.sort(key=lambda d: severity_order[d.severity])

        logger.info(f"[RLM:Evaluator] Scanned {len(log_lines)} lines → {len(diagnoses)} issues found")
        return diagnoses

    def scan_probes(self, probe_results: List[dict]) -> List[Diagnosis]:
        """Scanne les résultats des health probes Docker.

        Args:
            probe_results: Liste de dicts avec "name", "status", "latency_ms".

        Returns:
            Liste des Diagnosis pour les probes en échec ou lentes.
        """
        diagnoses: List[Diagnosis] = []

        for probe in probe_results:
            name = probe.get("name", "unknown")
            status = probe.get("status", "unknown")
            latency = probe.get("latency_ms", 0)

            # Container crash / unhealthy
            if status not in ("running", "healthy"):
                self._diagnosis_counter += 1
                diagnoses.append(Diagnosis(
                    id=f"DIAG-{self._diagnosis_counter:04d}",
                    category=IssueCategory.CONTAINER_CRASH,
                    severity=IssueSeverity.CRITICAL,
                    error_message=f"Container '{name}' status: {status}",
                    cause=f"Le service {name} est tombé ou en état dégradé",
                    fix_hint=f"docker restart {name} ou vérifier les logs",
                ))

            # Latence excessive
            elif latency > self.latency_threshold_ms:
                self._diagnosis_counter += 1
                diagnoses.append(Diagnosis(
                    id=f"DIAG-{self._diagnosis_counter:04d}",
                    category=IssueCategory.LATENCY,
                    severity=IssueSeverity.MEDIUM,
                    error_message=f"Container '{name}' latency: {latency:.0f}ms",
                    cause=f"Le service {name} répond trop lentement",
                    fix_hint=f"Investiguer la charge CPU/RAM de {name}",
                ))

        logger.info(
            f"[RLM:Evaluator] Probed {len(probe_results)} services → "
            f"{len(diagnoses)} issues"
        )
        return diagnoses

    def detect_patterns(
        self,
        diagnoses: List[Diagnosis],
        threshold: int = 3,
    ) -> List[Diagnosis]:
        """Détecte les patterns d'erreurs récurrents.

        Si la même erreur apparaît ≥ threshold fois, crée un
        diagnostic de type PATTERN_ERROR avec sévérité augmentée.

        Args:
            diagnoses: Liste de diagnostics existants.
            threshold: Nombre minimum de récurrences.

        Returns:
            Liste des patterns détectés.
        """
        # Grouper par message d'erreur
        counts: dict = {}
        for d in diagnoses:
            key = d.error_message[:100]
            counts[key] = counts.get(key, 0) + 1

        patterns = []
        for msg, count in counts.items():
            if count >= threshold:
                self._diagnosis_counter += 1
                patterns.append(Diagnosis(
                    id=f"DIAG-{self._diagnosis_counter:04d}",
                    category=IssueCategory.PATTERN_ERROR,
                    severity=IssueSeverity.HIGH,
                    error_message=f"Pattern récurrent ({count}x): {msg}",
                    cause=f"L'erreur '{msg[:50]}...' se répète {count} fois",
                    fix_hint="Investigation approfondie requise — bug systémique probable",
                ))

        if patterns:
            logger.warning(f"[RLM:Evaluator] {len(patterns)} recurring patterns detected!")

        return patterns
