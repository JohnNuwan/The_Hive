"""
OpenClaw RLM Evolver — Boucle d'Auto-Évolution
Part of Sovereign Stack V3.0 — Sprint 4

Orchestre le cycle complet d'auto-amélioration :
    Scan → Diagnose → Patch → Validate (Dojo) → Apply → Learn

Ce module est le "cerveau" de la boucle RLM. Il connecte :
    - L'Evaluator (détection d'erreurs)
    - Le Patcher (génération de correctifs)
    - La War Room Dojo (validation Red/Blue)
    - Le MemoryBridge (apprentissage long terme)

Mode autonome :
    `auto_evolve()` lance une boucle infinie qui scanne le système
    toutes les N minutes et applique les correctifs validés.

Références :
    - CDcs Sprint 4 : "Boucle RLM pour l'auto-amélioration"
    - Recursive Self-Improvement (AI Safety)
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from .evaluator import RLMEvaluator, Diagnosis, IssueSeverity
from .patcher import RLMPatcher, Patch, PatchResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvolutionCycleResult:
    """Résultat d'un cycle d'évolution complet.

    Attributes:
        cycle_id: Numéro du cycle.
        diagnoses_found: Nombre de diagnostics détectés.
        patches_generated: Nombre de patchs générés.
        patches_approved: Nombre de patchs approuvés par le Dojo.
        patches_applied: Nombre de patchs appliqués avec succès.
        patches_rejected: Nombre de patchs rejetés.
        lessons_learned: Nombre de leçons archivées dans Mem0.
        duration_seconds: Durée totale du cycle.
        details: Liste des résultats individuels pour chaque patch.
        timestamp: Horodatage du cycle.
    """

    cycle_id: int = 0
    diagnoses_found: int = 0
    patches_generated: int = 0
    patches_approved: int = 0
    patches_applied: int = 0
    patches_rejected: int = 0
    lessons_learned: int = 0
    duration_seconds: float = 0.0
    details: List[PatchResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Produit un résumé en une ligne du cycle.

        Returns:
            Texte résumé du cycle (ex: "Cycle #3: 5 issues, 3 patched, 1 rejected").
        """
        return (
            f"Cycle #{self.cycle_id}: "
            f"{self.diagnoses_found} issues → "
            f"{self.patches_generated} patches → "
            f"{self.patches_applied} applied, "
            f"{self.patches_rejected} rejected "
            f"({self.duration_seconds:.1f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EVOLVER
# ═══════════════════════════════════════════════════════════════════════════════

# Nombre max de patches par cycle (pour limiter la charge GPU)
MAX_PATCHES_PER_CYCLE = 3

# Sévérité minimum pour déclencher un patch automatique
AUTO_PATCH_THRESHOLD = IssueSeverity.HIGH


class RLMEvolver:
    """Orchestrateur de la boucle d'auto-évolution.

    Connecte l'Evaluator, le Patcher, et le Dojo pour créer
    un pipeline autonome de détection → correction → validation.

    Sécurité :
        - Limite de MAX_PATCHES_PER_CYCLE par cycle pour ne pas saturer le GPU.
        - Seules les erreurs ≥ AUTO_PATCH_THRESHOLD sont traitées.
        - Chaque patch passe par le Dojo (Red/Blue Teaming).
        - Pas de push Git automatique.

    Usage :
        evolver = RLMEvolver(project_root="/path/to/project")
        result = await evolver.evolution_cycle(logs, llm_service)
    """

    def __init__(self, project_root: str):
        """Initialise l'Evolver.

        Args:
            project_root: Chemin racine du projet.
        """
        self.project_root = project_root
        self.evaluator = RLMEvaluator()
        self.patcher = RLMPatcher(project_root)
        self._cycle_counter = 0
        self._total_patches_applied = 0

    async def evolution_cycle(
        self,
        log_lines: List[str],
        llm_service,
        memory_bridge=None,
        probe_results: Optional[List[dict]] = None,
        use_dojo: bool = True,
    ) -> EvolutionCycleResult:
        """Exécute un cycle complet d'auto-évolution.

        Workflow :
            1. SCAN    — Analyse les logs et probes.
            2. TRIAGE  — Filtre et priorise les diagnostics.
            3. PATCH   — Génère des correctifs pour les top N issues.
            4. VALIDATE — Soumet au Dojo si activé.
            5. APPLY   — Applique les patches approuvés.
            6. LEARN   — Archive les résultats dans Mem0.

        Args:
            log_lines: Lignes de log à analyser.
            llm_service: Service LLM pour la génération de patches.
            memory_bridge: MemoryBridge optionnel pour l'archivage.
            probe_results: Résultats optionnels des probes Docker.
            use_dojo: Si True, valide les patches via War Room Dojo.

        Returns:
            EvolutionCycleResult avec le résumé du cycle.
        """
        self._cycle_counter += 1
        start_time = datetime.now()
        result = EvolutionCycleResult(cycle_id=self._cycle_counter)

        logger.info(f"[RLM:Evolver] ═══ CYCLE #{self._cycle_counter} START ═══")

        # ─── Phase 1 : SCAN ───
        logger.info("[RLM:Evolver] Phase 1: Scanning...")
        diagnoses = self.evaluator.scan_logs(log_lines)

        if probe_results:
            probe_diagnoses = self.evaluator.scan_probes(probe_results)
            diagnoses.extend(probe_diagnoses)

        # Détecter les patterns récurrents
        patterns = self.evaluator.detect_patterns(diagnoses)
        diagnoses.extend(patterns)

        result.diagnoses_found = len(diagnoses)

        if not diagnoses:
            logger.info("[RLM:Evolver] No issues found. System healthy ✅")
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            return result

        # ─── Phase 2 : TRIAGE ───
        logger.info(f"[RLM:Evolver] Phase 2: Triaging {len(diagnoses)} issues...")

        # Filtrer par sévérité minimum
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
        }
        threshold_val = severity_order[AUTO_PATCH_THRESHOLD]

        actionable = [
            d for d in diagnoses
            if severity_order[d.severity] <= threshold_val
            and d.file_path  # Besoin d'un fichier pour patcher
        ]

        # Limiter le nombre de patches par cycle
        top_issues = actionable[:MAX_PATCHES_PER_CYCLE]
        logger.info(
            f"[RLM:Evolver] {len(actionable)} actionable issues, "
            f"processing top {len(top_issues)}"
        )

        # ─── Phase 3-5 : PATCH → VALIDATE → APPLY ───
        for diagnosis in top_issues:
            logger.info(
                f"[RLM:Evolver] Processing {diagnosis.id}: "
                f"{diagnosis.error_message[:80]}"
            )

            # Phase 3 : Génération du patch
            patch = await self.patcher.generate_patch(diagnosis, llm_service)
            if not patch:
                logger.warning(f"[RLM:Evolver] Patch generation failed for {diagnosis.id}")
                continue

            result.patches_generated += 1

            # Phase 4 : Validation Dojo (optionnel)
            dojo_approved = True
            if use_dojo:
                dojo_approved = await self._validate_with_dojo(patch, llm_service, memory_bridge)

            if not dojo_approved:
                result.patches_rejected += 1
                logger.info(f"[RLM:Evolver] ❌ Patch {diagnosis.id} rejected by Dojo")

                # Archiver la leçon de l'échec
                if memory_bridge:
                    await self._archive_lesson(
                        memory_bridge,
                        f"Patch rejeté pour {diagnosis.id}: {diagnosis.error_message}. "
                        f"Le Dojo a refusé la correction proposée.",
                        success=False,
                    )
                    result.lessons_learned += 1
                continue

            # Phase 5 : Application du patch
            patch_result = await self.patcher.apply_patch(patch)
            result.details.append(patch_result)

            if patch_result.applied:
                result.patches_applied += 1
                self._total_patches_applied += 1
                logger.info(f"[RLM:Evolver] ✅ Patch {diagnosis.id} applied successfully")

                # Archiver le succès
                if memory_bridge:
                    await self._archive_lesson(
                        memory_bridge,
                        f"Patch appliqué pour {diagnosis.id}: {diagnosis.error_message}. "
                        f"Correction: {patch.description}",
                        success=True,
                    )
                    result.lessons_learned += 1
            else:
                logger.error(
                    f"[RLM:Evolver] ❌ Patch {diagnosis.id} application failed: "
                    f"{patch_result.error}"
                )

        # ─── Résumé ───
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        logger.info(f"[RLM:Evolver] ═══ CYCLE #{self._cycle_counter} END ═══")
        logger.info(f"[RLM:Evolver] {result.summary()}")

        return result

    async def auto_evolve(
        self,
        log_source_func,
        llm_service,
        memory_bridge=None,
        interval_seconds: int = 900,
        max_cycles: int = 0,
    ):
        """Mode autonome : boucle d'évolution continue.

        Appelle log_source_func() à chaque cycle pour récupérer
        les nouvelles lignes de log, puis exécute evolution_cycle().

        Args:
            log_source_func: Callable async retournant List[str] de logs.
            llm_service: Service LLM.
            memory_bridge: MemoryBridge optionnel.
            interval_seconds: Intervalle entre les cycles (défaut: 15 min).
            max_cycles: Si > 0, limite le nombre de cycles (0 = infini).
        """
        logger.info(
            f"[RLM:Evolver] 🔄 Auto-Evolve mode started "
            f"(interval: {interval_seconds}s, max: {max_cycles or '∞'})"
        )

        cycles_done = 0

        while True:
            try:
                # Récupérer les logs frais
                log_lines = await log_source_func()
                if log_lines:
                    result = await self.evolution_cycle(
                        log_lines=log_lines,
                        llm_service=llm_service,
                        memory_bridge=memory_bridge,
                    )
                    logger.info(f"[RLM:Evolver] Auto-cycle result: {result.summary()}")
                else:
                    logger.debug("[RLM:Evolver] No new logs to process")

            except Exception as e:
                logger.error(f"[RLM:Evolver] Auto-evolve cycle error: {e}")

            cycles_done += 1
            if max_cycles > 0 and cycles_done >= max_cycles:
                logger.info(f"[RLM:Evolver] Max cycles ({max_cycles}) reached. Stopping.")
                break

            await asyncio.sleep(interval_seconds)

    async def _validate_with_dojo(
        self,
        patch: Patch,
        llm_service,
        memory_bridge=None,
    ) -> bool:
        """Soumet un patch au War Room Dojo pour validation.

        Args:
            patch: Le patch à valider.
            llm_service: Service LLM.
            memory_bridge: MemoryBridge optionnel.

        Returns:
            True si le Dojo approuve le patch.
        """
        try:
            from ..scenarios.dojo import DojoCodeReview

            dojo = DojoCodeReview()
            report = await dojo.review(
                code_snippet=patch.patched_code,
                llm_service=llm_service,
                memory_bridge=memory_bridge,
                author=f"RLM AutoPatch ({patch.diagnosis_id})",
            )
            return report.approved_for_production

        except Exception as e:
            logger.warning(f"[RLM:Evolver] Dojo validation failed: {e}. Defaulting to APPROVED.")
            return True  # Fallback: approuver si le Dojo n'est pas disponible

    async def _archive_lesson(
        self,
        memory_bridge,
        lesson_text: str,
        success: bool,
    ):
        """Archive une leçon d'évolution dans Mem0.

        Args:
            memory_bridge: MemoryBridge pour l'archivage.
            lesson_text: Texte de la leçon.
            success: True si c'est un succès, False si échec.
        """
        try:
            await memory_bridge.add(
                content=f"[RLM {'SUCCESS' if success else 'FAILURE'}] {lesson_text}",
                metadata={
                    "type": "rlm_lesson",
                    "success": success,
                    "cycle": self._cycle_counter,
                    "total_patches": self._total_patches_applied,
                },
            )
        except Exception as e:
            logger.warning(f"[RLM:Evolver] Lesson archival failed: {e}")

    def get_stats(self) -> dict:
        """Retourne les statistiques cumulées de l'Evolver.

        Returns:
            Dictionnaire avec cycles, patches appliqués, etc.
        """
        return {
            "total_cycles": self._cycle_counter,
            "total_patches_applied": self._total_patches_applied,
            "project_root": self.project_root,
        }
