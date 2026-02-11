"""
OpenClaw RLM Patcher — Module de Génération de Correctifs
Part of Sovereign Stack V3.0 — Sprint 4

Génère des patches de code via le LLM (Gemma 3) pour corriger
les anomalies détectées par l'Evaluator. Chaque patch est validé
par la War Room Dojo avant application.

Workflow :
    1. Reçoit un Diagnosis de l'Evaluator.
    2. Lit le fichier source incriminé.
    3. Génère un diff/patch via le LLM.
    4. Soumet le patch au Dojo (Red/Blue Teaming).
    5. Si approuvé : applique le patch + git commit.
    6. Si rejeté : archive la leçon dans Mem0.

Sécurité :
    - Backup automatique avant chaque patch.
    - Rollback en cas d'erreur d'application.
    - Aucun push automatique (commit local uniquement).

Références :
    - CDcs Sprint 4 : "Auto-Réparation via Gemma 3"
    - War Room Dojo : Red/Blue Teaming pour validation
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

from .evaluator import Diagnosis, IssueSeverity

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Patch:
    """Un patch de code généré par le LLM.

    Attributes:
        diagnosis_id: ID du diagnostic qui a déclenché ce patch.
        file_path: Chemin du fichier à patcher.
        original_code: Code source original.
        patched_code: Code avec le correctif appliqué.
        diff_text: Représentation diff du patch (pour review humaine).
        description: Description textuelle de ce que fait le patch.
        confidence: Confiance du LLM dans le patch (0.0 → 1.0).
        timestamp: Horodatage de la génération.
    """

    diagnosis_id: str
    file_path: str
    original_code: str
    patched_code: str
    diff_text: str = ""
    description: str = ""
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PatchResult:
    """Résultat de l'application d'un patch.

    Attributes:
        patch: Le patch appliqué.
        applied: True si le patch a été appliqué avec succès.
        dojo_approved: True si le Dojo a approuvé le patch.
        rolled_back: True si un rollback a été effectué.
        error: Message d'erreur si échec.
        git_committed: True si le commit git a réussi.
    """

    patch: Patch
    applied: bool = False
    dojo_approved: bool = False
    rolled_back: bool = False
    error: str = ""
    git_committed: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# PATCHER
# ═══════════════════════════════════════════════════════════════════════════════

# Répertoire de backup (relatif au projet)
BACKUP_DIR = ".rlm_backups"


class RLMPatcher:
    """Générateur et applicateur de patches de code.

    Utilise le LLM pour générer des correctifs à partir d'un diagnostic,
    crée des backups de sécurité, et peut rollback en cas d'échec.

    Usage :
        patcher = RLMPatcher(project_root="/path/to/project")
        patch = await patcher.generate_patch(diagnosis, llm_service)
        result = await patcher.apply_patch(patch)
    """

    def __init__(self, project_root: str):
        """Initialise le Patcher.

        Args:
            project_root: Chemin racine du projet (pour les backups et git).
        """
        self.project_root = project_root
        self.backup_dir = os.path.join(project_root, BACKUP_DIR)
        os.makedirs(self.backup_dir, exist_ok=True)

    async def generate_patch(
        self,
        diagnosis: Diagnosis,
        llm_service,
    ) -> Optional[Patch]:
        """Génère un patch de code pour corriger une anomalie.

        Lit le fichier source, envoie le contexte au LLM, et parse
        le code corrigé retourné.

        Args:
            diagnosis: Le diagnostic à corriger.
            llm_service: Service LLM pour la génération.

        Returns:
            Patch généré, ou None si le fichier n'existe pas ou le LLM échoue.
        """
        if not diagnosis.file_path or not os.path.exists(diagnosis.file_path):
            logger.warning(
                f"[RLM:Patcher] Cannot patch: file '{diagnosis.file_path}' not found"
            )
            return None

        # Lire le fichier source
        try:
            with open(diagnosis.file_path, "r", encoding="utf-8") as f:
                original_code = f.read()
        except Exception as e:
            logger.error(f"[RLM:Patcher] Failed to read file: {e}")
            return None

        # Tronquer si trop long pour le context window
        if len(original_code) > 8000:
            logger.warning("[RLM:Patcher] File too large, truncating to 8000 chars")
            original_code = original_code[:8000]

        # Construire le prompt pour le LLM
        from shared import ChatMessage, Role

        prompt = (
            f"Tu es un développeur senior Python. Corrige le bug suivant:\n\n"
            f"DIAGNOSTIC:\n"
            f"  - Erreur: {diagnosis.error_message}\n"
            f"  - Fichier: {diagnosis.file_path}\n"
            f"  - Ligne: {diagnosis.line_number}\n"
            f"  - Cause: {diagnosis.cause}\n"
            f"  - Indice: {diagnosis.fix_hint}\n\n"
            f"CODE SOURCE ACTUEL:\n```python\n{original_code}\n```\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Retourne UNIQUEMENT le code Python corrigé complet.\n"
            f"2. Ne change que le minimum nécessaire pour corriger le bug.\n"
            f"3. Ajoute des commentaires `# RLM FIX:` pour marquer tes changements.\n"
            f"4. N'ajoute PAS de blocs markdown (pas de ```).\n"
            f"5. Décris ta correction en UNE LIGNE en commentaire tout en haut du fichier."
        )

        messages = [ChatMessage(role=Role.USER, content=prompt)]

        try:
            patched_code, thoughts = await llm_service.generate_response(
                messages=messages,
                system_prompt="Tu es un expert Python en auto-réparation de code. Sois minimal et précis.",
                role="coder",
                max_tokens=4000,
                temperature=0.2,  # Basse pour du code déterministe
            )
        except Exception as e:
            logger.error(f"[RLM:Patcher] LLM generation failed: {e}")
            return None

        # Nettoyer la réponse (parfois le LLM ajoute des backticks)
        patched_code = self._clean_code_response(patched_code)

        # Générer un diff textuel simple
        diff_text = self._generate_diff(original_code, patched_code, diagnosis.file_path)

        patch = Patch(
            diagnosis_id=diagnosis.id,
            file_path=diagnosis.file_path,
            original_code=original_code,
            patched_code=patched_code,
            diff_text=diff_text,
            description=f"Auto-fix for {diagnosis.id}: {diagnosis.error_message[:100]}",
        )

        logger.info(
            f"[RLM:Patcher] Patch generated for {diagnosis.id} "
            f"({len(patched_code)} chars)"
        )
        return patch

    async def apply_patch(
        self,
        patch: Patch,
        auto_commit: bool = True,
    ) -> PatchResult:
        """Applique un patch au système de fichiers.

        Crée un backup avant application, écrit le code patché,
        et effectue un git commit si demandé.

        Args:
            patch: Le patch à appliquer.
            auto_commit: Si True, fait un git commit après application.

        Returns:
            PatchResult indiquant le succès ou l'échec.
        """
        result = PatchResult(patch=patch)

        # ─── Backup ───
        try:
            self._create_backup(patch.file_path)
        except Exception as e:
            result.error = f"Backup failed: {e}"
            logger.error(f"[RLM:Patcher] {result.error}")
            return result

        # ─── Écriture du patch ───
        try:
            with open(patch.file_path, "w", encoding="utf-8") as f:
                f.write(patch.patched_code)
            result.applied = True
            logger.info(f"[RLM:Patcher] Patch applied to {patch.file_path}")
        except Exception as e:
            result.error = f"Write failed: {e}"
            logger.error(f"[RLM:Patcher] {result.error}")
            # Rollback automatique
            await self.rollback(patch.file_path)
            result.rolled_back = True
            return result

        # ─── Git Commit ───
        if auto_commit:
            try:
                import asyncio
                proc = await asyncio.create_subprocess_shell(
                    f'cd "{self.project_root}" && git add "{patch.file_path}" && '
                    f'git commit -m "[RLM Auto-Fix] {patch.diagnosis_id}: {patch.description[:50]}"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                result.git_committed = proc.returncode == 0
                if not result.git_committed:
                    logger.warning(f"[RLM:Patcher] Git commit failed: {stderr.decode()[:200]}")
            except Exception as e:
                logger.warning(f"[RLM:Patcher] Git commit exception: {e}")

        return result

    async def rollback(self, file_path: str) -> bool:
        """Restaure un fichier depuis son backup.

        Args:
            file_path: Chemin du fichier à restaurer.

        Returns:
            True si le rollback a réussi.
        """
        backup_path = self._get_backup_path(file_path)
        if not os.path.exists(backup_path):
            logger.error(f"[RLM:Patcher] No backup found for {file_path}")
            return False

        try:
            shutil.copy2(backup_path, file_path)
            logger.info(f"[RLM:Patcher] ↩️ Rollback success: {file_path}")
            return True
        except Exception as e:
            logger.error(f"[RLM:Patcher] Rollback failed: {e}")
            return False

    def _create_backup(self, file_path: str):
        """Crée un backup du fichier avant modification.

        Args:
            file_path: Chemin du fichier à sauvegarder.
        """
        backup_path = self._get_backup_path(file_path)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(file_path, backup_path)
        logger.debug(f"[RLM:Patcher] Backup created: {backup_path}")

    def _get_backup_path(self, file_path: str) -> str:
        """Calcule le chemin de backup pour un fichier.

        Args:
            file_path: Chemin du fichier original.

        Returns:
            Chemin du fichier de backup.
        """
        # Rendre le chemin relatif au projet
        rel_path = os.path.relpath(file_path, self.project_root)
        return os.path.join(self.backup_dir, rel_path + ".bak")

    def _clean_code_response(self, response: str) -> str:
        """Nettoie la réponse du LLM pour extraire le code pur.

        Supprime les markdown fences, les explications textuelles,
        et les artefacts de génération.

        Args:
            response: Réponse brute du LLM.

        Returns:
            Code Python nettoyé.
        """
        # Supprimer les fences markdown
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()

        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                return parts[1].strip()

        return response.strip()

    def _generate_diff(self, original: str, patched: str, file_path: str) -> str:
        """Génère un diff textuel simplifié entre deux versions.

        Args:
            original: Code original.
            patched: Code patché.
            file_path: Chemin du fichier (pour l'en-tête).

        Returns:
            Diff formaté en texte.
        """
        orig_lines = original.splitlines()
        patch_lines = patched.splitlines()

        diff_lines = [f"--- {file_path} (original)", f"+++ {file_path} (patched)"]

        max_lines = max(len(orig_lines), len(patch_lines))
        for i in range(max_lines):
            orig = orig_lines[i] if i < len(orig_lines) else ""
            patc = patch_lines[i] if i < len(patch_lines) else ""

            if orig != patc:
                if orig:
                    diff_lines.append(f"- {orig}")
                if patc:
                    diff_lines.append(f"+ {patc}")

        return "\n".join(diff_lines)
