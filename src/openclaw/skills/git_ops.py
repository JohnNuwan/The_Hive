"""
OpenClaw Skill: Git Operations
Part of Sovereign Stack V3.0

Permet à l'agent de manipuler le code source via Git (Self-Evolution).
Ces opérations sont essentielles pour le pipeline d'auto-amélioration
(Sprint 5: RLM Loop & Evolutionary Code).

Sécurité :
    - git_status et git_diff sont en lecture seule (safe).
    - git_commit effectue un `git add .` puis commit. Pas de push automatique
      pour éviter les déploiements accidentels.

Skills disponibles :
    - git_status : État du repository (fichiers modifiés).
    - git_diff   : Différences concrètes dans le code (truncated à 2000 chars).
    - git_commit : Commit avec message préfixé [OpenClaw].
"""

import asyncio
import logging
from .registry import skill

logger = logging.getLogger(__name__)


async def _run_git_command(command: str, truncate: int = 0) -> str:
    """Exécute une commande Git de manière asynchrone.

    Helper interne pour éviter la duplication de code entre les skills Git.

    Args:
        command: La commande Git à exécuter (ex: "git status").
        truncate: Si > 0, tronque la sortie à ce nombre de caractères.

    Returns:
        La sortie standard de la commande, ou un message d'erreur.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.warning(f"Git command failed: {command} -> {error_msg}")
            return f"Git Error: {error_msg}"

        output = stdout.decode().strip()
        if truncate > 0 and len(output) > truncate:
            output = output[:truncate] + "\n\n[... TRUNCATED ...]"
        return output

    except Exception as e:
        logger.error(f"Git execution error: {e}")
        return f"Execution Error: {e}"


@skill("git_status", "Affiche l'état du repository (fichiers modifiés)")
async def git_status() -> str:
    """Affiche l'état courant du repository Git.

    Équivalent de `git status`. Lecture seule, sans effet de bord.

    Returns:
        La sortie de `git status` ou un message d'erreur.
    """
    return await _run_git_command("git status")


@skill("git_diff", "Affiche les différences concrètes dans le code")
async def git_diff() -> str:
    """Affiche les modifications non-committées dans le repository.

    Équivalent de `git diff`. La sortie est tronquée à 2000 caractères
    pour éviter un context overflow dans le LLM.

    Returns:
        La sortie tronquée de `git diff` ou un message d'erreur.
    """
    return await _run_git_command("git diff", truncate=2000)


@skill("git_commit", "Commit les changements avec un message (Auto-Coding)")
async def git_commit(message: str) -> str:
    """Ajoute tous les fichiers modifiés et crée un commit.

    Le message de commit est automatiquement préfixé par "[OpenClaw]"
    pour identifier les commits générés par l'agent.

    Sécurité : Pas de `git push` automatique. Le push doit être
    déclenché manuellement ou par un workflow CI/CD validé.

    Args:
        message: Description des changements pour le message de commit.

    Returns:
        Confirmation du commit ou message d'erreur.
    """
    # BUG FIX: Attendre correctement la fin du `git add`
    add_result = await _run_git_command("git add .")
    if "Error" in add_result:
        return f"Git Add Failed: {add_result}"

    # Sanitize le message pour éviter les injections shell
    safe_message = message.replace('"', "'").replace("`", "'")
    return await _run_git_command(f'git commit -m "[OpenClaw] {safe_message}"')
