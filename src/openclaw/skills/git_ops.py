"""
OpenClaw Skill: Git Operations
Part of Sovereign Stack V3.0

Permet à l'agent de manipuler le code source (Self-Evolution).
"""

import asyncio
import subprocess
from .registry import skill
import logging

logger = logging.getLogger(__name__)

@skill("git_status", "Affiche l'état du repository (fichiers modifiés)")
async def git_status() -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            "git status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return f"Git Error: {stderr.decode()}"
        return stdout.decode()
    except Exception as e:
        return f"Execution Error: {e}"

@skill("git_diff", "Affiche les différences concrètes dans le code")
async def git_diff() -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            "git diff",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        return stdout.decode()[:2000] # Truncate pour éviter context overflow
    except Exception as e:
        return f"Execution Error: {e}"

@skill("git_commit", "Commit les changements avec un message (Auto-Coding)")
async def git_commit(message: str) -> str:
    """Ajoute tous les fichiers et commit"""
    try:
        # 1. Add
        await asyncio.create_subprocess_shell("git add .")
        
        # 2. Commit
        proc = await asyncio.create_subprocess_shell(
            f'git commit -m "[OpenClaw] {message}"',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            return f"Commit Failed (rien à commiter ?): {stdout.decode()} {stderr.decode()}"
            
        return f"Commit Success: {stdout.decode()}"
    except Exception as e:
        return f"Execution Error: {e}"
