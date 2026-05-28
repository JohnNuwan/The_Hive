"""Controle l'hygiene des secrets dans les fichiers suivis par Git."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = {
    ".env.example",
    "Biblio_IA/Guides/08-Integrations.md",
}
IGNORE_PREFIXES = (
    "Documentation/",
    "Biblio_IA/",
    "src/eva-nervous/vendor/",
    "src/eva-nexus/node_modules/",
)
IGNORE_FILENAMES = {
    "fleet.config.example.json",
    "config.example.json",
}
SCANNED_SUFFIXES = {
    ".bat",
    ".cmd",
    ".env",
    ".go",
    ".ini",
    ".json",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "mot_de_passe": re.compile(
        r"(?im)^\s*[^#\n]{0,80}\b(password|mot[_ -]?de[_ -]?passe|passwd)\b[^=\n:]{0,40}[:=]\s*['\"]?[^$\\s'\"#][^'\"\n#]{3,}"
    ),
    "token_api": re.compile(
        r"(?im)^\s*[^#\n]{0,80}\b(api[_-]?key|token|secret)\b[^=\n:]{0,40}[:=]\s*['\"]?[^$\\s'\"#][^'\"\n#]{6,}"
    ),
    "ssh_hive": re.compile(
        r"(?im)^\s*(HIVE_SSH_PASSWORD|HIVE_SUDO_PASSWORD)\s*[:=]\s*['\"]?[^$\\s'\"#][^'\"\n#]{3,}"
    ),
    "connexion_mt5": re.compile(
        r"(?im)^\s*(MT5_(LOGIN|PASSWORD|SERVER)|\"mt5_(login|password|server)\")\s*[:=]\s*['\"]?[^$\\s'\"#][^'\"\n#]{1,}"
    ),
}


def _tracked_files() -> list[Path]:
    """Retourne la liste des fichiers suivis par Git."""

    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files: list[Path] = []
    for raw_line in result.stdout.splitlines():
        candidate = Path(raw_line.strip())
        if candidate.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        files.append(candidate)
    return files


def _is_allowlisted(path: Path) -> bool:
    """Indique si un chemin est explicitement autorise."""

    normalized = path.as_posix()
    if normalized in ALLOWLIST:
        return True
    if path.name in IGNORE_FILENAMES:
        return True
    if ".example." in path.name:
        return True
    return any(normalized.startswith(prefix) for prefix in IGNORE_PREFIXES)


def _scan_file(path: Path) -> list[str]:
    """Scanne un fichier suivi et retourne les alertes detectees."""

    absolute_path = ROOT / path
    if _is_allowlisted(path) or not absolute_path.exists() or absolute_path.is_dir():
        return []
    try:
        content = absolute_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    alerts: list[str] = []
    for rule_name, pattern in SECRET_PATTERNS.items():
        if pattern.search(content):
            alerts.append(rule_name)
    return alerts


def main() -> int:
    """Execute le controle et retourne un code process compatible CI."""

    failures: list[str] = []
    for path in _tracked_files():
        alerts = _scan_file(path)
        if not alerts:
            continue
        failures.append(f"{path.as_posix()}: {', '.join(alerts)}")

    if failures:
        print("Secrets potentiels detectes dans des fichiers suivis :")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("Aucun secret evident detecte dans les fichiers suivis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
