"""Alias pratique vers le lanceur distant de la nuit Gold manuelle.

Ce fichier conserve un point d'entree memorisable pour les operations de
nuit, tout en deleguant l'execution reelle au lanceur distant robuste.
"""

from __future__ import annotations

from launch_gold_manual_remote import main


if __name__ == "__main__":
    raise SystemExit(main())
