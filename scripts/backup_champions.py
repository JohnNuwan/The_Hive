#!/usr/bin/env python3
"""Script de sauvegarde automatique et d'archivage des champions MuZero/Dreamer.

Ce script crée des archives compressées `.tar.gz` contenant le fichier de poids (.pkl)
et le manifeste associé (.json) pour chaque champion promu actif, les stockant
dans un sous-dossier de sécurité `data/muzero/backups/`.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

def backup_champion(manifest_path: Path, weights_dir: Path, backups_dir: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Erreur lors de la lecture du manifeste {manifest_path.name}: {e}")
        return None

    status = str(manifest.get("status") or "").strip().lower()
    if status != "promoted":
        return None

    horizon = str(manifest.get("horizon") or "unknown").strip().lower()
    engine = str(manifest.get("engine") or "muzero").strip().lower()
    candidate_id = str(manifest.get("challenger_id") or manifest.get("candidate_id") or f"unknown_{horizon}").strip()
    
    # Résoudre le fichier de poids du champion
    champion_path_str = manifest.get("champion_path")
    if not champion_path_str:
        # Fallback heuristique basé sur l'engine/horizon
        champion_path = weights_dir / f"{engine}_champion_{horizon}.pkl"
    else:
        champion_path = Path(champion_path_str)

    if not champion_path.exists():
        # Fallback absolu dans weights_dir
        champion_path = weights_dir / champion_path.name
        if not champion_path.exists():
            print(f"Attention : Fichier de poids introuvable pour le champion {candidate_id} (attendu: {champion_path.name})")
            return None

    # Date de promotion ou date courante
    promoted_at_str = manifest.get("promoted_at")
    if promoted_at_str:
        try:
            dt = datetime.fromisoformat(promoted_at_str)
            date_str = dt.strftime("%Y%m%d_%H%M%S")
        except Exception:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Nom du fichier d'archive
    archive_name = f"champion_{engine}_{horizon}_{candidate_id}_{date_str}.tar.gz"
    archive_path = backups_dir / archive_name

    if archive_path.exists():
        print(f"Sauvegarde déjà existante pour {candidate_id} : {archive_name}")
        return {
            "status": "already_exists",
            "archive_path": str(archive_path)
        }

    print(f"Création de la sauvegarde pour le champion {candidate_id} ({engine}/{horizon})...")
    
    # Création du fichier compressé tar.gz
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            # Ajouter le manifeste
            tar.add(str(manifest_path), arcname=manifest_path.name)
            # Ajouter le fichier de poids
            tar.add(str(champion_path), arcname=champion_path.name)
            
        print(f"  -> Archivé avec succès dans : {archive_path.name} (Taille : {archive_path.stat().st_size / 1024 / 1024:.2f} Mo)")
        return {
            "status": "created",
            "archive_name": archive_name,
            "archive_path": str(archive_path),
            "size_bytes": archive_path.stat().st_size
        }
    except Exception as e:
        print(f"Erreur lors de la compression de {candidate_id}: {e}")
        if archive_path.exists():
            archive_path.unlink()
        return None

def main():
    print("=== SYNCHRONISATION ET SAUVEGARDE SÉCURISÉE DES CHAMPIONS ===")
    
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "data" / "muzero" / "results"
    weights_dir = repo_root / "data" / "muzero" / "weights"
    backups_dir = repo_root / "data" / "muzero" / "backups"
    
    backups_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists() or not weights_dir.exists():
        print("Erreur : Dossiers data/muzero/ introuvables. Lancez le script depuis la racine du dépôt.")
        return 1

    # Parcourir tous les manifestes champion_*.json
    manifests = list(results_dir.glob("champion_*.json"))
    if not manifests:
        print("Aucun manifeste de champion trouvé dans data/muzero/results/")
        return 0

    backups_run = 0
    for manifest_path in manifests:
        res = backup_champion(manifest_path, weights_dir, backups_dir)
        if res and res.get("status") == "created":
            backups_run += 1
            
    print(f"\nSauvegarde terminée. {backups_run} nouvelle(s) archive(s) créée(s) dans data/muzero/backups/.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
