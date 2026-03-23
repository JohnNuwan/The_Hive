"""Ingestion des contenus Biblio_IA dans la memoire vectorielle/graph."""

import asyncio
import glob
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Permet l'import de shared sans installation globale.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "shared"))

from shared.memory_bridge import get_memory_bridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_biblio_path() -> Path | None:
    """Resout le dossier source de Biblio_IA.

    Priorite:
    1) variable BIBLIO_IA_PATH
    2) <repo>/Biblio_IA
    3) <repo>/Documentation/Biblio_IA
    """
    env_path = os.getenv("BIBLIO_IA_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(REPO_ROOT / "Biblio_IA")
    candidates.append(REPO_ROOT / "Documentation" / "Biblio_IA")

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


async def main() -> None:
    """Ingere tous les .md de Biblio_IA vers la memoire d'EVA."""
    source_dir = resolve_biblio_path()
    if source_dir is None:
        logger.error(
            "Dossier Biblio_IA introuvable. Definis BIBLIO_IA_PATH ou place le dossier en racine/Documentation."
        )
        return

    logger.info(f"Demarrage ingestion Biblio_IA depuis: {source_dir}")
    bridge = get_memory_bridge()

    md_files = glob.glob(str(source_dir / "**" / "*.md"), recursive=True)
    logger.info(f"{len(md_files)} fichiers Markdown detectes.")

    ignored = {"readme.md", "index.md", "guide_navigation.md"}
    success_count = 0
    error_count = 0

    for file_path in tqdm(md_files, desc="Ingestion Biblio_IA"):
        try:
            filename = os.path.basename(file_path)
            if filename.lower() in ignored:
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                continue

            category = os.path.basename(os.path.dirname(file_path))
            metadata = {
                "source": "Biblio_IA",
                "filename": filename,
                "category": category,
                "type": "prompt_or_method",
            }
            content_with_context = (
                f"Titre: {filename}\n"
                f"Categorie: {category}\n\n"
                f"Contenu:\n{content}"
            )

            await bridge.add(content_with_context, user_id="system_biblio", metadata=metadata)
            success_count += 1

        except Exception as e:
            logger.error(f"Erreur ingestion {file_path}: {e}")
            error_count += 1

    logger.info("Ingestion terminee.")
    logger.info(f"Fichiers indexes: {success_count}/{len(md_files)}")
    logger.info(f"Erreurs: {error_count}")

    try:
        stats = await bridge.get_stats()
        logger.info(f"Etat memoire: {stats}")
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())