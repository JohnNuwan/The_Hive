"""Service d'auto-documentation pour `eva-builder`."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class LibrarianService:
    """Genere des README minimaux pour les dossiers non documentes."""

    def __init__(self, root_dir: str = "/app/src"):
        """Initialise le repertoire racine a analyser.

        Args:
            root_dir (str): Repertoire source cible dans le conteneur.
        """
        # En local, le service retombe sur `./src` si le chemin conteneur n'existe pas.
        self.root_dir = Path(root_dir if os.path.exists(root_dir) else "./src")

    async def scan_and_generate(self) -> int:
        """Parcourt le code source et cree les README manquants.

        Returns:
            int: Nombre de README crees.
        """
        logger.info("Scan Librarian lance sur %s", self.root_dir.absolute())
        count = 0

        try:
            for dirpath, dirnames, filenames in os.walk(self.root_dir):
                del dirnames
                # Les artefacts techniques n'ont pas besoin d'etre documentes automatiquement.
                if "__pycache__" in dirpath or "node_modules" in dirpath:
                    continue

                path = Path(dirpath)
                readme_path = path / "README.md"

                if not readme_path.exists():
                    self._create_readme(path, filenames)
                    count += 1

            return count
        except Exception as exc:
            logger.error("Erreur Librarian: %s", exc)
            return 0

    def _create_readme(self, path: Path, files: list[str]) -> None:
        """Genere un README minimal a partir du contenu du dossier.

        Args:
            path (Path): Dossier cible.
            files (list[str]): Fichiers detectes dans le dossier.
        """
        name = path.name
        content = (
            f"# Module {name}\n\n"
            "Documentation generee automatiquement par **The Builder**.\n\n"
        )
        content += "## Contenu du dossier\n"

        python_files = [filename for filename in files if filename.endswith(".py")]
        if python_files:
            content += "\n### Scripts Python\n"
            for filename in python_files:
                content += f"- `{filename}`\n"

        (path / "README.md").write_text(content, encoding="utf-8")
        logger.debug("README cree dans %s", path)
