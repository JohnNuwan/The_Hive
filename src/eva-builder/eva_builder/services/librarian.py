"""
Service Librarian - Auto-Documentation THE HIVE
Convertit le code source en documentation Markdown
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class LibrarianService:
    """Gère l'auto-documentation du projet"""

    def __init__(self, root_dir: str = "/app/src"):
        # En dev local, on peut pointer vers le dossier actuel
        self.root_dir = Path(root_dir if os.path.exists(root_dir) else "./src")

    async def scan_and_generate(self) -> int:
        """Parcourt le code source et génère des README.md là où ils manquent"""
        logger.info(f"Scan Librarian lancé sur {self.root_dir.absolute()}")
        count = 0
        
        try:
            for dirpath, dirnames, filenames in os.walk(self.root_dir):
                # Ignorer dossiers techniques
                if "__pycache__" in dirpath or "node_modules" in dirpath:
                    continue
                
                path = Path(dirpath)
                readme_path = path / "README.md"
                
                if not readme_path.exists():
                    # Création d'un README minimaliste basé sur le dossier
                    self._create_readme(path, filenames)
                    count += 1
                    
            return count
        except Exception as e:
            logger.error(f"Librarian error: {e}")
            return 0

    def _create_readme(self, path: Path, files: list[str]):
        """Génère un fichier README.md automatique"""
        name = path.name
        content = f"# Module {name}\n\nDocumentation générée automatiquement par **The Builder**.\n\n"
        content += "## Contenu du dossier\n"
        
        python_files = [f for f in files if f.endswith(".py")]
        if python_files:
            content += "\n### Scripts Python\n"
            for f in python_files:
                content += f"- `{f}`\n"
                
        with open(path / "README.md", "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.debug(f"README créé dans {path}")
