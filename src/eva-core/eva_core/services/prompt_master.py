import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptMaster:
    """Charge et injecte des prompts/methodes depuis Biblio_IA."""

    def __init__(self, templates_dir: str):
        """Initialise le PromptMaster.

        Args:
            templates_dir: Repertoire principal de templates.
        """
        self.templates_dir = templates_dir
        self._search_roots = self._build_search_roots(templates_dir)

        # Mapping methode -> fichier attendu (recherche recursive autorisee)
        self.methods_map = {
            "react": "METHODE_REACT_EXPLICATION.md",
            "critic": "METHODE_CRITIC_EXPLICATION.md",
            "bmad": "METHODE_BMAD_COMPLETE.md",
            "tot": "METHODE_TREE_OF_THOUGHTS.md",
            "cot": "METHODE_COT_COMPLETE.md",
            "ltm": "METHODE_LEAST_TO_MOST.md",
            "stepback": "METHODE_STEP_BACK_EXPLICATION.md",
            "cod": "METHODE_CONTEXTUAL_COMPRESSION.md",
        }

        logger.info("PromptMaster actif (Biblio_IA).")

    def _build_search_roots(self, templates_dir: str) -> list[str]:
        """Construit la liste des repertoires de recherche de templates."""
        roots: list[str] = []

        def add(path: str) -> None:
            if path and path not in roots:
                roots.append(path)

        cwd = os.getcwd()
        add(templates_dir)
        add(os.path.abspath(templates_dir))
        add(os.path.join(cwd, templates_dir))
        add(os.path.join(cwd, "Biblio_IA"))
        add(os.path.join(cwd, "Documentation", "Biblio_IA"))

        return roots

    def _resolve_template_path(self, relative_path: str) -> str:
        """Resout un chemin de template en cherchant dans les repertoires connus."""
        if not relative_path:
            return ""

        # Priorite: comportement historique (join direct templates_dir + relative_path)
        direct_path = os.path.join(self.templates_dir, relative_path)
        if os.path.exists(direct_path):
            return direct_path

        # Si chemin relatif explicite, tester chaque racine
        if any(sep in relative_path for sep in ("/", "\\")):
            for root in self._search_roots:
                candidate = os.path.join(root, relative_path)
                if os.path.exists(candidate):
                    return candidate

        # Fallback: recherche recursive par nom de fichier
        basename = os.path.basename(relative_path)
        for root in self._search_roots:
            root_path = Path(root)
            if not root_path.exists() or not root_path.is_dir():
                continue
            matches = sorted(root_path.rglob(basename))
            if matches:
                return str(matches[0])

        return ""

    def _load_template(self, relative_path: str) -> str:
        """Charge un template depuis Biblio_IA.

        Args:
            relative_path: Chemin relatif ou nom de fichier.

        Returns:
            Contenu du template, sinon chaine vide.
        """
        full_path = self._resolve_template_path(relative_path)
        if full_path:
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Erreur lecture template {full_path}: {e}")
        return ""

    def wrap_with_method(self, text: str, method: str = "costar") -> str:
        """Emballe une requete avec un protocole de methode.

        Args:
            text: Texte utilisateur a traiter.
            method: Methode cible (react, bmad, critic, ...).

        Returns:
            Prompt enrichi avec la methode si disponible.
        """
        method_path = self.methods_map.get(method.lower())
        method_template = self._load_template(method_path) if method_path else ""

        if not method_template:
            logger.debug(f"Methode introuvable ou vide: {method}")
            return text

        return (
            f"### PROTOCOLE {method.upper()} ORIGINEL (Biblio_IA)\n"
            f"{method_template}\n\n"
            f"### MISSION\n{text}"
        )

    def get_expert_injector(self, expert_name: str) -> str:
        """Retourne un injecteur de contexte specialise pour un expert.

        Args:
            expert_name: Nom de l'expert cible.

        Returns:
            Prompt specialise si trouve, sinon fallback generique.
        """
        mapping = {
            "banker": "Analyse_Code.md",
            "builder": "Analyse_Code.md",
            "shadow": "OSINT_Profil.md",
            "core": "METHODE_COT_COMPLETE.md",
        }

        rel_path = mapping.get(expert_name.lower())
        if rel_path:
            content = self._load_template(rel_path)
            if content:
                return content

        return f"Tu es Expert {expert_name}. Reflechis etape par etape."


if __name__ == "__main__":
    pm = PromptMaster(templates_dir="Biblio_IA")
    print(pm.wrap_with_method("Analyse le Nasdaq", method="react"))