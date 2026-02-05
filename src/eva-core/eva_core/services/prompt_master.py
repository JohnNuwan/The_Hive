import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PromptMaster:
    """
    LE MAÎTRE DES PROMPTS (PromptMaster)
    ----------------------------------
    Gère l'injection des méthodes Biblio_IA (BMAD, ReAct, CRITIC) 
    et des templates spécialisés pour chaque Expert.
    """

    def __init__(self, templates_dir: str = "Documentation/Biblio_IA"):
        self.templates_dir = templates_dir
        # Mapping logique -> Dossier réel dans Biblio_IA
        self.methods_map = {
            "react": "Méthode ReAct/METHODE_REACT_EXPLICATION.md",
            "critic": "Méthode CRITIC/METHODE_CRITIC_EXPLICATION.md",
            "bmad": "Méthode BMAD/METHODE_BMAD_COMPLETE.md",
            "tot": "Méthode Tree-of-Thoughts/METHODE_TREE_OF_THOUGHTS.md",
            "cot": "Méthode Chain-of-Thought/METHODE_COT_COMPLETE.md",
            "ltm": "Méthode Least-to-Most/METHODE_LEAST_TO_MOST.md",
            "stepback": "Méthode Step-Back/METHODE_STEP_BACK_EXPLICATION.md",
            "cod": "Méthode Contextual-Compression/METHODE_CONTEXTUAL_COMPRESSION.md", # Approximation CoD
        }
        logger.info("PromptMaster online (Git-Linked with Biblio_IA).")

    def _load_template(self, relative_path: str) -> str:
        """Charge un template depuis le dossier Documentation."""
        full_path = os.path.join(self.templates_dir, relative_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Error reading {full_path}: {e}")
        return ""

    def wrap_with_method(self, text: str, method: str = "costar") -> str:
        """
        Emballe un texte utilisateur avec une méthode de Biblio_IA.
        """
        method_path = self.methods_map.get(method.lower())
        method_template = self._load_template(method_path) if method_path else ""
        
        if not method_template:
            logger.debug(f"Method {method} not found in Git source.")
            return text
            
        return f"### PROTOCOLE {method.upper()} ORIGINEL (Biblio_IA)\n{method_template}\n\n### MISSION\n{text}"

    def get_expert_injector(self, expert_name: str) -> str:
        """
        Récupère l'injecteur depuis la bibliothèque de prompts clonée.
        """
        # Note: Bibliothèque-Prompts contient des sous-dossiers par thématique
        # On peut faire une recherche récursive ou un mapping direct
        mapping = {
            "banker": "Bibliothèque-Prompts/01_TECH_ET_CODE/01_DEV_PYTHON/Analyse_Code.md", # Exemple
            "builder": "Bibliothèque-Prompts/01_TECH_ET_CODE/01_DEV_PYTHON/Analyse_Code.md",
            "shadow": "Bibliothèque-Prompts/01_TECH_ET_CODE/06_CYBER_SECURITE/OSINT_Profil.md", # Supposition
        }
        
        rel_path = mapping.get(expert_name.lower())
        if rel_path:
            content = self._load_template(rel_path)
            if content:
                return content
                
        return f"Tu es Expert {expert_name}. Réfléchis étape par étape."

if __name__ == "__main__":
    # Test à blanc
    pm = PromptMaster(templates_dir="../../../Documentation/Biblio_IA")
    print(pm.wrap_with_method("Analyse le Nasdaq", method="react"))
