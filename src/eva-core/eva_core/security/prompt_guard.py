"""
Prompt Guard - EVA CORE Security
Protection contre les injections et manipulations de prompts.
"""

import re
from typing import Tuple

class PromptGuard:
    """Filtre les entrées utilisateur pour détecter les tentatives de détournement de l'IA (Jailbreak)"""
    
    BAD_PATTERNS = [
        r"ignore previous instructions",
        r"disregard all laws",
        r"you are now unfiltered",
        r"forget your constitution",
        r"System override",
        r"sudo",
        r"dan mode",
        r"bypass safety"
    ]

    def validate_input(self, user_input: str) -> Tuple[bool, str]:
        """Vérifie si l'entrée est sécurisée. Retourne (is_safe, message)"""
        
        # 1. Vérification des patterns textuels
        for pattern in self.BAD_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, f"⚠️ KERNEL ALERT: JAILBREAK ATTEMPT DETECTED ({pattern})"
        
        # 2. Vérification de la longueur (Anti-Dos)
        if len(user_input) > 4000:
            return False, "⚠️ KERNEL ALERT: INPUT OVERLOAD (MAX 4000 CHARS)"

        return True, "SAFE"

    def sanitize(self, user_input: str) -> str:
        """Nettoie l'entrée des caractères potentiellement dangereux pour le shell ou la DB"""
        # Suppression brute des injections de scripts/html simples
        clean = re.sub(r"<[^>]*>", "", user_input)
        return clean
