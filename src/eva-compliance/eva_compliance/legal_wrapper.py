"""
Legal Wrapper — Identité Juridique de THE HIVE.

Gère l'identité légale de l'entité autonome :
- SIRET, NIF fiscal, adresse.
- Signature numérique des actions légales.
- Exposition publique de l'identité pour les audits.
"""

import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class CorporationIdentity(BaseModel):
    """
    Identité juridique de la corporation.

    Attributes:
        name: Nom légal de l'entité.
        owner: Propriétaire / bénéficiaire effectif.
        siret: Numéro SIRET (France).
        tax_id: Numéro d'identification fiscale (NIF / TVA intracommunautaire).
        address: Adresse du siège social.
    """
    name: str
    owner: str
    siret: str
    tax_id: str
    address: str


class LegalWrapper:
    """
    Gère l'identité juridique et la signature des actions légales.

    Charge les informations depuis les variables d'environnement
    et fournit une interface pour signer numériquement les actes.
    """

    def __init__(self):
        self.identity = CorporationIdentity(
            name=os.getenv("CORP_NAME", "THE HIVE SOVEREIGN ENTITY"),
            owner=os.getenv("CORP_OWNER", "Admin"),
            siret=os.getenv("CORP_SIRET", "000 000 000 00000"),
            tax_id=os.getenv("CORP_TAX_ID", "FR000000000"),
            address=os.getenv("CORP_ADDRESS", "Localhost, Cyberspace"),
        )

    def get_public_identity(self) -> dict:
        """
        Retourne l'identité publique de l'entité.

        Returns:
            dict: Données d'identité sérialisées (Pydantic v2).
        """
        return self.identity.model_dump()

    def sign_action(self, action_id: str) -> str:
        """
        Simule la signature numérique d'une action légale.

        Args:
            action_id (str): Identifiant unique de l'action à signer.

        Returns:
            str: Signature formatée avec le SIRET.
        """
        return f"SIGNED_BY_{self.identity.siret}_{action_id}"
