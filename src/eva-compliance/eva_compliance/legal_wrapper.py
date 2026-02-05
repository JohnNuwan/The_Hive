import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class CorporationIdentity(BaseModel):
    name: str
    owner: str
    siret: str
    tax_id: str
    address: str

class LegalWrapper:
    """Gère l'identité juridique et la signature des actions"""
    
    def __init__(self):
        self.identity = CorporationIdentity(
            name=os.getenv("CORP_NAME", "THE HIVE SOVEREIGN ENTITY"),
            owner=os.getenv("CORP_OWNER", "Admin"),
            siret=os.getenv("CORP_SIRET", "000 000 000 00000"),
            tax_id=os.getenv("CORP_TAX_ID", "FR000000000"),
            address=os.getenv("CORP_ADDRESS", "Localhost, Cyberspace")
        )

    def get_public_identity(self):
        return self.identity.dict()

    def sign_action(self, action_id: str):
        """Simule la signature numérique d'une action légale"""
        return f"SIGNED_BY_{self.identity.siret}_{action_id}"
