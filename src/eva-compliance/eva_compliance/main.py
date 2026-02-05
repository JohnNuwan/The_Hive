from fastapi import FastAPI
from eva_compliance.legal_wrapper import LegalWrapper
from eva_compliance.tax_estimator import TaxEstimator

app = FastAPI(title="EVA Compliance")
legal = LegalWrapper()
tax = TaxEstimator()

@app.get("/health")
async def health():
    return {"status": "online", "service": "compliance"}

@app.get("/identity")
async def get_identity():
    """Retourne l'identité juridique de la corporation"""
    return legal.get_public_identity()

@app.post("/estimate-tax")
async def estimate_tax(amount: float):
    """Estime et provisionne virtuellement les taxes"""
    return tax.calculate_provision(amount)
