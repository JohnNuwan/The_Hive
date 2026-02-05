from fastapi import FastAPI
from eva_substrate.energy_monitor import EnergyMonitor
from eva_substrate.circadian_rhythm import CircadianRhythm
from eva_substrate.resource_allocator import ResourceAllocator

app = FastAPI(title="EVA Substrate")
monitor = EnergyMonitor()
rhythm = CircadianRhythm()
allocator = ResourceAllocator()

@app.get("/health")
async def health():
    return {"status": "online", "service": "substrate"}

@app.get("/metrics")
async def get_metrics():
    """Retourne les métriques énergétiques et hardware"""
    return monitor.get_current_consumption()

@app.get("/mode")
async def get_mode():
    """Retourne le mode actuel (Jour/Nuit)"""
    return rhythm.get_current_mode()

@app.post("/allocate")
async def allocate_tpus(profile: str):
    """Alloue les TPUs selon un profil spécifique"""
    return allocator.set_profile(profile)
