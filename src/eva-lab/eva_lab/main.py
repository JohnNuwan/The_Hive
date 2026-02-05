from fastapi import FastAPI
from eva_lab.arena import Arena
from eva_lab.dreamer_model import DreamerModel
from eva_lab.genetic_updater import GeneticUpdater

app = FastAPI(title="EVA Lab")
arena = Arena()
dreamer = DreamerModel()
updater = GeneticUpdater()

@app.get("/health")
async def health():
    return {"status": "online", "service": "lab"}

@app.post("/run-sim")
async def run_simulation(algo_id: str):
    """Lance un combat de stratégies dans l'Arena"""
    return arena.run_combat(algo_id)

@app.get("/insights")
async def get_insights():
    """Récupère les prédictions du World Model (Dreamer)"""
    return dreamer.predict_future_market()

@app.post("/evolve")
async def trigger_evolution():
    """Déclenche la boucle génétique d'amélioration"""
    return updater.check_for_updates()
