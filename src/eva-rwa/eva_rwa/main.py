from fastapi import FastAPI
from eva_rwa.token_bridge import TokenBridge
from eva_rwa.iot_controller import IotController

app = FastAPI(title="EVA RWA")
bridge = TokenBridge()
iot = IotController()

@app.get("/health")
async def health():
    return {"status": "online", "service": "rwa"}

@app.get("/assets")
async def get_assets():
    """Liste les actifs réels tokenisés"""
    return bridge.get_portfolio()

@app.get("/iot/telemetry")
async def get_telemetry():
    """Récupère les données des capteurs physiques"""
    return iot.get_latest_data()
