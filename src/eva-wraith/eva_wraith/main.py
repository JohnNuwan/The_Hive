"""
The Wraith - Agent Vision
Expert D: Vision et Skeleton Tracking
"""

from fastapi import FastAPI

app = FastAPI(title="The Wraith API")

@app.get("/health")
async def health():
    return {"status": "skeleton", "service": "wraith", "note": "Waiting for Coral TPU"}
