import httpx
import asyncio

SERVICES = {
    "core": "http://localhost:8000/health",
    "banker": "http://localhost:8100/health",
    "sentinel": "http://localhost:8200/health",
    "compliance": "http://localhost:8300/health",
    "substrate": "http://localhost:8400/health",
    "accountant": "http://localhost:8500/health",
    "lab": "http://localhost:8600/health",
    "rwa": "http://localhost:8700/health",
    "nexus": "http://localhost:3030"
}

async def verify():
    print("🔍 Starting Global Hive Health Check...")
    print("-" * 50)
    async with httpx.AsyncClient() as client:
        for name, url in SERVICES.items():
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    print(f"✅ {name:<12} : ONLINE")
                else:
                    print(f"⚠️ {name:<12} : WARN ({resp.status_code})")
            except Exception as e:
                print(f"❌ {name:<12} : OFFLINE ({type(e).__name__})")
    print("-" * 50)
    print("Health Check Complete.")

if __name__ == "__main__":
    asyncio.run(verify())
