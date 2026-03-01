"""
THE HIVE — Sanity Test Suite (P3)
──────────────────────────────────
Vérifie que tous les services sont accessibles et répondent correctement.
Usage:  python tests/test_sanity.py [HOST]
"""

import asyncio
import sys
import httpx
import json
from datetime import datetime

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.5"

SERVICES = {
    "core":        {"port": 8080, "health": "/health"},
    "kernel":      {"port": 8800, "health": "/health"},
    "sentinel":    {"port": 8200, "health": "/health"},
    "compliance":  {"port": 8300, "health": "/health"},
    "substrate":   {"port": 8400, "health": "/health"},
    "accountant":  {"port": 8500, "health": "/health"},
    "lab":         {"port": 8600, "health": "/health"},
    "rwa":         {"port": 8700, "health": "/health"},
    "quant-lab":   {"port": 8701, "health": "/health"},
    "shadow":      {"port": 8900, "health": "/health"},
    "builder":     {"port": 9000, "health": "/health"},
    "muse":        {"port": 9100, "health": "/health"},
    "sage":        {"port": 9200, "health": "/health"},
    "researcher":  {"port": 9300, "health": "/health"},
    "wraith":      {"port": 9400, "health": "/health"},
    "nervous":     {"port": 9090, "health": "/health"},
    "nexus":       {"port": 3030, "health": "/"},
    "grafana":     {"port": 3000, "health": "/api/health"},
}

INFRA = {
    "redis":       {"port": 6379},
    "qdrant":      {"port": 6333},
    "neo4j":       {"port": 7474},
    "timescaledb": {"port": 5432},
}


async def check_service(name: str, port: int, path: str) -> dict:
    """Check a single HTTP service."""
    url = f"http://{HOST}:{port}{path}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            return {
                "name": name,
                "port": port,
                "status": "✅ OK" if resp.status_code < 400 else f"⚠️ {resp.status_code}",
                "code": resp.status_code,
                "latency_ms": round(resp.elapsed.total_seconds() * 1000),
            }
    except httpx.ConnectError:
        return {"name": name, "port": port, "status": "❌ UNREACHABLE", "code": 0, "latency_ms": -1}
    except httpx.ReadTimeout:
        return {"name": name, "port": port, "status": "⏱️ TIMEOUT", "code": 0, "latency_ms": -1}
    except Exception as e:
        return {"name": name, "port": port, "status": f"❌ {e}", "code": 0, "latency_ms": -1}


async def check_intelligence(host: str) -> dict:
    """Check intelligence endpoints on Core."""
    url = f"http://{host}:8080/intelligence/status"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


async def main():
    print(f"\n{'='*60}")
    print(f"  🐝 THE HIVE — Sanity Test Suite")
    print(f"  Host: {HOST}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Test all services
    print("📡 SERVICE HEALTH CHECKS")
    print(f"{'─'*60}")

    tasks = [
        check_service(name, svc["port"], svc["health"])
        for name, svc in SERVICES.items()
    ]
    results = await asyncio.gather(*tasks)

    ok_count = 0
    fail_count = 0
    for r in results:
        icon = r["status"]
        latency = f"{r['latency_ms']}ms" if r["latency_ms"] >= 0 else "N/A"
        print(f"  {r['name']:15s} :{r['port']:5d}  {icon:20s} {latency:>6s}")
        if "OK" in r["status"]:
            ok_count += 1
        else:
            fail_count += 1

    print(f"\n{'─'*60}")
    print(f"  Total: {ok_count} ✅  {fail_count} ❌  / {len(results)} services")

    # Intelligence status
    print(f"\n🧠 INTELLIGENCE STATUS")
    print(f"{'─'*60}")
    intel = await check_intelligence(HOST)
    print(f"  {json.dumps(intel, indent=2)}")

    # Summary
    print(f"\n{'='*60}")
    if fail_count == 0:
        print("  🟢 ALL SERVICES OPERATIONAL — E.V.A. IS FULLY ONLINE")
    elif fail_count <= 3:
        print(f"  🟡 DEGRADED — {fail_count} service(s) offline")
    else:
        print(f"  🔴 CRITICAL — {fail_count} services down!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
