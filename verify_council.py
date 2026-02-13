import httpx
import asyncio
import json
from uuid import uuid4

async def test_council():
    base_url = "http://localhost:8000"
    session_id = str(uuid4())

    async with httpx.AsyncClient(timeout=600.0) as client:
        # 1. Test General Chat (llama3.2:1b)
        print("--- Test 1: General Chat ---")
        response = await client.post(
            f"{base_url}/chat",
            json={"message": "Bonjour EVA, comment vas-tu ?", "session_id": session_id}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json().get('message')}")
        print(f"Expert: {response.json().get('metadata', {}).get('expert')}")

        # 2. Test Banker Expert (qwen2.5-coder:3b)
        print("\n--- Test 2: Banker Expert ---")
        response = await client.post(
            f"{base_url}/chat",
            json={"message": "Peux-tu ouvrir un trade de 0.1 lot BUY sur XAUUSD ?", "session_id": session_id}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json().get('message')}")
        print(f"Expert: {response.json().get('metadata', {}).get('expert')}")

if __name__ == "__main__":
    asyncio.run(test_council())
