
import asyncio
from unittest.mock import MagicMock, AsyncMock

async def test():
    m = MagicMock()
    m.start_monitoring = AsyncMock()

    print(f"Calling start_monitoring: {m.start_monitoring()}")
    print(f"Is awaitable? {asyncio.iscoroutine(m.start_monitoring())}")

    await m.start_monitoring()
    print("Awaited successfully")

if __name__ == "__main__":
    asyncio.run(test())
