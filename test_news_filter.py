import asyncio
import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Append src to path to import eva_banker and shared
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'eva-banker')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'shared')))

from eva_banker.services.news_filter import NewsFilterService

async def main():
    service = NewsFilterService(filter_minutes=30)
    print("Fetching economic calendar from ForexFactory...")
    events = await service._fetch_economic_calendar()
    
    print(f"Total structured events parsed: {len(events)}")
    for e in events[:5]:
        print(f"- [{e['impact']}] {e['time']} : {e['currency']} {e['name']}")
        
    print("\nSimulating check calendar (this will trigger telegram warnings if a High event is happening right now):")
    await service._check_calendar()
    print(f"Is filter currently active? {service.is_active}")

if __name__ == "__main__":
    asyncio.run(main())
