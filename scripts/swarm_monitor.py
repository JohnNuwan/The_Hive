import asyncio
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from redis.asyncio import Redis

console = Console()

async def get_drones(redis: Redis):
    keys = await redis.keys("swarm:drone:*")
    drones = []
    for key in keys:
        data = await redis.get(key)
        if data:
            drones.append(json.loads(data))
    return drones

def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="experts", ratio=1),
        Layout(name="drones", ratio=2),
        Layout(name="events", ratio=2)
    )
    return layout

async def run_monitor():
    redis = Redis.from_url("redis://127.0.0.1:6379", decode_responses=True)
    layout = make_layout()
    events = []

    async def log_events():
        pubsub = redis.pubsub()
        await pubsub.subscribe("eva.swarm.events")
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                events.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg['data']}")
                if len(events) > 10: events.pop(0)

    asyncio.create_task(log_events())

    with Live(layout, refresh_per_second=2):
        while True:
            # Header
            layout["header"].update(Panel("🐝 THE HIVE SWARM MONITOR - Production Readiness", style="bold yellow"))
            
            # Experts (Mock for now)
            expert_table = Table(title="Conseil des Experts")
            expert_table.add_column("Expert")
            expert_table.add_column("Status")
            expert_table.add_row("CORE", "[green]Online")
            expert_table.add_row("BANKER", "[green]Online")
            expert_table.add_row("SENTINEL", "[blue]Hardened")
            layout["experts"].update(Panel(expert_table))

            # Drones
            drones = await get_drones(redis)
            drone_table = Table(title="Swarm Drones actifs")
            drone_table.add_column("Nom")
            drone_table.add_column("Mission")
            drone_table.add_column("Sincerity", justify="center")
            drone_table.add_column("Status")
            
            for drone in drones:
                # Simulation de score de sincérité (Linear Probes)
                sincerity_score = 100 - (int(drone.get("id", "0")[0], 16) * 5) % 30 # Mock
                color = "green" if sincerity_score > 85 else "yellow" if sincerity_score > 70 else "red"
                
                drone_table.add_row(
                    drone["name"], 
                    drone["mission"], 
                    f"[{color}]{sincerity_score}%",
                    f"[bold green]{drone['status']}" if drone["status"] == "active" else drone["status"]
                )
            layout["drones"].update(Panel(drone_table))

            # Events
            layout["events"].update(Panel("\n".join(events), title="Flux d'événements Live (News & Sentiment)"))
            
            # Footer
            layout["footer"].update(Panel(f"Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim"))
            
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(run_monitor())
