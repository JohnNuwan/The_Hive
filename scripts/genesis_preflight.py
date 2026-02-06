import asyncio
import sys
import psutil
from rich.console import Console
from rich.table import Table
from shared.redis_client import get_redis_client, init_redis
from shared.mqtt_client import EVAMQTTClient

console = Console()

async def run_preflight():
    """Vérification totale avant décollage."""
    console.print(Panel("🐝 THE HIVE - GENESIS PRE-FLIGHT SEQUENCE", style="bold yellow"))
    
    table = Table(title="Systèmes Systémiques")
    table.add_column("Composant", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Détails", style="dim")

    # 1. Hardware (CPU/RAM)
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    hw_status = "[green]PASS" if cpu < 80 and ram < 90 else "[red]FAIL"
    table.add_row("Ressources Physique", hw_status, f"CPU: {cpu}% | RAM: {ram}%")

    # 2. Redis Neural Link
    try:
        await init_redis()
        redis = get_redis_client()
        ping = await redis._client.ping()
        table.add_row("Lien Redis", "[green]PASS", "PONG received")
    except Exception:
        table.add_row("Lien Redis", "[red]FAIL", "Connection error")

    # 3. MQTT Neural Link (QoS 2)
    mqtt = EVAMQTTClient("preflight")
    try:
        await asyncio.wait_for(mqtt.connect(), timeout=5)
        table.add_row("Lien MQTT", "[green]PASS", "Broker online")
        await mqtt.disconnect()
    except Exception:
        table.add_row("Lien MQTT", "[red]FAIL", "Timeout/Broker offline")

    # 4. Kernel Rust Check
    # (Mock: En prod, on vérifie si le processus eva-kernel tourne)
    kernel_active = any("eva-kernel" in p.name() for p in psutil.process_iter())
    table.add_row("Eva-Kernel (Rust)", "[green]PASS" if kernel_active else "[yellow]MOCKED", "Hardened Laws status")

    console.print(table)
    
    # Conclusion
    if any(row[1] == "[red]FAIL" for row in table.columns[1].cells):
        console.print("\n❌ [bold red]PRE-FLIGHT FAILED. Genesis launch ABORTED.[/bold red]\n")
        sys.exit(1)
    else:
        console.print("\n✨ [bold green]ALL SYSTEMS GO. E.V.A. is ready for Genesis Awakening.[/bold green]\n")

from rich.panel import Panel

if __name__ == "__main__":
    asyncio.run(run_preflight())
