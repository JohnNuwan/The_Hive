import asyncio
import sys
import json
from rich.console import Console
from rich.panel import Panel
from shared.redis_client import get_redis_client, init_redis

console = Console()

async def emergency_halt():
    """Déclenche l'arrêt d'urgence via le Kernel."""
    redis = get_redis_client()
    await redis.publish("eva.banker.requests", {"action": "KILL_SWITCH", "reason": "Manual command via Genesis CLI"})
    console.print(Panel("🚨 [bold red]EMERGENCY HALT SIGNAL SENT TO KERNEL[/bold red]", border_style="red"))

async def swarm_status():
    """Affiche le statut rapide de la ruche."""
    redis = get_redis_client()
    keys = await redis._client.keys("swarm:drone:*")
    console.print(f"🐝 [bold yellow]THE HIVE STATUS:[/bold yellow] {len(keys)} Drones actifs.")

async def force_audit():
    """Demande à l'Expert Researcher de générer un rapport immédiat."""
    redis = get_redis_client()
    await redis.publish("eva.swarm.command", {"action": "GENERATE_AUDIT", "target": "researcher"})
    console.print("[bold green]✅ Demande d'audit envoyée à l'Expert Researcher.[/bold green]")

async def main():
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python genesis_cli.py [halt|status|audit][/yellow]")
        return

    await init_redis()
    cmd = sys.argv[1].lower()

    if cmd == "halt":
        await emergency_halt()
    elif cmd == "status":
        await swarm_status()
    elif cmd == "audit":
        await force_audit()
    else:
        console.print(f"[red]Commande inconnue: {cmd}[/red]")

if __name__ == "__main__":
    asyncio.run(main())
