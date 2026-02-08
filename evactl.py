#!/usr/bin/env python3
"""
EvaCTL - Command Line Interface for THE HIVE
Outil d'administration pour piloter E.V.A. sans interface graphique.

Usage:
    python evactl.py status
    python evactl.py panic --reason="Market Crash"
    python evactl.py logs --service=core
"""

import argparse
import sys
import os
import json
import time
from datetime import datetime

# Configuration simulée pour mode Offline
SERVICES = {
    "core": "http://localhost:8000",
    "banker": "http://localhost:8100",
    "sentinel": "http://localhost:8200",
    "compliance": "http://localhost:8300",
    "substrate": "http://localhost:8400",
    "accountant": "http://localhost:8500",
    "lab": "http://localhost:8600",
    "rwa": "http://localhost:8700"
}

LOCKDOWN_FILE = "lockdown.mode"

def check_status(args):
    """Vérifie l'état des services (Ping HTTP ou vérification Process)"""
    print(f"[EvaCTL] Checking Hive Status at {datetime.now().isoformat()}...")
    print("-" * 60)
    print(f"{'SERVICE':<15} | {'URL':<25} | {'STATUS':<10}")
    print("-" * 60)
    
    import httpx
    
    all_up = True
    for name, url in SERVICES.items():
        try:
            # On tente de joindre le point d'entrée health ou l'index
            response = httpx.get(f"{url}/health", timeout=1.0)
            if response.status_code == 200:
                status = "UP ✅"
            else:
                status = f"WARN ({response.status_code})"
                all_up = False
        except (httpx.ConnectError, httpx.TimeoutException):
            status = "DOWN ❌"
            all_up = False
        
        print(f"{name:<15} | {url:<25} | {status:<10}")
    
    print("-" * 60)
    if os.path.exists(LOCKDOWN_FILE):
        print("!!! SYSTEM IS IN LOCKDOWN MODE (PANIC ACTIVE) !!!")
    else:
        print("OK - System is operating normally (No Lockdown).")

def trigger_panic(args):
    """Active le Kill-Switch d'urgence"""
    print(f"[EvaCTL] INITIATING PANIC PROTOCOL !")
    print(f"Reason: {args.reason}")
    
    confirm = input("Are you sure you want to SHUT DOWN everything? (yes/NO): ")
    if confirm.lower() == "yes":
        with open(LOCKDOWN_FILE, "w") as f:
            data = {
                "timestamp": datetime.now().isoformat(),
                "reason": args.reason,
                "triggered_by": "admin_cli"
            }
            json.dump(data, f, indent=2)
        print("STOP - LOCKDOWN FILE CREATED. KERNEL SHOULD REACT IMMEDIATELY.")
    else:
        print("Protocol aborted.")

def show_logs(args):
    """Affiche les logs (Simulé pour l'instant)"""
    print(f"Showing logs for service: {args.service}")
    log_path = f"logs/{args.service}.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            print(f.read())
    else:
        print(f"No log file found at {log_path}. (System might be offline)")

def main():
    parser = argparse.ArgumentParser(description="EvaCTL - The Hive Administration Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: status
    parser_status = subparsers.add_parser("status", help="Check system health")
    
    # Command: panic
    parser_panic = subparsers.add_parser("panic", help="Trigger Emergency Kill-Switch")
    parser_panic.add_argument("--reason", type=str, default="Manual Override", help="Reason for panic")

    # Command: logs
    parser_logs = subparsers.add_parser("logs", help="View service logs")
    parser_logs.add_argument("--service", type=str, default="core", help="Service name (core, banker...)")

    args = parser.parse_args()

    if args.command == "status":
        check_status(args)
    elif args.command == "panic":
        trigger_panic(args)
    elif args.command == "logs":
        show_logs(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
