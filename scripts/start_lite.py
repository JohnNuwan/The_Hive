"""
THE HIVE — Lite Mode Launcher (Windows Native)
═══════════════════════════════════════════════

Lance tous les services Python nativement sur Windows.
L'infrastructure (Redis, Qdrant) est lancée via Docker.
MT5 est attendu localement sur Windows.

Usage:
    python scripts/start_lite.py              # Lance tout
    python scripts/start_lite.py --infra-only # Lance seulement l'infrastructure Docker
    python scripts/start_lite.py --no-infra   # Lance seulement les services Python (infra déjà up)
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).parent.parent
COMPOSE_FILE = ROOT_DIR / "Documentation" / "Config" / "docker-compose.lite.yaml"

# Services Python avec leur module uvicorn et port
PYTHON_SERVICES = [
    {"name": "Core",       "module": "eva_core.main:app",       "port": 8000, "dir": "src/eva-core"},
    {"name": "Banker",     "module": "eva_banker.main:app",      "port": 8100, "dir": "src/eva-banker"},
    {"name": "Sentinel",   "module": "eva_sentinel.main:app",    "port": 8200, "dir": "src/eva-sentinel"},
    {"name": "Compliance", "module": "eva_compliance.main:app",  "port": 8300, "dir": "src/eva-compliance"},
    {"name": "Substrate",  "module": "eva_substrate.main:app",   "port": 8400, "dir": "src/eva-substrate"},
    {"name": "Accountant", "module": "eva_accountant.main:app",  "port": 8500, "dir": "src/eva-accountant"},
    {"name": "Lab",        "module": "eva_lab.main:app",         "port": 8600, "dir": "src/eva-lab"},
    {"name": "Shadow",     "module": "eva_shadow.main:app",      "port": 8900, "dir": "src/eva-shadow"},
    {"name": "Builder",    "module": "eva_builder.main:app",     "port": 9000, "dir": "src/eva-builder"},
    {"name": "Muse",       "module": "eva_muse.main:app",        "port": 9100, "dir": "src/eva-muse"},
    {"name": "Sage",       "module": "eva_sage.main:app",        "port": 9200, "dir": "src/eva-sage"},
    {"name": "Researcher", "module": "eva_researcher.main:app",  "port": 9300, "dir": "src/eva-researcher"},
    {"name": "Wraith",     "module": "eva_wraith.main:app",      "port": 9400, "dir": "src/eva-wraith"},
]

# ═══════════════════════════════════════════════════════════════════════════════
# COULEURS TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════

class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def banner():
    print(f"""{C.GREEN}{C.BOLD}
    ╔═══════════════════════════════════════════════════╗
    ║           🐝  T H E   H I V E  🐝               ║
    ║          Lite Mode — Windows Native               ║
    ╚═══════════════════════════════════════════════════╝{C.RESET}
    """)

def log(msg, color=C.GREEN):
    timestamp = time.strftime("%H:%M:%S")
    print(f"  {C.CYAN}[{timestamp}]{C.RESET} {color}{msg}{C.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATIONS PRÉ-LANCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def check_prerequisites():
    """Vérifie que tout est en place"""
    log("Vérification des prérequis...")
    
    errors = []
    
    # Python
    log(f"  Python {sys.version.split()[0]}", C.GREEN)
    
    # Docker
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, timeout=5, encoding="utf-8", errors="replace")
        if result.returncode == 0:
            log(f"  Docker: {result.stdout.strip()}", C.GREEN)
        else:
            errors.append("Docker non disponible")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        errors.append("Docker non installé ou non démarré")
    
    # Ollama
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            log(f"  Ollama: {len(models)} modèles ({', '.join(models[:3])})", C.GREEN)
        else:
            log("  Ollama: démarré mais pas de modèles", C.YELLOW)
    except Exception:
        log("  Ollama: NON DÉTECTÉ — le chat LLM ne fonctionnera pas", C.YELLOW)
    
    # MT5
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            info = mt5.account_info()
            if info:
                log(f"  MT5: Connecté (Compte {info.login}, Serveur {info.server})", C.GREEN)
            else:
                log("  MT5: Initialisé mais pas de compte connecté", C.YELLOW)
            mt5.shutdown()
        else:
            log("  MT5: Installé mais ne peut pas s'initialiser (vérifiez qu'il est ouvert)", C.YELLOW)
    except ImportError:
        log("  MT5: Package MetaTrader5 non installé (pip install MetaTrader5)", C.RED)
    except Exception as e:
        log(f"  MT5: Erreur — {e}", C.YELLOW)
    
    # Packages critiques
    for pkg in ["fastapi", "uvicorn", "redis", "pydantic"]:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"Package '{pkg}' manquant")
    
    if errors:
        for err in errors:
            log(f"  ERREUR: {err}", C.RED)
        return False
    
    log("Prérequis OK", C.GREEN)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE DOCKER
# ═══════════════════════════════════════════════════════════════════════════════

def start_infrastructure():
    """Démarre Redis, Qdrant, TimescaleDB via Docker"""
    log("Démarrage de l'infrastructure Docker...")
    
    # Nettoyage des conteneurs orphelins qui pourraient bloquer
    log("  Nettoyage des conteneurs existants...")
    for container_name in [
        "hive-infra-redis",
        "hive-infra-qdrant",
        "hive-infra-timescale",
        "hive-infra-mosquitto",
    ]:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, timeout=15, encoding="utf-8", errors="replace"
        )
    
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--force-recreate"],
            capture_output=True, timeout=120, encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            log("Infrastructure Docker démarrée (Redis, Qdrant, TimescaleDB, Mosquitto)", C.GREEN)
        else:
            # Parfois stderr contient juste des logs de création, vérifier si les containers tournent
            check = subprocess.run(
                ["docker", "ps", "--filter", "name=hive-infra", "--format", "{{.Names}}"],
                capture_output=True, timeout=10, encoding="utf-8", errors="replace"
            )
            running = [c.strip() for c in check.stdout.strip().split("\n") if c.strip()]
            if len(running) >= 2:
                log(f"Infrastructure partiellement démarrée: {', '.join(running)}", C.YELLOW)
            else:
                log(f"Erreur Docker: {result.stderr[:300]}", C.RED)
                return False
    except subprocess.TimeoutExpired:
        log("Timeout Docker — vérifiez que Docker Desktop est lancé", C.RED)
        return False
    except FileNotFoundError:
        log("Docker introuvable", C.RED)
        return False
    
    # Attendre que Redis soit prêt
    log("Attente de Redis...")
    for i in range(15):
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379, password=os.getenv("REDIS_PASSWORD", "devpassword"), socket_timeout=2)
            r.ping()
            log("Redis prêt", C.GREEN)
            return True
        except Exception:
            time.sleep(1)
    
    log("Redis n'a pas répondu dans les temps", C.RED)
    return False


def stop_infrastructure():
    """Arrête l'infrastructure Docker"""
    log("Arrêt de l'infrastructure Docker...")
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
        capture_output=True, timeout=30, encoding="utf-8", errors="replace"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICES PYTHON
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup_ports():
    """Tue les processus Python qui occupent nos ports (restes d'un lancement précédent)"""
    log("  Nettoyage des ports occupes...")
    all_ports = [svc["port"] for svc in PYTHON_SERVICES]
    
    for port in all_ports:
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, timeout=5,
                encoding="utf-8", errors="replace"
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    proc_id = parts[-1]
                    if proc_id and proc_id != "0":
                        subprocess.run(
                            ["taskkill", "/PID", proc_id, "/F"],
                            capture_output=True, timeout=5,
                            encoding="utf-8", errors="replace"
                        )
                        log(f"    Port {port}: process {proc_id} tue", C.YELLOW)
        except Exception:
            pass
    
    time.sleep(1)


def start_python_services():
    """Lance tous les services Python avec uvicorn"""
    processes = []
    
    # Nettoyer les ports avant de lancer
    cleanup_ports()
    
    # PYTHONPATH pour que les imports shared fonctionnent
    env = os.environ.copy()
    python_paths = [
        str(ROOT_DIR / "src" / "shared"),
        str(ROOT_DIR / "src" / "eva-core"),  # Pour les imports cross-service
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ";".join(python_paths + ([existing] if existing else []))
    # Fix Windows encoding for emoji in logs
    env["PYTHONIOENCODING"] = "utf-8"
    
    log("Démarrage des services Python...")
    
    for svc in PYTHON_SERVICES:
        svc_dir = ROOT_DIR / svc["dir"]
        
        # Vérifier que le répertoire existe
        if not svc_dir.exists():
            log(f"  {svc['name']}: répertoire manquant ({svc['dir']})", C.YELLOW)
            continue
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            svc["module"],
            "--host", "0.0.0.0",
            "--port", str(svc["port"]),
            "--log-level", "info",
        ]
        
        # Ajouter le répertoire du service au PYTHONPATH
        svc_env = env.copy()
        svc_env["PYTHONPATH"] = str(svc_dir) + ";" + svc_env["PYTHONPATH"]
        
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                env=svc_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            processes.append({"name": svc["name"], "port": svc["port"], "proc": proc})
            log(f"  {svc['name']:12s} → http://localhost:{svc['port']} (PID {proc.pid})", C.GREEN)
        except Exception as e:
            log(f"  {svc['name']}: ERREUR au lancement — {e}", C.RED)
    
    return processes


def start_frontend():
    """Lance le frontend Nexus en mode dev"""
    nexus_dir = ROOT_DIR / "src" / "eva-nexus"
    
    # Sur Windows, npm est npm.cmd
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    
    if not (nexus_dir / "node_modules").exists():
        log("Installation des dépendances frontend (npm install)...", C.YELLOW)
        subprocess.run([npm_cmd, "install"], cwd=str(nexus_dir), capture_output=True, timeout=120, shell=True, encoding="utf-8", errors="replace")
    
    log("Démarrage du frontend Nexus (Vite dev server)...")
    
    env = os.environ.copy()
    proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(nexus_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    
    log(f"  Nexus UI    → http://localhost:3001 (PID {proc.pid})", C.GREEN)
    return proc


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_health(processes):
    """Vérifie la santé des services après démarrage"""
    log("\nVérification de santé (5s d'attente)...")
    time.sleep(5)
    
    import httpx
    
    for svc in processes:
        try:
            resp = httpx.get(f"http://localhost:{svc['port']}/health", timeout=3)
            if resp.status_code == 200:
                log(f"  {svc['name']:12s} → ✅ ONLINE", C.GREEN)
            else:
                log(f"  {svc['name']:12s} → ⚠️  HTTP {resp.status_code}", C.YELLOW)
        except Exception:
            log(f"  {svc['name']:12s} → ❌ OFFLINE (peut être en démarrage)", C.RED)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="THE HIVE Lite Mode Launcher")
    parser.add_argument("--infra-only", action="store_true", help="Lance seulement l'infrastructure Docker")
    parser.add_argument("--no-infra", action="store_true", help="Ne lance pas l'infrastructure Docker")
    parser.add_argument("--no-frontend", action="store_true", help="Ne lance pas le frontend")
    parser.add_argument("--services", nargs="+", help="Liste des services à lancer (ex: Core Banker Sentinel)")
    args = parser.parse_args()

    banner()
    
    if not check_prerequisites():
        log("Des prérequis manquent. Corrigez les erreurs ci-dessus.", C.RED)
        sys.exit(1)
    
    # Infrastructure
    if not args.no_infra:
        if not start_infrastructure():
            log("L'infrastructure n'a pas démarré correctement.", C.RED)
            sys.exit(1)
    
    if args.infra_only:
        log("\nInfrastructure lancée. Utilisez --no-infra pour ne lancer que les services.", C.GREEN)
        return
    
    # Services Python
    all_processes = start_python_services()
    
    # Frontend
    frontend_proc = None
    if not args.no_frontend:
        frontend_proc = start_frontend()
    
    # Health check
    check_health(all_processes)
    
    # Résumé
    print(f"""
{C.GREEN}{C.BOLD}═══════════════════════════════════════════════════════
  🐝 THE HIVE est opérationnel en mode Lite!
═══════════════════════════════════════════════════════{C.RESET}

  {C.CYAN}Frontend:{C.RESET}  http://localhost:3001
  {C.CYAN}Core API:{C.RESET}  http://localhost:8000/docs
  {C.CYAN}Banker:{C.RESET}    http://localhost:8100/docs
  {C.CYAN}Grafana:{C.RESET}   http://localhost:3000 (si Docker full)
  
  {C.YELLOW}Ctrl+C pour arrêter tous les services{C.RESET}
""")
    
    # Attendre Ctrl+C
    dead_services = set()
    try:
        while True:
            time.sleep(5)
            # Vérifier si des processus sont morts (rapport unique)
            for svc in all_processes:
                if svc["name"] not in dead_services and svc["proc"].poll() is not None:
                    dead_services.add(svc["name"])
                    log(f"  {svc['name']} s'est arrêté (code {svc['proc'].returncode})", C.RED)
    except KeyboardInterrupt:
        log("\nArrêt de THE HIVE...", C.YELLOW)
        
        # Kill tous les processus
        for svc in all_processes:
            try:
                svc["proc"].terminate()
            except Exception:
                pass
        
        if frontend_proc:
            try:
                frontend_proc.terminate()
            except Exception:
                pass
        
        if not args.no_infra:
            stop_infrastructure()
        
        log("THE HIVE arrêté proprement.", C.GREEN)


if __name__ == "__main__":
    main()
