#!/usr/bin/env python3
"""
Genesis Check - The Hive Construction Validator
Ce script tente d'importer toutes les classes principales des Agents pour valider
l'intégrité du code Python, les paths et la syntaxe avant le déploiement.
"""

import sys
import os
import importlib

# Ajout du dossier src au PYTHONPATH pour simuler l'environnement Docker
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
shared_path = os.path.join(src_path, "shared") # Fix pour le module shared
sys.path.append(src_path)
sys.path.append(shared_path)

# Mocking des libs tierces manquantes (car on est en environnement restreint)
from unittest.mock import MagicMock
sys.modules["psutil"] = MagicMock()
sys.modules["paho"] = MagicMock()
sys.modules["paho.mqtt"] = MagicMock()
sys.modules["paho.mqtt.client"] = MagicMock()
sys.modules["web3"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()
sys.modules["asyncpg"] = MagicMock()
sys.modules["sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy.orm"] = MagicMock()
sys.modules["sqlalchemy.ext.asyncio"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()
sys.modules["cryptography"] = MagicMock()
sys.modules["cryptography.fernet"] = MagicMock()
sys.modules["jose"] = MagicMock()
sys.modules["passlib"] = MagicMock()
sys.modules["passlib.context"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["qdrant_client"] = MagicMock()
sys.modules["qdrant_client.models"] = MagicMock()
sys.modules["qdrant_client.http"] = MagicMock()
sys.modules["bs4"] = MagicMock()

# Liste des modules à vérifier
# Format: (Nom Module, Chemin d'import, Classe/Objet à vérifier)
MODULES_TO_CHECK = [
    # --- NOYAU CENTRAL ---
    ("Expert A: CORE", "eva-core.eva_core.main", "app"),
    ("Expert E: BUILDER (Architect)", "eva-builder.eva_builder.main", "app"),
    ("Expert F: SENTINEL (Cyber)", "eva-sentinel.eva_sentinel.main", "app"),

    # --- DIVISION FINANCIÈRE ---
    ("Expert B: BANKER", "eva-banker.eva_banker.main", "app"),
    ("Expert J: ADVOCATE (Compliance)", "eva-compliance.eva_compliance.main", "app"),
    ("Expert K: SOVEREIGN (RWA)", "eva-rwa.eva_rwa.main", "app"), # Fix import path if needed, check main.py location

    # --- DIVISION INTELLIGENCE ---
    ("Expert C: SHADOW", "eva-shadow.eva_shadow.main", "app"),
    ("Expert D: WRAITH (Vision)", "eva-wraith.eva_wraith.main", "app"),
    ("Expert G: MUSE (Creative)", "eva-muse.eva_muse.main", "app"),
    ("Expert H: SAGE (Health)", "eva-sage.eva_sage.main", "app"),
    ("Expert I: RESEARCHER (Lab)", "eva-lab.eva_lab.main", "app"),

    # --- INFRASTRUCTURE ---
    ("Infra: SUBSTRATE", "eva-substrate.eva_substrate.main", "app"),
    ("Shared Models", "shared.shared.models", "AgentMessage"),
    ("Shared ROE", "shared.shared.roe", "get_all_roe"),
]

def run_genesis():
    print("GENESIS CHECK: Validating Code Integrity...")
    print("=" * 60)
    
    failures = []
    successes = 0

    for name, import_path, obj_name in MODULES_TO_CHECK:
        print(f"Testing {name:.<40}", end=" ")
        try:
            # Ajustement des paths d'import basés sur la structure de fichiers standard
            folder_name = import_path.split(".")[0] # ex: eva-core
            specific_path = os.path.join(src_path, folder_name)
            if specific_path not in sys.path:
                sys.path.append(specific_path)
                
            # Tentative d'import
            module_name = ".".join(import_path.split(".")[1:]) 
            module = importlib.import_module(module_name)
            
            if obj_name:
                getattr(module, obj_name)
            
            print("OK")
            successes += 1
            
        except ImportError as e:
            print(f"FAILED (Import)")
            failures.append(f"{name}: {str(e)}")
        except AttributeError as e:
            print(f"FAILED (Attr)")
            failures.append(f"{name}: {str(e)}")
        except Exception as e:
            # Catching generic exception to prevent crash
            print(f"FAILED (Error)")
            failures.append(f"{name}: {str(e)}")

    print("=" * 60)
    if not failures:
        print(f"GENESIS COMPLETE. All {successes} modules are syntactically valid.")
        return 0
    else:
        print(f"GENESIS WARNING. {len(failures)} modules failed to load:")
        for f in failures:
            print(f"  - {f}")
        return 1

if __name__ == "__main__":
    sys.exit(run_genesis())
