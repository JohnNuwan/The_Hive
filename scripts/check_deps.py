import sys
import os

print(f"Python: {sys.version}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
print(f"sys.path: {sys.path}")

def check_import(module_name):
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"{module_name}: OK ({version})")
    except Exception as e:
        print(f"{module_name}: FAILED ({e})")
        # import traceback
        # traceback.print_exc()

check_import("fastapi")
check_import("uvicorn")
check_import("neo4j")
check_import("langchain_ollama")
check_import("mem0")

# Testing Application Import
sys.path.append(os.path.join(os.getcwd(), "src", "eva-core"))
sys.path.append(os.path.join(os.getcwd(), "src", "shared"))

try:
    from eva_core import main
    print("EVA Core Main: Import OK")
except Exception as e:
    print(f"EVA Core Main Import Failed: {e}")
    import traceback
    traceback.print_exc()
