#!/bin/bash
# ollama serve &
# sleep 5
echo "--- PIP LIST ---"
python3 -m pip list
echo "--- STARTING UVICORN ---"
export PYTHONPATH=$PYTHONPATH:/app/eva-core
python3 -m uvicorn eva_core.main:app --host 0.0.0.0 --port 8080
