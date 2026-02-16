@echo off
echo 🛡️ Starting The Sentinel (Native Mode for Metrics)...
set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-sentinel
venv\Scripts\python -m uvicorn eva_sentinel.main:app --host 0.0.0.0 --port 8200
pause
