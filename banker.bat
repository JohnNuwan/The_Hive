@echo off
echo 🏦 Starting The Banker (Native Mode for MT5)...

:: Check for MetaTrader5 dependency
venv\Scripts\python -c "import MetaTrader5" 2>nul
if %errorlevel% neq 0 (
    echo 📦 Installing missing MetaTrader5 dependency...
    venv\Scripts\python -m pip install MetaTrader5
)

set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-banker
set MOCK_MT5=false
venv\Scripts\python -m uvicorn eva_banker.main:app --host 0.0.0.0 --port 8100
pause
