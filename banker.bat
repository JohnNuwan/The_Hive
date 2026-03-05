@echo off
setlocal
title THE HIVE - EXPERT BANKER (MT5)

:start
echo 🏦 [%DATE% %TIME%] Starting The Banker (Native Mode for MT5)...

:: Check for MetaTrader5 dependency
if not exist "venv\Scripts\python.exe" (
    echo ❌ ERROR: Virtual environment 'venv' not found!
    echo Please run 'python -m venv venv' first.
    pause
    exit /b 1
)

venv\Scripts\python -c "import MetaTrader5" 2>nul
if %errorlevel% neq 0 (
    echo 📦 Installing missing MetaTrader5 dependency...
    venv\Scripts\python -m pip install MetaTrader5
)

:: Setup Environment
set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-banker
set MOCK_MT5=false

:: Run Service
venv\Scripts\python -m uvicorn eva_banker.main:app --host 0.0.0.0 --port 8100 --env-file .env

echo.
echo ⚠️ [%DATE% %TIME%] The Banker process has exited.
echo 🔄 Restarting in 5 seconds (Press Ctrl+C to abort)...
timeout /t 5 >nul
goto start
