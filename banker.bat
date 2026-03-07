@echo off
setlocal
title THE HIVE - EXPERT BANKER (MT5)

:start
echo [%DATE% %TIME%] Demarrage de The Banker (mode natif MT5)...

:: Verification de la dependance MetaTrader5
if not exist "venv\Scripts\python.exe" (
    echo ERROR: environnement virtuel 'venv' introuvable.
    echo Lance d'abord: python -m venv venv
    pause
    exit /b 1
)

venv\Scripts\python -c "import MetaTrader5" 2>nul
if %errorlevel% neq 0 (
    echo Installation de la dependance MetaTrader5...
    venv\Scripts\python -m pip install MetaTrader5
)

:: Configuration locale du Banker
if "%HIVE_SERVER_HOST%"=="" set HIVE_SERVER_HOST=192.168.1.6
if "%REDIS_HOST%"=="" set REDIS_HOST=%HIVE_SERVER_HOST%
if "%REDIS_PORT%"=="" set REDIS_PORT=6379

set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-banker
set MOCK_MT5=false
set PAPER_TRADING=false

echo Serveur THE HIVE cible: %HIVE_SERVER_HOST%
echo Redis cible: %REDIS_HOST%:%REDIS_PORT%
echo API Banker exposee sur ce PC: 0.0.0.0:8100

:: Ouverture firewall Windows pour accessibilite reseau du banker (si possible)
netsh advfirewall firewall show rule name="THE_HIVE_BANKER_8100" >nul 2>nul
if %errorlevel% neq 0 (
    netsh advfirewall firewall add rule name="THE_HIVE_BANKER_8100" dir=in action=allow protocol=TCP localport=8100 >nul 2>nul
)

:: Lancement du service
venv\Scripts\python -m uvicorn eva_banker.main:app --host 0.0.0.0 --port 8100 --env-file .env

echo.
echo [%DATE% %TIME%] Processus Banker arrete.
echo Redemarrage dans 5 secondes (Ctrl+C pour annuler)...
timeout /t 5 >nul
goto start