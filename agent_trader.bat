@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

REM === THE HIVE - LLM SWARM HEAD TRADER ===
REM Lance l'Agent Trader qui lit les analyses et prend des decisions.

if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
) else (
    if exist "venv\Scripts\python.exe" (
        set PYTHON_EXE=venv\Scripts\python.exe
    ) else (
        echo ERROR: Le dossier virtuel ^(venv ou .venv^) est introuvable.
        pause
        exit /b 1
    )
)

set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-banker;%CD%\src\eva-lab

echo ======================================================
echo     THE HIVE - HEAD TRADER AGENT (LLM SWARM)
echo     Mode d'execution en cours...
echo ======================================================
echo.

%PYTHON_EXE% -X utf8 scripts\agent_trader_daemon.py

pause
