@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

:: === HERMES CHALLENGE FACTORY - THE HIVE ===
:: Agent autonome de production de champions prop firm (FTMO/FTUK)
:: Tourne en boucle toutes les 2 heures

:: Parametres optionnels
if "%~1"=="dry" (
    set DRY_RUN=--dry-run
    set FIRM=ftmo
    set BALANCE=10000
) else (
    set DRY_RUN=
    if "%~1"=="" (
        set FIRM=ftmo
    ) else (
        set FIRM=%~1
    )
    if "%~2"=="" (
        set BALANCE=10000
    ) else (
        set BALANCE=%~2
    )
)

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

set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-banker

echo ======================================================
echo   HERMES CHALLENGE FACTORY - Prop Firm Champion Agent
echo   Firm : %FIRM% ^| Balance : %BALANCE% EUR
if "%DRY_RUN%"=="--dry-run" echo   MODE : DRY-RUN (aucun envoi Discord)
echo ======================================================
echo.

%PYTHON_EXE% -X utf8 scripts\hermes_challenge_factory.py --firm %FIRM% --balance %BALANCE% --loop --interval-minutes 120 %DRY_RUN%

echo.
echo [%DATE% %TIME%] Agent arrete.
pause
