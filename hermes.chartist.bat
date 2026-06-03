@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

REM === HERMES CHARTIST DAEMON - THE HIVE ===
REM Lance le briefing technique chartiste sur Discord et dans la Memoire
REM Symboles par defaut : XAUUSD, EURUSD, US100.cash
REM Modifiable via arguments : hermes.chartist.bat XAUUSD,BTCUSD H4 200

REM Parametres optionnels
if "%~1"=="" (
    set SYMBOLS=XAUUSD,EURUSD,US100.cash,BTCUSD
) else (
    set SYMBOLS=%~1
)
if "%~2"=="" (
    set TIMEFRAME=H4
) else (
    set TIMEFRAME=%~2
)
if "%~3"=="" (
    set BARS=200
) else (
    set BARS=%~3
)

REM Verification venv (.venv ou venv)
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

REM Chargement du PYTHONPATH
set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-banker;%CD%\src\eva-lab

echo.
echo ======================================================
echo     HERMES CHARTIST - Briefing Technique Proactif
echo     Symboles : %SYMBOLS%
echo     Timeframe : %TIMEFRAME% - Historique : %BARS% bougies
echo ======================================================
echo.

:start
echo [%DATE% %TIME%] Lancement du Daemon Chartist [TIMEFRAME = H4]...
%PYTHON_EXE% -X utf8 scripts\hermes_chartist_daemon.py --symbols %SYMBOLS% --timeframe H4 --bars %BARS%

echo [%DATE% %TIME%] Lancement du Daemon Chartist [TIMEFRAME = H1]...
%PYTHON_EXE% -X utf8 scripts\hermes_chartist_daemon.py --symbols %SYMBOLS% --timeframe H1 --bars %BARS%

echo [%DATE% %TIME%] Lancement du Daemon Chartist [TIMEFRAME = M15]...
%PYTHON_EXE% -X utf8 scripts\hermes_chartist_daemon.py --symbols %SYMBOLS% --timeframe M15 --bars 200

echo.
echo [%DATE% %TIME%] Cycle multi-timeframe termine. Prochain envoi dans 4 heures...
timeout /t 14400 >nul
goto start
