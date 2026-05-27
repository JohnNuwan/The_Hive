@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

:: === HERMES CHARTIST DAEMON — THE HIVE ===
:: Lance le briefing technique chartiste sur Discord (#analyse-technique)
:: Symboles par défaut : XAUUSD, EURUSD, US100.cash
:: Modifiable via arguments : hermes.chartist.bat XAUUSD,BTCUSD H4 200

:: Paramètres optionnels (via arguments ou valeurs par défaut)
if "%~1"=="" (
    set SYMBOLS=XAUUSD,EURUSD,US100.cash
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

:: Vérification venv
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Le venv 'venv' est introuvable. Lance d'abord: python -m venv venv
    pause
    exit /b 1
)

:: Chargement du PYTHONPATH
set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-banker

echo.
echo ======================================================
echo     HERMES CHARTIST — Briefing Technique Proactif
echo     Symboles : %SYMBOLS%
echo     Timeframe : %TIMEFRAME% — Historique : %BARS% bougies
echo ======================================================
echo.

:start
echo [%DATE% %TIME%] Lancement du Daemon Chartist [TIMEFRAME = H4]...
venv\Scripts\python -X utf8 scripts\hermes_chartist_daemon.py --symbols %SYMBOLS% --timeframe H4 --bars %BARS%

echo [%DATE% %TIME%] Lancement du Daemon Chartist [TIMEFRAME = H1]...
venv\Scripts\python -X utf8 scripts\hermes_chartist_daemon.py --symbols %SYMBOLS% --timeframe H1 --bars %BARS%

echo [%DATE% %TIME%] Lancement du Daemon Chartist [TIMEFRAME = M15]...
venv\Scripts\python -X utf8 scripts\hermes_chartist_daemon.py --symbols %SYMBOLS% --timeframe M15 --bars 200

echo.
echo [%DATE% %TIME%] Cycle multi-timeframe terminé. Prochain envoi dans 4 heures...
timeout /t 14400 >nul
goto start
