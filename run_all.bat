@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM === THE HIVE - LANCEUR GLOBAL ===
REM Version 100% ASCII pure pour eviter les bugs d'interpretation CMD Windows.

echo.
echo ======================================================================
echo           THE HIVE - SYSTEM SWARM GLOBAL LAUNCHER
echo ======================================================================
echo  Ce script va demarrer l'ensemble des modules locaux necessaires
echo  dans des terminaux distincts.
echo ======================================================================
echo.

echo [1/4] Demarrage de la Flotte de Copy Trading (Master + Followers)...
start "THE HIVE - BANKER FLEET" cmd /c "run_fleet.bat"
timeout /t 5 >nul

echo [2/4] Demarrage du synchronisateur d'historiques Fetch and Push...
start "THE HIVE - FETCH DATA DAEMON" cmd /k "daemon.fetch_data.bat"
timeout /t 2 >nul

echo [3/4] Demarrage de l'Agent Chartist Analyste Technique...
start "THE HIVE - HERMES CHARTIST DAEMON" cmd /k "hermes.chartist.bat"
timeout /t 2 >nul

echo [4/4] Demarrage de l'Agent Head Trader Decideur Swarm...
start "THE HIVE - AGENT TRADER DAEMON" cmd /k "agent_trader.bat"

echo.
echo ======================================================================
echo  TOUS LES DAEMONS ONT ETE LANCES AVEC SUCCES !
echo  - 4 fenetres independantes ont ete ouvertes sur votre bureau.
echo  - Vous pouvez les minimiser pour les laisser tourner en continu.
echo ======================================================================
echo.
pause
