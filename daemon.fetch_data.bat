@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

REM === THE HIVE - DAEMON FETCH DATA ===
REM Lance la recuperation des donnees historiques MT5 en boucle
REM pour alimenter le serveur d entrainement.

REM Parametres optionnels
if "%~1"=="" (
    set INTERVAL_SECONDS=3600
) else (
    set INTERVAL_SECONDS=%~1
)

REM Verification venv
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
echo     THE HIVE - DAEMON FETCH DATA
echo     Actualisation de l'historique MT5
echo     Intervalle : %INTERVAL_SECONDS% secondes
echo ======================================================
echo.

:start
echo [%DATE% %TIME%] Lancement de l'extraction des historiques MT5...
%PYTHON_EXE% -X utf8 scripts\fetch_history.py

echo [%DATE% %TIME%] Envoi des historiques vers le serveur GPU...
%PYTHON_EXE% -X utf8 scripts\push_history.py

echo.
echo [%DATE% %TIME%] Extraction terminee. Prochaine extraction dans %INTERVAL_SECONDS% secondes...
timeout /t %INTERVAL_SECONDS% >nul
goto start
