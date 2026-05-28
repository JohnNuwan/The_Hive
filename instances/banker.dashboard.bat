@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8
title THE HIVE - DASHBOARD BANKER

if not exist "%~dp0..\venv\Scripts\python.exe" (
    echo ERROR: environnement virtuel 'venv' introuvable.
    exit /b 1
)

set PYTHONPATH=%~dp0..\src\shared;%~dp0..\src\eva-banker
"%~dp0..\venv\Scripts\python.exe" -X utf8 -m eva_banker.dashboard %*
