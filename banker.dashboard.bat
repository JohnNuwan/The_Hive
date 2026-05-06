@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8
title THE HIVE - DASHBOARD BANKER

if not exist "venv\Scripts\python.exe" (
    echo ERROR: environnement virtuel 'venv' introuvable.
    exit /b 1
)

set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-banker
venv\Scripts\python -X utf8 -m eva_banker.dashboard %*
