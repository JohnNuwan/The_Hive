@echo off
title THE HIVE - ALL HORIZONS LAUNCHER
echo =======================================================
echo          THE HIVE - LANCEUR TRIPLE HORIZON (MASTER)
echo =======================================================
echo Ce script va lancer les 3 instances du Banker en parallele :
echo  1. SCALP    (Port 8100, Lots: 0.05 max)
echo  2. INTRADAY (Port 8140, Lots: 0.02 max)
echo  3. SWING    (Port 8150, Lots: 0.01 max)
echo.
echo IMPORTANT : Assurez-vous que MetaTrader 5 est bien ouvert.
echo =======================================================
pause

echo Lancement du Banker SCALP (M5)...
start "THE HIVE - SCALP MASTER 10K" cmd /k "call "%~dp0banker.master.bat""

echo Lancement du Banker INTRADAY (H1)...
start "THE HIVE - INTRADAY MASTER 10K" cmd /k "call "%~dp0banker.intraday.bat""

echo Lancement du Banker SWING (D1)...
start "THE HIVE - SWING MASTER 10K" cmd /k "call "%~dp0banker.swing.bat""

echo.
echo Les 3 instances ont ete lancees dans des terminaux dedies !
echo Vous pouvez fermer ce lanceur principal.
timeout /t 5
