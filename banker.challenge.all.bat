@echo off
setlocal
start "THE HIVE - FTUK 333382206" "%~dp0banker.ftuk100k_333382206.bat"
timeout /t 3 >nul
start "THE HIVE - FTUK 333382300" "%~dp0banker.ftuk100k_333382300.bat"
timeout /t 3 >nul
start "THE HIVE - FTMO Challenge 50K" "%~dp0banker.ftmo50k.bat"
timeout /t 3 >nul
start "THE HIVE - FTMO Master 10K" "%~dp0banker.master.bat"
