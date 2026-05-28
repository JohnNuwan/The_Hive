@echo off
setlocal
start "THE HIVE - FTUK Challenge 100K" "%~dp0banker.ftuk100k.bat"
timeout /t 3 >nul
start "THE HIVE - FTMO Challenge 50K" "%~dp0banker.ftmo50k.bat"
timeout /t 3 >nul
start "THE HIVE - FTMO Master 10K" "%~dp0banker.master.bat"
