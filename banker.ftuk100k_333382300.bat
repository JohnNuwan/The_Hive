@echo off
setlocal
set "BANKER_ENV_FILE=%~dp0.env.banker.ftuk100k.local"
set "BANKER_API_PORT=8120"
set "BANKER_BIND_HOST=127.0.0.1"
set "BANKER_INSTANCE_NAME=FTUK Funded 100K 333382300"
set "BANKER_FOLLOWER_MODE=true"
set "MT5_LOGIN=333382300"
set "MT5_PASSWORD=o1!GhuImff"
set "MT5_SERVER=FTUKMarkets-Trade"
set "MT5_TERMINAL_PATH=C:/Users/nandi/Desktop/FTMO_CHALLENGE/John_100K_API_333382300/terminal64.exe"
set "MT5_TERMINAL_PORTABLE=false"
set "MT5_TRY_ALTERNATE_PORTABLE_MODE=false"
call "%~dp0banker.bat"
