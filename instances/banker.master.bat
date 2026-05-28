@echo off
setlocal
set "BANKER_ENV_FILE=%~dp0.env.banker.master.local"
set "BANKER_API_PORT=8100"
set "BANKER_BIND_HOST=0.0.0.0"
set "BANKER_INSTANCE_NAME=FTMO Master 10K"
set "BANKER_FOLLOWER_MODE=false"
call "%~dp0..\banker.bat"
