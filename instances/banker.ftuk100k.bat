@echo off
setlocal
set "BANKER_ENV_FILE=%~dp0.env.banker.ftuk100k.local"
set "BANKER_API_PORT=8120"
set "BANKER_BIND_HOST=127.0.0.1"
set "BANKER_INSTANCE_NAME=FTUK Funded 100K 333382300"
set "BANKER_FOLLOWER_MODE=true"
call "%~dp0..\banker.bat"
