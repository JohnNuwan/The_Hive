@echo off
setlocal
set "BANKER_ENV_FILE=%~dp0.env.banker.ftmo541283689.local"
set "BANKER_API_PORT=8190"
set "BANKER_BIND_HOST=127.0.0.1"
set "BANKER_INSTANCE_NAME=FTMO Challenge 541283689"
set "BANKER_FOLLOWER_MODE=true"
call "%~dp0..\banker.bat"
