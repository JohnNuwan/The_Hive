@echo off
setlocal
set "BANKER_ENV_FILE=%~dp0.env.banker.ftmo50k.local"
set "BANKER_API_PORT=8110"
set "BANKER_BIND_HOST=127.0.0.1"
set "BANKER_INSTANCE_NAME=FTMO Challenge 50K"
set "BANKER_FOLLOWER_MODE=true"
call "%~dp0banker.bat"
