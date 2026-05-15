@echo off
setlocal
set "BANKER_ENV_FILE=%~dp0.env.banker.ftuk100k_333382206.local"
set "BANKER_API_PORT=8130"
set "BANKER_BIND_HOST=127.0.0.1"
set "BANKER_INSTANCE_NAME=FTUK Challenge 100K 333382206"
set "BANKER_FOLLOWER_MODE=true"
call "%~dp0banker.bat"
