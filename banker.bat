@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8
if "%BANKER_RICH_LOGS%"=="" set BANKER_RICH_LOGS=true
if "%BANKER_ENV_FILE%"=="" set "BANKER_ENV_FILE=.env"
if "%BANKER_API_PORT%"=="" set BANKER_API_PORT=8100
if "%BANKER_BIND_HOST%"=="" set BANKER_BIND_HOST=0.0.0.0
if "%BANKER_ENABLE_TUNNEL%"=="" set BANKER_ENABLE_TUNNEL=true
if "%BANKER_INSTANCE_NAME%"=="" (
    title THE HIVE - EXPERT BANKER (MT5)
) else (
    title THE HIVE - %BANKER_INSTANCE_NAME%
)

:start
echo [%DATE% %TIME%] Demarrage de The Banker (mode natif MT5)...

:: Verification de la dependance MetaTrader5
if not exist "venv\Scripts\python.exe" (
    echo ERROR: environnement virtuel 'venv' introuvable.
    echo Lance d'abord: python -m venv venv
    pause
    exit /b 1
)

venv\Scripts\python -X utf8 -c "import MetaTrader5" 2>nul
if %errorlevel% neq 0 (
    echo Installation de la dependance MetaTrader5...
    venv\Scripts\python -X utf8 -m pip install MetaTrader5
)

:: Configuration locale du Banker
if "%HIVE_SERVER_HOST%"=="" set HIVE_SERVER_HOST=192.168.1.6
if "%HIVE_SSH_USER%"=="" set HIVE_SSH_USER=aza
if "%REDIS_HOST%"=="" set REDIS_HOST=%HIVE_SERVER_HOST%
if "%REDIS_PORT%"=="" set REDIS_PORT=6379
if "%MUZERO_LIVE_SELECTION_POLICY%"=="" set MUZERO_LIVE_SELECTION_POLICY=champion_only
if "%BANKER_REQUIRE_VALID_CHAMPION%"=="" set BANKER_REQUIRE_VALID_CHAMPION=true
if "%BANKER_ENSEMBLE_ENABLED%"=="" set BANKER_ENSEMBLE_ENABLED=true
if "%BANKER_ENSEMBLE_MIN_EDGE%"=="" set BANKER_ENSEMBLE_MIN_EDGE=0.15
if "%BANKER_FORCE_MAINTENANCE%"=="" set BANKER_FORCE_MAINTENANCE=false
if "%HIVE_TUNNEL_REMOTE_PORT%"=="" set HIVE_TUNNEL_REMOTE_PORT=18100
if "%HIVE_TUNNEL_RELAY_PORT%"=="" set HIVE_TUNNEL_RELAY_PORT=18101
set "BANKER_TUNNEL_KEY=%USERPROFILE%\.ssh\the_hive_banker_tunnel"
set "BANKER_SSH_BIN=%SystemRoot%\System32\OpenSSH\ssh.exe"

set PYTHONPATH=%CD%\src\shared;%CD%\src\eva-banker
set MOCK_MT5=false
set PAPER_TRADING=false

echo Serveur THE HIVE cible: %HIVE_SERVER_HOST%
echo Redis cible: %REDIS_HOST%:%REDIS_PORT%
echo API Banker exposee sur ce PC: %BANKER_BIND_HOST%:%BANKER_API_PORT%
echo Fichier de configuration: %BANKER_ENV_FILE%
echo Tunnel SSH actif: %BANKER_ENABLE_TUNNEL%
echo Ensemble MuZero/Dreamer: %BANKER_ENSEMBLE_ENABLED%

:: Ouverture firewall Windows pour accessibilite reseau du banker (si possible)
netsh advfirewall firewall show rule name="THE_HIVE_BANKER_%BANKER_API_PORT%" >nul 2>nul
if %errorlevel% neq 0 (
    netsh advfirewall firewall add rule name="THE_HIVE_BANKER_%BANKER_API_PORT%" dir=in action=allow protocol=TCP localport=%BANKER_API_PORT% >nul 2>nul
)

if /I "%BANKER_ENABLE_TUNNEL%"=="true" (
    if exist "%BANKER_TUNNEL_KEY%" (
        call :ensure_server_relay
        call :start_reverse_tunnel
    ) else (
        echo WARN: cle SSH du tunnel absente: %BANKER_TUNNEL_KEY%
        echo WARN: Nexus verra le banker comme hors-ligne tant que le tunnel n'est pas provisionne.
    )
) else (
    echo INFO: tunnel SSH desactive pour cette instance Banker.
)

call :check_existing_banker
if %errorlevel% equ 2 (
    echo INFO: une instance saine de The Banker ecoute deja sur le port %BANKER_API_PORT%.
    echo INFO: aucun second lancement n'est autorise.
    exit /b 0
)
if %errorlevel% equ 3 (
    echo ERROR: le port %BANKER_API_PORT% est deja occupe par un autre processus.
    exit /b 1
)

:: Lancement du service
venv\Scripts\python -X utf8 -m uvicorn eva_banker.main:app --host %BANKER_BIND_HOST% --port %BANKER_API_PORT% --env-file "%BANKER_ENV_FILE%" --no-access-log

echo.
echo [%DATE% %TIME%] Processus Banker arrete.
echo Redemarrage dans 5 secondes (Ctrl+C pour annuler)...
timeout /t 5 >nul
goto start

:check_existing_banker
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$ErrorActionPreference = 'SilentlyContinue';" ^
 "$port = %BANKER_API_PORT%;" ^
 "$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1;" ^
 "if (-not $conn) { exit 0 }" ^
 "try { $resp = Invoke-WebRequest -UseBasicParsing -Uri ('http://127.0.0.1:' + $port + '/health') -TimeoutSec 2; if ($resp.StatusCode -eq 200) { exit 2 } } catch {}" ^
 "exit 3"
goto :eof

:ensure_server_relay
echo Initialisation du relay distant du Banker...
"%BANKER_SSH_BIN%" -i "%BANKER_TUNNEL_KEY%" -o StrictHostKeyChecking=accept-new %HIVE_SSH_USER%@%HIVE_SERVER_HOST% "bash /home/aza/The_Hive/scripts/start_banker_tunnel_relay.sh 0.0.0.0 %HIVE_TUNNEL_RELAY_PORT% 127.0.0.1 %HIVE_TUNNEL_REMOTE_PORT%" >nul 2>nul
if %errorlevel% neq 0 (
    echo WARN: impossible de lancer le relay distant sur le serveur.
) else (
    echo Relay distant actif sur le port %HIVE_TUNNEL_RELAY_PORT%.
)
goto :eof

:start_reverse_tunnel
echo Initialisation du reverse tunnel SSH du Banker...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$ErrorActionPreference = 'SilentlyContinue';" ^
 "$key = '%BANKER_TUNNEL_KEY%';" ^
 "$ssh = '%BANKER_SSH_BIN%';" ^
 "$server = '%HIVE_SSH_USER%@%HIVE_SERVER_HOST%';" ^
 "$remote = '127.0.0.1:%HIVE_TUNNEL_REMOTE_PORT%:127.0.0.1:%BANKER_API_PORT%';" ^
 "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'ssh.exe' -and $_.CommandLine -like '*the_hive_banker_tunnel*' -and $_.CommandLine -like ('*127.0.0.1:%HIVE_TUNNEL_REMOTE_PORT%:127.0.0.1:%BANKER_API_PORT%*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force };" ^
 "Start-Process -FilePath $ssh -WindowStyle Hidden -ArgumentList @('-i', $key, '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', '-o', 'ExitOnForwardFailure=yes', '-N', '-R', $remote, $server) | Out-Null"
if %errorlevel% neq 0 (
    echo WARN: echec du demarrage du tunnel SSH.
) else (
    echo Tunnel SSH actif vers %HIVE_SERVER_HOST%:%HIVE_TUNNEL_REMOTE_PORT%.
)
goto :eof
