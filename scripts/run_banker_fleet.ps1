param(
    [switch]$SkipPrelogin
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root "venv\Scripts\python.exe"
$SupervisorPidDir = Join-Path $Root "logs\banker_supervisors"

$Instances = @(
    @{ Port = 8110; Env = "instances/.env.banker.ftmo50k.local"; Host = "127.0.0.1" },
    # --- Comptes retires le 2026-06-04 : evolues en funded, anciens credentials invalides ---
    # @{ Port = 8170; Env = "instances/.env.banker.ftmo541264545.local"; Host = "127.0.0.1" },
    @{ Port = 8190; Env = "instances/.env.banker.ftmo541283689.local"; Host = "127.0.0.1" },
    # @{ Port = 8120; Env = "instances/.env.banker.ftuk100k.local"; Host = "127.0.0.1" },  # MORT: terminal MT5 reassigne + quasi breach ($261 restant)
    # @{ Port = 8130; Env = "instances/.env.banker.ftuk100k_333382206.local"; Host = "127.0.0.1" },
    # Le compte 333382355 est retire du fleet actif.
    # Le compte 333382356 reste mis de cote jusqu'a validation des identifiants.
    # @{ Port = 8180; Env = "instances/.env.banker.ftuk100k_333382439.local"; Host = "127.0.0.1" },
    @{ Port = 8100; Env = "instances/.env.banker.master.local"; Host = "0.0.0.0" }
)

function Stop-BankerApis {
    $allProcesses = Get-CimInstance Win32_Process
    $bankerChildren = $allProcesses | Where-Object {
        $_.CommandLine -like "*eva_banker.banker_runner*" -or
        $_.CommandLine -like "*uvicorn eva_banker.main*"
    }
    $bankerParentIds = $bankerChildren |
        Where-Object { $_.ParentProcessId -and $_.ParentProcessId -ne $PID } |
        Select-Object -ExpandProperty ParentProcessId -Unique
    $bankerParents = $allProcesses | Where-Object {
        $bankerParentIds -contains $_.ProcessId -and
        $_.ProcessId -ne $PID -and
        $_.Name -in @("powershell.exe", "pwsh.exe", "cmd.exe")
    }

    if (Test-Path $SupervisorPidDir) {
        Get-ChildItem -Path $SupervisorPidDir -Filter "*.pid" -ErrorAction SilentlyContinue | ForEach-Object {
            $pidText = Get-Content -Path $_.FullName -Raw -ErrorAction SilentlyContinue
            $supervisorPid = 0
            if ([int]::TryParse(($pidText -as [string]).Trim(), [ref]$supervisorPid)) {
                Stop-Process -Id $supervisorPid -Force -ErrorAction SilentlyContinue
            }
        }
        Remove-Item -Path (Join-Path $SupervisorPidDir "*.pid") -Force -ErrorAction SilentlyContinue
    }

    $processes = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*eva_banker.banker_runner*" -or
        $_.CommandLine -like "*uvicorn eva_banker.main*" -or
        $_.CommandLine -like "*run_one_banker.ps1*" -or
        $_.CommandLine -like "*banker.master.bat*" -or
        $_.CommandLine -like "*banker.ftmo50k.bat*" -or
        $_.CommandLine -like "*banker.ftmo541264545.bat*" -or
        $_.CommandLine -like "*banker.ftmo541283689.bat*" -or
        $_.CommandLine -like "*banker.ftuk100k*"
    }

    foreach ($process in $bankerParents) {
        try {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Warning "Impossible d'arreter le superviseur Banker $($process.ProcessId): $_"
        }
    }

    foreach ($process in $processes) {
        try {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Warning "Impossible d'arreter le processus $($process.ProcessId): $_"
        }
    }

    Remove-Item (Join-Path $Root "logs\mt5_account_claims\*.json") -Force -ErrorAction SilentlyContinue
}

function ConvertTo-EncodedPowerShellCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script
    )

    $bytes = [System.Text.Encoding]::Unicode.GetBytes($Script)
    return [Convert]::ToBase64String($bytes)
}

function Quote-PowerShellLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return "'" + ($Value -replace "'", "''") + "'"
}

function Start-BankerInstance {
    param(
        [hashtable]$Instance
    )

    $envFile = Join-Path $Root $Instance.Env
    $rootLiteral = Quote-PowerShellLiteral -Value $Root
    $pythonLiteral = Quote-PowerShellLiteral -Value $Python
    $hostLiteral = Quote-PowerShellLiteral -Value ([string]$Instance.Host)
    $envFileLiteral = Quote-PowerShellLiteral -Value $envFile
    $pyPathLiteral = Quote-PowerShellLiteral -Value ((Join-Path $Root "src\shared") + ";" + (Join-Path $Root "src\eva-banker"))
    $script = @"
Set-Location -LiteralPath $rootLiteral
`$env:PYTHONUTF8 = '1'
`$env:PYTHONIOENCODING = 'utf-8'
`$env:PYTHONPATH = $pyPathLiteral
`$env:BANKER_ENV_FILE = $envFileLiteral
while (`$true) {
    & $pythonLiteral -X utf8 -m eva_banker.banker_runner --host $hostLiteral --port $($Instance.Port) --env-file $envFileLiteral
    Start-Sleep -Seconds 5
}
"@
    $encodedCommand = ConvertTo-EncodedPowerShellCommand -Script $script
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-EncodedCommand", $encodedCommand
    )

    New-Item -ItemType Directory -Path $SupervisorPidDir -Force | Out-Null
    $supervisor = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    Set-Content -Path (Join-Path $SupervisorPidDir ("{0}.pid" -f $Instance.Port)) -Value $supervisor.Id -Encoding ASCII
}

function Test-BankerHealth {
    param(
        [int]$Port
    )

    try {
        $health = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/health" -f $Port) -TimeoutSec 5
        return "{0}: {1} mt5={2}" -f $Port, $health.status, $health.mt5_connected
    } catch {
        return "{0}: DOWN" -f $Port
    }
}

function Test-EnvRequiresAutoLogin {
    param(
        [string]$EnvFile
    )

    if (-not (Test-Path $EnvFile)) {
        return $false
    }

    $content = Get-Content -Path $EnvFile -ErrorAction SilentlyContinue
    $hasLogin = [bool]($content | Select-String -Pattern "^\s*MT5_LOGIN\s*=" -Quiet)
    $hasPassword = [bool]($content | Select-String -Pattern "^\s*MT5_PASSWORD\s*=" -Quiet)
    $hasServer = [bool]($content | Select-String -Pattern "^\s*MT5_SERVER\s*=" -Quiet)
    return ($hasLogin -and $hasPassword -and $hasServer)
}

Set-Location $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Join-Path $Root "src\shared") + ";" + (Join-Path $Root "src\eva-banker")

Stop-BankerApis
Start-Sleep -Seconds 3

$preloginStatus = @{}
if (-not $SkipPrelogin) {
    foreach ($instance in $Instances) {
        $envFile = Join-Path $Root $instance.Env
        & $Python (Join-Path $Root "scripts\prelogin_mt5_accounts.py") $envFile
        $preloginStatus[$instance.Env] = ($LASTEXITCODE -eq 0)
    }
}

foreach ($instance in $Instances) {
    $envFile = Join-Path $Root $instance.Env
    if (
        -not $SkipPrelogin -and
        $preloginStatus.ContainsKey($instance.Env) -and
        -not $preloginStatus[$instance.Env] -and
        (Test-EnvRequiresAutoLogin -EnvFile $envFile)
    ) {
        Write-Warning ("Instance ignoree car le pre-login MT5 a echoue: {0} (port {1})" -f $instance.Env, $instance.Port)
        continue
    }

    Start-BankerInstance -Instance $instance
    Start-Sleep -Seconds 5
}

Start-Sleep -Seconds 30
foreach ($instance in $Instances) {
    Test-BankerHealth -Port $instance.Port
}

Write-Host "Keeping banker fleet supervisor alive to prevent sandbox process pruning."
while ($true) {
    Start-Sleep -Seconds 3600
}

