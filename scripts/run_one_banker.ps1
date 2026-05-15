param(
    [Parameter(Mandatory = $true)]
    [string]$Root,

    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$BindHost,

    [Parameter(Mandatory = $true)]
    [int]$Port,

    [Parameter(Mandatory = $true)]
    [string]$EnvFile
)

$ErrorActionPreference = "Stop"

Set-Location $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Join-Path $Root "src\shared") + ";" + (Join-Path $Root "src\eva-banker")
$env:BANKER_ENV_FILE = $EnvFile

while ($true) {
    & $Python -X utf8 -m eva_banker.banker_runner --host $BindHost --port $Port --env-file $EnvFile
    Start-Sleep -Seconds 5
}
