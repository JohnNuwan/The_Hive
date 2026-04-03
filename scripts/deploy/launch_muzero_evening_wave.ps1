param(
    [double]$Hours = 5.5,
    [int]$MaxGenerations = 3,
    [int]$GenerationSize = 4,
    [int]$PollSeconds = 20,
    [string]$RandomSeed = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonCandidates = @(
    "C:\Users\nandi\Desktop\The Hive\The_Hive\venv\Scripts\python.exe",
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)
) | Where-Object { $_ -and (Test-Path $_) }

if (-not $pythonCandidates) {
    throw "Aucun interpreteur Python exploitable n'a ete trouve pour lancer la vague MuZero du soir."
}

$pythonExe = $pythonCandidates[0]
$scriptPath = Join-Path $repoRoot "scripts\deploy\run_muzero_night_ga_hunt.py"
if (-not (Test-Path $scriptPath)) {
    throw "Le script de chasse MuZero nocturne est introuvable: $scriptPath"
}

$arguments = @(
    $scriptPath,
    "--hours", $Hours.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--max-generations", $MaxGenerations,
    "--generation-size", $GenerationSize,
    "--poll-seconds", $PollSeconds
)

if (-not [string]::IsNullOrWhiteSpace($RandomSeed)) {
    $arguments += @("--random-seed", $RandomSeed.Trim())
}

Write-Host "Lancement de la vague MuZero du soir..." -ForegroundColor Cyan
Write-Host ("Python : {0}" -f $pythonExe)
Write-Host ("Commande : {0} {1}" -f $pythonExe, ($arguments -join " "))

Push-Location $repoRoot
try {
    & $pythonExe @arguments
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "La vague MuZero du soir s'est terminee avec le code $exitCode."
}

Write-Host "Vague MuZero du soir terminee." -ForegroundColor Green
