param(
    [int]$StartIndex = 0,
    [int]$EndIndex = 199,
    [string]$PythonExe = "C:\Users\sugarfree\.conda\envs\GoTHyper\python.exe",
    [string]$BaseUrl = "https://api.chatanywhere.tech/v1",
    [switch]$IncludeGenEval
)

$ErrorActionPreference = "Continue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is required."
}
if (-not $env:OPENAI_BASE_URL) {
    $env:OPENAI_BASE_URL = $BaseUrl
}

New-Item -ItemType Directory -Force -Path "runs\mix" | Out-Null
New-Item -ItemType Directory -Force -Path "eval\results\mix" | Out-Null

for ($i = $StartIndex; $i -le $EndIndex; $i++) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] running mix question index $i"
    & $PythonExe -m hyper_branch.cli `
        --question-file questions\mix\questions.json `
        --question-index $i `
        --config configs\mix.yaml `
        --allow-failure

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] question index $i exited with code $LASTEXITCODE"
    }
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$EvalDir = "eval\results\mix\full_$Timestamp"
$EvalArgs = @(
    "eval\get_score.py",
    "--question-file", "questions\mix\questions.json",
    "--runs-dir", "runs\mix",
    "--limit", "200",
    "--output-dir", $EvalDir,
    "--workers", "1"
)

if (-not $IncludeGenEval) {
    $EvalArgs += "--skip-gen"
}

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] evaluating runs into $EvalDir"
& $PythonExe @EvalArgs
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] batch finished"
