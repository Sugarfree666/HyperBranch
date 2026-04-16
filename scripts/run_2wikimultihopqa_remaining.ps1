param(
    [int]$StartIndex = 100,
    [int]$EndIndex = 999,
    [string]$PythonExe = "python",
    [string]$BaseUrl = "https://api.chatanywhere.tech/v1",
    [int]$MaxAttempts = 0,
    [int]$RetryDelaySeconds = 20,
    [int]$EvalStartIndex = 0,
    [int]$EvalLimit = 1000,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$QuestionFile = "questions\2wikimultihopqa\questions.json"
$ConfigFile = "configs\2wikimultihopqa.yaml"
$RunsDir = "runs\2wikimultihopqa"

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is required."
}
if (-not $env:OPENAI_BASE_URL) {
    $env:OPENAI_BASE_URL = $BaseUrl
}

if (-not (Test-Path $QuestionFile)) {
    throw "Question file not found: $QuestionFile"
}
if (-not (Test-Path $ConfigFile)) {
    throw "Config file not found: $ConfigFile"
}

New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null
New-Item -ItemType Directory -Force -Path "eval\results\2wikimultihopqa" | Out-Null

function Get-RunDirFromOutput {
    param([object[]]$OutputLines)

    for ($i = $OutputLines.Count - 1; $i -ge 0; $i--) {
        $line = [string]$OutputLines[$i]
        if ($line -match "^run_dir=(.+)$") {
            return $Matches[1].Trim()
        }
    }
    return $null
}

function Test-RunSucceeded {
    param([string]$RunDir)

    if (-not $RunDir) {
        return $false
    }
    return Test-Path (Join-Path $RunDir "artifacts\final_answer.json")
}

function Get-RunErrorMessage {
    param([string]$RunDir)

    $errorPath = Join-Path $RunDir "artifacts\error.json"
    if (-not (Test-Path $errorPath)) {
        return ""
    }
    try {
        $payload = Get-Content $errorPath -Raw | ConvertFrom-Json
        return [string]$payload.error_message
    }
    catch {
        return ""
    }
}

function Test-AuthError {
    param([string]$Message)

    return (
        $Message -match "HTTP 401" -or
        $Message -match "invalid_api_key" -or
        $Message -match "Incorrect API key"
    )
}

function Remove-RetryRunDir {
    param(
        [string]$RunDir,
        [string]$Reason
    )

    if (-not $RunDir -or -not (Test-Path $RunDir)) {
        return
    }

    $resolvedRun = (Resolve-Path $RunDir).Path
    $resolvedRunsRoot = (Resolve-Path $RunsDir).Path
    if (-not $resolvedRun.StartsWith($resolvedRunsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete path outside ${RunsDir}: $resolvedRun"
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] deleting failed run: $resolvedRun"
    if ($Reason) {
        Write-Host "reason: $Reason"
    }
    Remove-Item -LiteralPath $resolvedRun -Recurse -Force
}

function Invoke-QuestionWithRetry {
    param([int]$QuestionIndex)

    $attempt = 1
    while ($true) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] running 2wikimultihopqa index $QuestionIndex attempt $attempt"

        $stdoutPath = Join-Path $env:TEMP "hyperbranch_2wiki_${QuestionIndex}_${attempt}_stdout.log"
        $stderrPath = Join-Path $env:TEMP "hyperbranch_2wiki_${QuestionIndex}_${attempt}_stderr.log"
        try {
            $arguments = @(
                "-m",
                "hyper_branch.cli",
                "--question-file",
                $QuestionFile,
                "--question-index",
                [string]$QuestionIndex,
                "--config",
                $ConfigFile,
                "--allow-failure"
            )

            $process = Start-Process `
                -FilePath $PythonExe `
                -ArgumentList $arguments `
                -WorkingDirectory $ProjectRoot `
                -RedirectStandardOutput $stdoutPath `
                -RedirectStandardError $stderrPath `
                -NoNewWindow `
                -Wait `
                -PassThru

            $exitCode = $process.ExitCode
            $output = if (Test-Path $stdoutPath) { Get-Content $stdoutPath } else { @() }
            $stderrOutput = if (Test-Path $stderrPath) { Get-Content $stderrPath } else { @() }
        }
        finally {
            if (Test-Path $stdoutPath) {
                Remove-Item -LiteralPath $stdoutPath -Force
            }
            if (Test-Path $stderrPath) {
                Remove-Item -LiteralPath $stderrPath -Force
            }
        }

        $stderrOutput | ForEach-Object { Write-Host $_ }
        $output | ForEach-Object { Write-Host $_ }

        $runDir = Get-RunDirFromOutput $output
        if ($exitCode -eq 0 -and (Test-RunSucceeded $runDir)) {
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] index $QuestionIndex succeeded: $runDir"
            return
        }

        $errorMessage = if ($runDir) { Get-RunErrorMessage $runDir } else { "" }
        if (Test-AuthError $errorMessage) {
            throw "Authentication/API-key error for index $QuestionIndex. Fix OPENAI_API_KEY/OPENAI_BASE_URL before retrying. Run dir kept for diagnosis: $runDir"
        }

        if ($runDir) {
            Remove-RetryRunDir $runDir $errorMessage
        }
        else {
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] no run_dir found for failed attempt."
        }

        if ($MaxAttempts -gt 0 -and $attempt -ge $MaxAttempts) {
            throw "Index $QuestionIndex failed after $MaxAttempts attempt(s)."
        }

        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] retrying index $QuestionIndex after ${RetryDelaySeconds}s"
        Start-Sleep -Seconds $RetryDelaySeconds
        $attempt += 1
    }
}

for ($i = $StartIndex; $i -le $EndIndex; $i++) {
    Invoke-QuestionWithRetry -QuestionIndex $i
}

if (-not $SkipEval) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $EvalEndIndex = $EvalStartIndex + $EvalLimit - 1
    $EvalDir = "eval\results\2wikimultihopqa\index_${EvalStartIndex}_${EvalEndIndex}_em_f1_$Timestamp"

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] evaluating EM/F1 into $EvalDir"
    & $PythonExe eval\get_score.py `
        --question-file $QuestionFile `
        --runs-dir $RunsDir `
        --start-index $EvalStartIndex `
        --limit $EvalLimit `
        --output-dir $EvalDir `
        --workers 1 `
        --skip-rsim `
        --skip-gen
}

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] finished"
