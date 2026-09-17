$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repoPath
$runPath = Join-Path $repoPath 'artifacts\l0d_preflight'
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
$pythonPath = (Get-Command python -CommandType Application | Select-Object -First 1).Source
$logSuffix = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$process = Start-Process -FilePath $pythonPath `
    -ArgumentList '-m', 'row.experiments.preflight_l0d' `
    -WorkingDirectory $repoPath -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runPath "stdout_$logSuffix.log") `
    -RedirectStandardError (Join-Path $runPath "stderr_$logSuffix.log")
Write-Output "Started L0d preflight launcher PID $($process.Id)."
Write-Output 'Check: Get-Content artifacts\l0d_preflight\status.json'
Write-Output 'If startup fails before status exists, inspect the newest stderr log in artifacts\l0d_preflight.'
