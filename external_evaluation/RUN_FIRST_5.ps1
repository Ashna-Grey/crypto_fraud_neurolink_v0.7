$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
if (-not $env:ALCHEMY_RPC_URL -and -not $env:ALCHEMY_API_KEY) { throw 'Set your private Alchemy key/URL in this PowerShell session first.' }
python -m external_evaluation.run --probe-only
if ($LASTEXITCODE -ne 0) { throw 'API probe failed. Do not consume Alchemy credits.' }
python -m external_evaluation.run --max-wallets 5
if ($LASTEXITCODE -ne 0) { throw 'Evaluation returned an error.' }
Write-Output 'Open external_evaluation/results/report.html'
