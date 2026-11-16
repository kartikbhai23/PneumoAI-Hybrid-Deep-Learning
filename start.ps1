$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project environment missing. Run uv sync first.'
}
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Write-Host 'Port 8000 is already in use. Opening the existing server.'
} else {
    Start-Process -FilePath $python -ArgumentList '-m','uvicorn','main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'server.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'server.stderr.log')
}
Write-Host 'Checking evaluation results for the current checkpoint...'
& $python (Join-Path $PSScriptRoot 'evaluate.py') --if-needed
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Evaluation failed. The app can still open; see the evaluation error above.'
}
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $response = Invoke-WebRequest 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Start-Process 'http://127.0.0.1:8000/'
            Write-Host 'PneumoAI is running at http://127.0.0.1:8000/'
            exit 0
        }
    } catch { }
    Start-Sleep -Seconds 1
}
Get-Content (Join-Path $PSScriptRoot 'server.stderr.log') -Tail 20
throw 'Server did not become ready. See server.stderr.log.'
