$ErrorActionPreference = "Stop"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python was not found. Install Python 3 and try again."
    exit 1
}

$secureToken = Read-Host "Enter a new Hugging Face token (input is hidden)" -AsSecureString
$credential = New-Object System.Net.NetworkCredential -ArgumentList "", $secureToken
$env:HF_TOKEN = $credential.Password
$secureToken.Dispose()
$credential = $null

Push-Location $PSScriptRoot
try {
    $server = Start-Process -FilePath $python.Source -ArgumentList "app.py" -WorkingDirectory $PSScriptRoot -NoNewWindow -PassThru
    $ready = $false
    for ($attempt = 0; $attempt -lt 50; $attempt++) {
        if ($server.HasExited) {
            throw "The chatbot server stopped during startup. Check the Python error above."
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 1
            if ($health.ok) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 200
        }
    }

    if (-not $ready) {
        Stop-Process -Id $server.Id
        throw "The chatbot server did not start on port 8000. Close any other app using that port and try again."
    }

    Start-Process "http://127.0.0.1:8000"
    Write-Host "Orbit is running. Keep this window open; press Ctrl+C here to stop it."
    Wait-Process -Id $server.Id
} finally {
    Pop-Location
}
