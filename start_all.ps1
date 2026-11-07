# Start the local demo. Existing healthy services are reused.
[CmdletBinding()]
param([ValidateRange(5, 600)][int]$StartupTimeoutSeconds = 120)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$logDirectory = Join-Path $projectRoot 'tmp'

function Test-ServiceHealth([string]$Url, [string]$Kind) {
    try {
        $response = Invoke-RestMethod $Url -TimeoutSec 2
        switch ($Kind) {
            'api' { return ($response.status -eq 'ok' -and $response.generation_pipeline -eq 'source-excerpts-v1' -and $response.comparison_retrieval -eq 'balanced-v1') }
            'qdrant' { return ($response.status -eq 'ok') }
            'ui' { return ($response -eq 'ok') }
        }
    } catch { return $false }
    return $false
}

function Test-LocalPort([int]$Port) {
    $connection = [System.Net.Sockets.TcpClient]::new()
    try { $connection.Connect('127.0.0.1', $Port); return $true }
    catch { return $false }
    finally { $connection.Dispose() }
}

function Start-DemoService([string]$Name, [int]$Port, [string]$HealthUrl, [string[]]$Arguments) {
    if (Test-ServiceHealth $HealthUrl $Name) {
        Write-Host "$Name already running on port $Port."
        return
    }
    if (Test-LocalPort $Port) {
        throw "Port $Port is occupied by an unhealthy or different service. Stop that service and rerun the launcher."
    }
    $stdout = Join-Path $logDirectory "$Name-out.log"
    $stderr = Join-Path $logDirectory "$Name-err.log"
    $process = Start-Process -FilePath $python -ArgumentList $Arguments `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $process.Id | Set-Content (Join-Path $logDirectory "$Name.pid")
    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    do {
        if (Test-ServiceHealth $HealthUrl $Name) {
            Write-Host "$Name running on port $Port (PID $($process.Id))."
            return
        }
        $process.Refresh()
        if ($process.HasExited) { throw "$Name exited during startup. See $stderr" }
        Start-Sleep -Seconds 1
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "$Name did not respond within $StartupTimeoutSeconds seconds. See $stderr. The process may still be starting."
}

try {
    if (-not (Test-Path -LiteralPath $python)) {
        throw 'Missing .venv. Follow the README installation instructions first.'
    }
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    if (-not (Test-ServiceHealth 'http://127.0.0.1:6333/collections' 'qdrant')) {
        $dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
        $docker = if ($dockerCommand) { $dockerCommand.Source } else {
            Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'
        }
        if (-not (Test-Path -LiteralPath $docker)) {
            throw 'Qdrant is unavailable. Start Docker Desktop and Qdrant, then retry.'
        }
        & $docker compose -f (Join-Path $projectRoot 'docker-compose.yml') up -d
        if ($LASTEXITCODE -ne 0) { throw 'Docker could not start Qdrant. Check Docker Desktop.' }
        $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
        while (-not (Test-ServiceHealth 'http://127.0.0.1:6333/collections' 'qdrant')) {
            if ([DateTime]::UtcNow -ge $deadline) { throw 'Qdrant did not become available.' }
            Start-Sleep -Seconds 1
        }
    }
    Write-Host 'Qdrant is available.'
    Start-DemoService -Name 'api' -Port 8000 -HealthUrl 'http://127.0.0.1:8000/health' `
        -Arguments @('-m', 'uvicorn', 'app.api.main:app', '--host', '127.0.0.1', '--port', '8000', '--workers', '1')
    Start-DemoService -Name 'ui' -Port 8501 -HealthUrl 'http://127.0.0.1:8501/_stcore/health' `
        -Arguments @('-m', 'streamlit', 'run', 'ui/streamlit_app.py', '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true')
    Write-Host "`nOpen http://127.0.0.1:8501"
    Write-Host 'API documentation: http://127.0.0.1:8000/docs'
    Write-Host "Logs and process IDs: $logDirectory"
    Write-Host 'Services run in the background. The first answer loads the retrieval models.'
    Write-Host 'Startup checks confirm HTTP availability; use the README demo checklist to verify answers.'
} catch {
    Write-Error $_
    exit 1
}
