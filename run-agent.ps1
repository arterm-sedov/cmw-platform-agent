<#
.SYNOPSIS
    Manage the cmw-platform-agent background process.

.DESCRIPTION
    This script starts, stops, checks status, tails logs, or restarts the agent.
    It activates the virtual environment, launches the agent as a hidden background
    process, redirects output to a log file, and stores the PID for later control.

.PARAMETER Action
    The action to perform: start | stop | status | tail | restart. Default: start

.PARAMETER Python
    Python executable to use. Default: "python" (respects active venv).

.PARAMETER ScriptPath
    Path to the agent entrypoint. Default: .\agent_ng\app_ng_modular.py

.PARAMETER VenvPath
    Path to PowerShell activate script. Default: .\.venv\Scripts\Activate.ps1

.PARAMETER LogPath
    Path to the agent log file. Default: cmw-agent.log

.PARAMETER PidPath
    Path to the PID file. Default: cmw-agent.pid

.PARAMETER NoVenv
    Skip activating the virtual environment.

.EXAMPLE
    .\run-agent.ps1
    Starts the agent in background, logs to cmw-agent.log, writes PID to cmw-agent.pid

.EXAMPLE
    .\run-agent.ps1 -Action status
    Prints whether the agent is running and its PID if available.

.EXAMPLE
    .\run-agent.ps1 -Action tail
    Tails the agent log until interrupted (Ctrl+C).

.NOTES
    Ensure PowerShell execution policy allows running local scripts, e.g.:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#>

param (
    [ValidateSet("start","stop","status","tail","restart")]
    [string]$Action = "start",

    [string]$Python = "python",

    [string]$ScriptPath = ".\agent_ng\app_ng_modular.py",

    [string]$VenvPath = ".\.venv\Scripts\Activate.ps1",

    [string]$LogPath = "cmw-agent.log",

    [string]$PidFilePath = "cmw-agent.pid",

    [switch]$NoVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Warning $Message
}

function Write-Err {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
}

function Activate-Venv {
    param([string]$ActivatePath)
    if ($NoVenv) { return }
    if (-not (Test-Path -Path $ActivatePath)) {
        Write-Warn "Venv activate script not found at $ActivatePath. Skipping activation."
        return
    }
    Write-Info "Activating venv: $ActivatePath"
    . $ActivatePath
}

function Get-AgentPid {
    param([string]$Path)
    if (-not (Test-Path -Path $Path)) { return $null }
    try {
        $pidText = Get-Content -Path $Path -TotalCount 1 -ErrorAction Stop
        if ([string]::IsNullOrWhiteSpace($pidText)) { return $null }
        return [int]$pidText.Trim()
    } catch {
        return $null
    }
}

function Test-AgentRunning {
    param([int]$ProcessId)
    if (-not $ProcessId) { return $false }
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    return [bool]$proc
}

function Start-Agent {
    if (Test-Path -Path $PidFilePath) {
        $existingPid = Get-AgentPid -Path $PidFilePath
        if (Test-AgentRunning -ProcessId $existingPid) {
            Write-Info "Agent already running with PID $existingPid"
            return
        }
    }

    Activate-Venv -ActivatePath $VenvPath

    Write-Info "Starting agent in background..."
    $arguments = "/c $Python `"$ScriptPath`" >> `"$LogPath`" 2>>&1"
    $proc = Start-Process -FilePath cmd.exe -ArgumentList $arguments -WindowStyle Hidden -PassThru
    if (-not $proc) {
        Write-Err "Failed to start agent process"
        exit 1
    }
    Set-Content -NoNewline -Path $PidFilePath -Value $proc.Id
    Write-Info "Agent started with PID $($proc.Id). Logging to $LogPath"
}

function Stop-Agent {
    $processId = Get-AgentPid -Path $PidFilePath
    if (-not $processId) {
        Write-Warn "No PID found at $PidFilePath. Agent may not be running."
        return
    }
    if (-not (Test-AgentRunning -ProcessId $processId)) {
        Write-Warn "No process with PID $processId is running. Cleaning up PID file."
        Remove-Item -Path $PidFilePath -ErrorAction SilentlyContinue
        return
    }
    Write-Info "Stopping agent PID $processId..."
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 300
    if (Test-AgentRunning -ProcessId $processId) {
        Write-Warn "Process still running; attempting graceful stop again."
        Stop-Process -Id $processId -ErrorAction SilentlyContinue
    }
    if (Test-Path -Path $PidFilePath) { Remove-Item -Path $PidFilePath -ErrorAction SilentlyContinue }
    Write-Info "Agent stopped."
}

function Show-Status {
    $processId = Get-AgentPid -Path $PidFilePath
    if ($processId -and (Test-AgentRunning -ProcessId $processId)) {
        Write-Host "Running (PID $processId)" -ForegroundColor Green
    } else {
        Write-Host "Not running" -ForegroundColor Yellow
    }
}

function Tail-Logs {
    if (-not (Test-Path -Path $LogPath)) {
        Write-Warn "Log file not found at $LogPath. It will be created after start."
    }
    Write-Info "Tailing logs: $LogPath (Ctrl+C to stop)"
    Get-Content -Path $LogPath -Wait -ErrorAction SilentlyContinue
}

switch ($Action) {
    'start'   { Start-Agent }
    'stop'    { Stop-Agent }
    'status'  { Show-Status }
    'tail'    { Tail-Logs }
    'restart' { Stop-Agent; Start-Agent }
}


