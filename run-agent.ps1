<#
.SYNOPSIS
    Manage the cmw-platform-agent background process.

.DESCRIPTION
    This script starts, stops, checks status, tails logs, or restarts the agent.
    It activates the virtual environment, launches the agent as a hidden background
    process, redirects output to a log file, and stores the PID for later control.

.PARAMETER Action
    The action to perform: start | stop | status | tail | logs | restart. Default: start

.PARAMETER Python
    Python executable to use. Default: "python" (respects active venv).

.PARAMETER ScriptPath
    Path to the agent entrypoint. Default: .\agent_ng\app_ng_modular.py

.PARAMETER VenvPath
    Path to PowerShell activate script. Default: .\.venv\Scripts\Activate.ps1

.PARAMETER LogPath
    Fallback log file path when LOG_FILE environment variable is not set. Default: cmw-agent.log

.PARAMETER PidPath
    Path to the PID file. Default: cmw-agent.pid

.PARAMETER NoVenv
    Skip activating the virtual environment.

.PARAMETER Force
    Force stop all Python processes without confirmation when stopping the agent.

.EXAMPLE
    .\run-agent.ps1
    Starts the agent in background, uses LOG_FILE from environment for logging, writes PID to cmw-agent.pid

.EXAMPLE
    .\run-agent.ps1 -Action status
    Prints whether the agent is running and its PID if available.

.EXAMPLE
    .\run-agent.ps1 -Action tail
    Tails the agent log until interrupted (Ctrl+C).

.EXAMPLE
    .\run-agent.ps1 -Action stop -Force
    Stops the agent and all Python processes without asking for confirmation.

.EXAMPLE
    .\run-agent.ps1 -Action logs
    Shows all available log files with sizes and modification times.

.EXAMPLE
    .\run-agent.ps1 -Action tail
    Follows the latest log file in real-time (respects Python log rotation).

.NOTES
    Ensure PowerShell execution policy allows running local scripts, e.g.:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#>

param (
    [ValidateSet("start","stop","status","tail","restart","logs")]
    [string]$Action = "start",

    [string]$Python = "python",

    [string]$ScriptPath = ".\agent_ng\app_ng_modular.py",

    [string]$VenvPath = ".\.venv\Scripts\Activate.ps1",

    [string]$LogPath = "cmw-agent.log",

    [string]$PidFilePath = "cmw-agent.pid",

    [switch]$NoVenv,

    [switch]$Force
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

    # Resolve Python executable path
    $pythonExe = $Python
    if (-not $NoVenv -and (Test-Path -Path $VenvPath)) {
        $venvDir = Split-Path -Path (Split-Path -Path $VenvPath -Parent) -Parent
        $venvPython = Join-Path $venvDir "Scripts\python.exe"
        if (Test-Path -Path $venvPython) {
            $pythonExe = $venvPython
            Write-Info "Using venv Python: $pythonExe"
        }
    }

    # Resolve script path to absolute path
    $scriptFullPath = Resolve-Path -Path $ScriptPath -ErrorAction Stop
    $workingDir = Split-Path -Path $scriptFullPath -Parent

    Write-Info "Starting agent in background (detached from terminal)..."
    Write-Info "Working directory: $workingDir"
    Write-Info "Script: $scriptFullPath"
    
    # Start Python directly (not through cmd.exe) to ensure complete detachment
    # This ensures the process survives terminal closure
    $proc = Start-Process -FilePath $pythonExe `
        -ArgumentList "`"$scriptFullPath`"" `
        -WorkingDirectory $workingDir `
        -WindowStyle Hidden `
        -PassThru `
        -ErrorAction Stop
    
    if (-not $proc) {
        Write-Err "Failed to start agent process"
        exit 1
    }
    
    # Store the Python process PID (not cmd.exe)
    Set-Content -NoNewline -Path $PidFilePath -Value $proc.Id
    Write-Info "Agent started with PID $($proc.Id) (detached from terminal)."
    Write-Info "Logging configured via LOG_FILE environment variable."
    Write-Info "The agent will continue running even if this terminal is closed."
}

function Stop-Agent {
    $processId = Get-AgentPid -Path $PidFilePath
    $stopped = $false
    
    # First, try to stop the main process if we have a PID
    if ($processId) {
        if (Test-AgentRunning -ProcessId $processId) {
            Write-Info "Stopping main agent process PID $processId..."
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 500
            $stopped = $true
        } else {
            Write-Warn "No process with PID $processId is running. Cleaning up PID file."
            Remove-Item -Path $PidFilePath -ErrorAction SilentlyContinue
        }
    }
    
    # Find and stop all Python processes running our agent script
    Write-Info "Searching for Python processes running the agent..."
    $pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
        try {
            $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
            return $commandLine -and $commandLine.Contains($ScriptPath)
        } catch {
            return $false
        }
    }
    
    # Also find child processes of the main agent process
    if ($processId) {
        $childProcesses = Get-WmiObject Win32_Process | Where-Object { $_.ParentProcessId -eq $processId }
        foreach ($child in $childProcesses) {
            if ($child.ProcessName -eq "python") {
                $pythonProcesses += Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
            }
        }
    }
    
    if ($pythonProcesses) {
        Write-Info "Found $($pythonProcesses.Count) Python process(es) running the agent script"
        foreach ($proc in $pythonProcesses) {
            Write-Info "Stopping Python process PID $($proc.Id)..."
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            $stopped = $true
        }
        Start-Sleep -Milliseconds 500
    }
    
    # Fallback: stop all Python processes if we couldn't find specific ones
    if (-not $stopped) {
        Write-Warn "No specific agent processes found."
        $allPythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue
        if ($allPythonProcesses) {
            Write-Warn "Found $($allPythonProcesses.Count) Python process(es) running."
            Write-Host "Processes:" -ForegroundColor Yellow
            foreach ($proc in $allPythonProcesses) {
                try {
                    $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
                    Write-Host "  - PID $($proc.Id): $commandLine" -ForegroundColor Yellow
                } catch {
                    Write-Host "  - PID $($proc.Id): (unable to get command line)" -ForegroundColor Yellow
                }
            }
            
            if ($Force) {
                Write-Warn "Force mode enabled - stopping all Python processes without confirmation."
                $shouldStop = $true
            } else {
                $response = Read-Host "Do you want to stop ALL Python processes? This may affect other applications. (y/N)"
                $shouldStop = ($response -eq 'y' -or $response -eq 'Y' -or $response -eq 'yes' -or $response -eq 'Yes')
            }
            
            if ($shouldStop) {
                Write-Info "Stopping $($allPythonProcesses.Count) Python process(es)..."
                foreach ($proc in $allPythonProcesses) {
                    Write-Info "Stopping Python process PID $($proc.Id)..."
                    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                }
                $stopped = $true
            } else {
                Write-Warn "Skipping Python process termination. You may need to stop the agent manually."
            }
        }
    }
    
    # Clean up PID file
    if (Test-Path -Path $PidFilePath) { 
        Remove-Item -Path $PidFilePath -ErrorAction SilentlyContinue 
    }
    
    if ($stopped) {
        Write-Info "Agent stopped successfully."
    } else {
        Write-Warn "No running agent processes found."
    }
}

function Show-Status {
    $processId = Get-AgentPid -Path $PidFilePath
    $mainProcessRunning = $false
    $pythonProcessesRunning = 0
    
    # Check main process
    if ($processId -and (Test-AgentRunning -ProcessId $processId)) {
        $mainProcessRunning = $true
        Write-Host "Main process running (PID $processId)" -ForegroundColor Green
    }
    
    # Check for Python processes running our script
    $pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
        try {
            $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
            return $commandLine -and $commandLine.Contains($ScriptPath)
        } catch {
            return $false
        }
    }
    
    # Also check child processes of the main agent process
    if ($processId) {
        $childProcesses = Get-WmiObject Win32_Process | Where-Object { $_.ParentProcessId -eq $processId }
        foreach ($child in $childProcesses) {
            if ($child.ProcessName -eq "python") {
                $childProc = Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
                if ($childProc) {
                    $pythonProcesses += $childProc
                }
            }
        }
    }
    
    if ($pythonProcesses) {
        $pythonProcessesRunning = $pythonProcesses.Count
        Write-Host "Python agent processes running: $pythonProcessesRunning" -ForegroundColor Green
        foreach ($proc in $pythonProcesses) {
            try {
                $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
                $memoryMB = [math]::Round($proc.WorkingSet64 / 1MB, 1)
                Write-Host "  - PID $($proc.Id) (${memoryMB}MB)" -ForegroundColor Cyan
                if ($commandLine) {
                    Write-Host "    Command: $commandLine" -ForegroundColor Gray
                }
            } catch {
                Write-Host "  - PID $($proc.Id)" -ForegroundColor Cyan
            }
        }
    }
    
    if (-not $mainProcessRunning -and $pythonProcessesRunning -eq 0) {
        Write-Host "Not running" -ForegroundColor Yellow
    }
}

function Get-LatestLogFile {
    param([string]$BaseLogPath)
    
    # Check if LOG_FILE is set in environment
    $logFile = [System.Environment]::GetEnvironmentVariable("LOG_FILE")
    if ([string]::IsNullOrEmpty($logFile)) {
        $logFile = $BaseLogPath
    }
    
    # Get the base name without extension
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($logFile)
    $directory = [System.IO.Path]::GetDirectoryName($logFile)
    if ([string]::IsNullOrEmpty($directory)) {
        $directory = "."
    }
    
    # Find all log files matching the pattern (base.log, base.log.1, base.log.2, etc.)
    $pattern = "$baseName.log*"
    $logFiles = @(Get-ChildItem -Path $directory -Name $pattern -ErrorAction SilentlyContinue | Sort-Object)
    
    if ($logFiles.Count -eq 0) {
        return $logFile  # Return original path if no files found
    }
    
    # The latest file is the one with the highest number (or no number for current)
    # Sort by modification time to get the most recent
    $latestFile = $logFiles | ForEach-Object { 
        $fullPath = Join-Path $directory $_
        [PSCustomObject]@{
            Name = $_
            Path = $fullPath
            LastWrite = (Get-Item $fullPath -ErrorAction SilentlyContinue).LastWriteTime
        }
    } | Sort-Object LastWrite -Descending | Select-Object -First 1
    
    if ($latestFile) {
        return $latestFile.Path
    }
    
    return $logFile
}

function Tail-Logs {
    $latestLogFile = Get-LatestLogFile -BaseLogPath $LogPath
    
    if (-not (Test-Path -Path $latestLogFile)) {
        Write-Warn "Log file not found at $latestLogFile. It will be created after start."
    }
    Write-Info "Tailing logs: $latestLogFile (Ctrl+C to stop)"
    Write-Info "Note: This shows the latest log file. Use 'Get-ChildItem cmw-agent.log*' to see all rotated files."
    Get-Content -Path $latestLogFile -Wait -ErrorAction SilentlyContinue
}

function Show-LogFiles {
    # Check if LOG_FILE is set in environment
    $logFile = [System.Environment]::GetEnvironmentVariable("LOG_FILE")
    if ([string]::IsNullOrEmpty($logFile)) {
        $logFile = $LogPath
    }
    
    # Get the base name without extension
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($logFile)
    $directory = [System.IO.Path]::GetDirectoryName($logFile)
    if ([string]::IsNullOrEmpty($directory)) {
        $directory = "."
    }
    
    # Find all log files matching the pattern
    $pattern = "$baseName.log*"
    $logFiles = @(Get-ChildItem -Path $directory -Name $pattern -ErrorAction SilentlyContinue | Sort-Object)
    
    if ($logFiles.Count -eq 0) {
        Write-Warn "No log files found matching pattern: $pattern"
        return
    }
    
    Write-Info "Available log files (sorted by modification time):"
    Write-Host ""
    
    $logFiles | ForEach-Object { 
        $fullPath = Join-Path $directory $_
        $fileInfo = Get-Item $fullPath -ErrorAction SilentlyContinue
        if ($fileInfo) {
            $size = [math]::Round($fileInfo.Length / 1KB, 2)
            $lastWrite = $fileInfo.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
            $isLatest = ($_ -eq $logFiles[0])  # First in sorted list is usually latest
            
            $status = if ($isLatest) { " (LATEST)" } else { "" }
            Write-Host "  $_" -NoNewline
            Write-Host $status -ForegroundColor Green -NoNewline
            Write-Host " - $size KB - $lastWrite"
        }
    }
    
    Write-Host ""
    Write-Info "Use '.\run-agent.ps1 -Action tail' to follow the latest log file"
}

switch ($Action) {
    'start'   { Start-Agent }
    'stop'    { Stop-Agent }
    'status'  { Show-Status }
    'tail'    { Tail-Logs }
    'logs'    { Show-LogFiles }
    'restart' { Stop-Agent; Start-Agent }
}


