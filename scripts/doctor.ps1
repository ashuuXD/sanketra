# Pre-flight check. READ-ONLY: it changes nothing. Run it before training, after a Windows hiccup, or when something "hangs".
#   .\scripts\doctor.ps1
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
$dataHome = if ($env:SANKETRA_HOME) { $env:SANKETRA_HOME } else { Join-Path $env:USERPROFILE "SanketraData" }
$script:warn = 0; $script:fail = 0
function Show([string]$level, [string]$msg) {
  $color = @{ OK = "Green"; INFO = "Gray"; WARN = "Yellow"; FAIL = "Red" }[$level]
  Write-Host ("[{0}] {1}" -f $level, $msg) -ForegroundColor $color
  if ($level -eq "WARN") { $script:warn++ } elseif ($level -eq "FAIL") { $script:fail++ }
}
function Test-PortListening([int]$Port) {
  try { return [bool]([Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() | Where-Object { $_.Port -eq $Port }) } catch { return $false }
}

# 1. Where does the project live?
Show "INFO" "Project folder: $root"
if ($root -match '\\(Downloads|Desktop|Documents|OneDrive[^\\]*)(\\|$)') {
  Show "WARN" "The project is inside an indexed folder. Windows Search and antivirus scan every file in .venv (tens of thousands). Add this folder to the Windows Search exclusions and to your antivirus exclusions, or keep the project somewhere like C:\dev (a moved .venv must be recreated with .\scripts\setup.ps1 -Rebuild)."
} else { Show "OK" "Project is not inside Downloads/Desktop/Documents/OneDrive." }
if ($root.Length -gt 100) { Show "WARN" "The project path is long ($($root.Length) characters); Windows paths over 260 characters break tools." }

# 2. Data folder, models, disk space
if (Test-Path $dataHome) {
  Show "OK" "Data folder exists: $dataHome"
  $models = Get-ChildItem (Join-Path $dataHome "models") -Filter "sanketra_*.pt" -ErrorAction SilentlyContinue
  if ($models) { Show "OK" ("Models: " + (($models | ForEach-Object { "{0} ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB) }) -join ", ")) }
  else { Show "WARN" "No sanketra_*.pt in $dataHome\models" }
  try {
    $drive = New-Object System.IO.DriveInfo ($dataHome.Substring(0, 1))
    $freeGB = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    if ($freeGB -lt 5) { Show "FAIL" "Only $freeGB GB free on $($drive.Name)" } elseif ($freeGB -lt 15) { Show "WARN" "Only $freeGB GB free on $($drive.Name)" } else { Show "OK" "$freeGB GB free on $($drive.Name)" }
  } catch { Show "INFO" "Could not read free disk space." }
} else { Show "FAIL" "Data folder not found: $dataHome (it is created by any script that loads _env.ps1)" }
Show "INFO" ("SANKETRA_HOME (this session): " + $(if ($env:SANKETRA_HOME) { $env:SANKETRA_HOME } else { "not set, default is used" }) + "; (saved for your user): " + $(if ([Environment]::GetEnvironmentVariable("SANKETRA_HOME", "User")) { [Environment]::GetEnvironmentVariable("SANKETRA_HOME", "User") } else { "not set" }))

# 3. Python environment
$py = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $py) {
  try {
    $out = & $py -c "import sys, torch, ultralytics, cv2, fastapi, pytest; print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| ultralytics', ultralytics.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
      Show "OK" "$out"
      if ($out -notmatch "CUDA available: True") { Show "WARN" "CUDA is not available to torch: training would run on the CPU (very slow)." }
    } else { Show "FAIL" "The .venv exists but its packages do not import: $out   -> run .\scripts\setup.ps1 -Rebuild" }
  } catch { Show "FAIL" "Could not start the .venv python: $_" }
} else { Show "FAIL" ".venv not found in the project folder -> run .\scripts\setup.ps1" }

# 4. GPU memory in use
try {
  $g = & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>$null
  if ($g) {
    Show "INFO" "GPU: $g"
    $usedMiB = [int](($g -split ",")[1] -replace "[^0-9]", "")
    if ($usedMiB -gt 1500) { Show "WARN" "$usedMiB MiB of GPU memory are already in use: close the app, other training runs or other GPU programs first." } else { Show "OK" "GPU memory is mostly free." }
  }
} catch { Show "INFO" "nvidia-smi not available." }

# 5. Things already running
if (Test-PortListening 8000) { Show "WARN" "Something is listening on port 8000 (the Sanketra server?). Stop it with Ctrl+C in its terminal before training." } else { Show "OK" "Port 8000 is free." }
$procs = @(Get-Process -Name python, pythonw -ErrorAction SilentlyContinue)
if ($procs.Count -gt 0) {
  $list = ($procs | ForEach-Object { $st = try { $_.StartTime.ToString("HH:mm") } catch { "?" }; "PID $($_.Id) since $st" }) -join "; "
  Show "WARN" "$($procs.Count) python process(es) running: $list. If you did not start them just now, stop them (Stop-Process -Id <PID>)."
} else { Show "OK" "No python processes running." }

# 6. Windows Search (informational)
try { $s = Get-Service WSearch -ErrorAction Stop; Show "INFO" "Windows Search service: $($s.Status)" } catch { Show "INFO" "Windows Search service state unknown." }
Show "INFO" "Also check Task Manager: if Disk is near 100% for minutes, wait before starting anything heavy."

Write-Host ""
if ($script:fail -gt 0) { Write-Host "$($script:fail) problem(s) to fix first, $($script:warn) warning(s)." -ForegroundColor Red }
elseif ($script:warn -gt 0) { Write-Host "No failures, $($script:warn) warning(s): read them before training." -ForegroundColor Yellow }
else { Write-Host "All checks passed." -ForegroundColor Green }
