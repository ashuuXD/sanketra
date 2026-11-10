# One-time setup (Windows PowerShell). Run from the repo root: .\scripts\setup.ps1
param([switch]$Rebuild)   # -Rebuild: delete and recreate .venv (slow; only if it is broken)
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env.ps1"
# Prefer Python 3.12 (PyTorch CUDA wheels lag behind the newest Python)
$py = "python"
try { py -3.12 --version *> $null; if ($LASTEXITCODE -eq 0) { $py = "py -3.12" } } catch {}
Write-Host "Using:" ; Invoke-Expression "$py --version"; node --version; git --version
try { nvidia-smi --query-gpu=name,memory.total --format=csv } catch { Write-Host "nvidia-smi not found: CPU mode" }
$haveEnv = $false
if ((Test-Path ".\.venv\Scripts\python.exe") -and -not $Rebuild) {
  & ".\.venv\Scripts\python.exe" -c "import torch, ultralytics, cv2, fastapi" *> $null
  if ($LASTEXITCODE -eq 0) { $haveEnv = $true; Write-Host "Existing .venv works and is kept (use -Rebuild to recreate it)." }
}
Set-ExecutionPolicy -Scope Process Bypass -Force
if ($haveEnv) { .\.venv\Scripts\Activate.ps1 } else {
  if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
  Invoke-Expression "$py -m venv .venv"
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  # Install a CUDA build of torch FIRST (try newest index first), then the rest.
  $ok = $false
  foreach ($c in "cu128", "cu126", "cu124") {
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$c"
    if ($LASTEXITCODE -eq 0) { $ok = $true; Write-Host "Installed torch from $c"; break }
  }
  if (-not $ok) { Write-Warning "No CUDA torch wheel for this Python. Falling back to CPU torch (see docs/TROUBLESHOOTING.md)." }
  pip install -r requirements.txt
}
Push-Location frontend; npm install; npm run build; Pop-Location
python scripts/smoke_test.py
python scripts/make_demo_video.py
Write-Host "Setup done. Check the line 'CUDA available: True' above. Start the app with .\scripts\run.ps1"
