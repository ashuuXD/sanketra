# Sets up the data folder (default: %USERPROFILE%\SanketraData, outside the repository). Dot-sourced by the other scripts.
# SANKETRA_HOME is set for the current session only; this script never changes your permanent Windows settings.
# To use another folder, set $env:SANKETRA_HOME before running a script (or set it yourself in Windows environment variables).
if (-not $env:SANKETRA_HOME) { $env:SANKETRA_HOME = Join-Path $env:USERPROFILE "SanketraData" }
foreach ($d in "models", "datasets", "experiments\results", "experiments\runs", "data\uploads") {
  New-Item -ItemType Directory -Force -Path (Join-Path $env:SANKETRA_HOME $d) | Out-Null
}
Write-Host "Your data folder: $env:SANKETRA_HOME  (models, datasets, results live here, not in the project folder)"
$env:PYTHONUNBUFFERED = "1"     # show Python progress immediately instead of after the step ends
function Test-PortListening([int]$Port) {   # fast .NET check (Get-NetTCPConnection can hang for minutes on some Windows setups)
  try { return [bool]([Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() | Where-Object { $_.Port -eq $Port }) } catch { return $false }
}
