# Runs all four evaluations for a trained model (detection report, latency, calibration, robustness).
# Usage: .\scripts\evaluate_gtsdb.ps1 [-Name gtsdb_n] [-Imgsz 640]
param([string]$Name = "gtsdb_n", [int]$Imgsz = 640, [string]$Data = "gtsdb")
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env.ps1"
.\.venv\Scripts\Activate.ps1
$H = $env:SANKETRA_HOME; $m = "$H\models\sanketra_$Name.pt"; $d = "$H\datasets\prepared\$Data\data.yaml"
if (-not (Test-Path $m)) { throw "Model not found: $m" }
if (-not (Test-Path $d)) { throw "Prepared dataset not found: $d (run the training script first)" }
python -m evaluation.detection_report --model $m --data $d --imgsz $Imgsz --tag $Name
if ($LASTEXITCODE -ne 0) { throw "Step failed: detection_report" }
python -m evaluation.benchmark_latency --models $m --imgsz 320 416 640 --tag $Name
if ($LASTEXITCODE -ne 0) { throw "Step failed: benchmark_latency" }
python -m evaluation.calibration_eval --model $m --data $d --imgsz $Imgsz --tag $Name
if ($LASTEXITCODE -ne 0) { throw "Step failed: calibration_eval" }
Write-Host "Robustness runs about 30 validation passes (several minutes). Do not close this window."
python -m evaluation.robustness --model $m --data $d --split test --imgsz $Imgsz --tag $Name
if ($LASTEXITCODE -ne 0) { throw "Step failed: robustness" }
Write-Host "Done. Start the app (.\scripts\run.ps1) and open the Results tab."
