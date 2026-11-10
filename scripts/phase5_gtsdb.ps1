# Phase 5: train a real sign detector on GTSDB, then evaluate it. Everything is stored in your data folder
# (SanketraData), so you can delete and re-unzip the project afterwards without losing anything.
# Baseline comparison: .\scripts\phase5_gtsdb.ps1 -Name gtsdb_s -Base yolo11s.pt   (trains a larger model; compare in Results)
# Seeds: -Seed 1 changes only the training seed (the train/val/test split stays fixed), to measure run-to-run variation.
# Usage: .\scripts\phase5_gtsdb.ps1 [-Src "C:\path\to\FullIJCNN2013"] [-Mode category|full] [-Epochs 100] [-Name gtsdb_n]
param([string]$Src = "$env:USERPROFILE\Downloads\FullIJCNN2013", [string]$Mode = "category", [int]$Epochs = 100, [string]$Name = "gtsdb_n", [int]$Batch = 8, [int]$Imgsz = 640, [string]$Base = "yolo11n.pt", [int]$Seed = 0)
$ErrorActionPreference = "Continue"   # python writes progress to stderr; we check exit codes explicitly
. "$PSScriptRoot\_env.ps1"
.\.venv\Scripts\Activate.ps1
$H = $env:SANKETRA_HOME
$gt = Get-ChildItem -Path $Src -Recurse -Filter gt.txt -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $gt) { throw "gt.txt not found under $Src. Pass -Src with the folder where you extracted FullIJCNN2013.zip." }
$Src = $gt.DirectoryName
Write-Host "Using GTSDB folder: $Src"
if (Test-PortListening 8000) { Write-Warning "The Sanketra server is still running (port 8000). Stop it with Ctrl+C in its terminal first: training needs the GPU memory."; throw "stop the server and run again" }
Write-Host "[1/4] Converting the GTSDB images (a few minutes; this step prints little, wait for it)"
python -m training.adapters.gtsdb --src $Src --dst "$H\datasets\raw\gtsdb" --mode $Mode
if ($LASTEXITCODE -ne 0) { throw "Step failed: gtsdb adapter" }
Write-Host "[2/4] Splitting into train/val/test"
python -m training.prepare_dataset --src "$H\datasets\raw\gtsdb" --dst "$H\datasets\prepared\gtsdb" --names "$H\datasets\raw\gtsdb\names.txt" --group-regex "^(blk\d+)_" --seed 0
if ($LASTEXITCODE -ne 0) { throw "Step failed: prepare_dataset" }
Write-Host "[3/4] Training (one line per epoch; Ctrl+C stops it)"
python -m training.train --data "$H\datasets\prepared\gtsdb\data.yaml" --model $Base --epochs $Epochs --imgsz $Imgsz --batch $Batch --name $Name --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Step failed: train" }
$best = "$H\experiments\runs\$Name\weights\best.pt"
if (-not (Test-Path $best)) { throw "training did not produce $best" }
Copy-Item $best "$H\models\sanketra_$Name.pt" -Force
$note = "Trained on GTSDB (German Traffic Sign Detection Benchmark), $Mode mode, $Epochs epochs. Detects sign GROUPS (prohibitory, danger, mandatory, other) for German signs. It does not recognise Indian signs."
if ($Mode -eq "full") { $note = "Trained on GTSDB (German signs), all 43 classes, $Epochs epochs. It does not recognise Indian signs." }
@{ note = $note; dataset = "GTSDB"; mode = $Mode; epochs = $Epochs } | ConvertTo-Json | Set-Content "$H\models\sanketra_$Name.json" -Encoding UTF8
Write-Host "The app picks your newest sanketra_*.pt in $H\models automatically."
Write-Host "[4/4] Evaluating (detection, latency, calibration, robustness: several minutes)"
& "$PSScriptRoot\evaluate_gtsdb.ps1" -Name $Name -Imgsz $Imgsz
