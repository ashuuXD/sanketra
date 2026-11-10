# Phase 7: train on an Indian road-sign dataset you downloaded, then evaluate. Everything goes into SanketraData.
# Usage: .\scripts\phase7_indian.ps1 -Src "C:\path\to\extracted\dataset" [-Name indian_s] [-Base yolo11s.pt] [-Epochs 100] [-Proposal "<weights>"]
#  -Proposal is only needed when the dataset has class folders but NO boxes (a detector proposes the boxes: review them!).
param([Parameter(Mandatory = $true)][string]$Src, [string]$Name = "indian_s", [string]$Base = "yolo11s.pt", [int]$Epochs = 100,
      [int]$Batch = 8, [int]$Imgsz = 640, [string]$Proposal = "")
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env.ps1"
.\.venv\Scripts\Activate.ps1
$H = $env:SANKETRA_HOME
if (-not (Test-Path $Src)) { throw "Dataset folder not found: $Src" }
if (Test-PortListening 8000) { Write-Warning "Stop the Sanketra server (Ctrl+C) first: training needs the GPU memory."; throw "stop the server and run again" }
if ($Proposal -eq "") { python -m training.adapters.indian_import --src $Src --dst "$H\datasets\raw\indian" }
else { python -m training.adapters.indian_import --src $Src --dst "$H\datasets\raw\indian" --proposal-model $Proposal }
if ($LASTEXITCODE -ne 0) { throw "Step failed: import (read the message above; for class folders without boxes add -Proposal <weights>)" }
python -m training.prepare_dataset --src "$H\datasets\raw\indian" --dst "$H\datasets\prepared\indian" --names "$H\datasets\raw\indian\names.txt" --seed 0
if ($LASTEXITCODE -ne 0) { throw "Step failed: prepare_dataset" }
python -m training.registry_sync --names "$H\datasets\raw\indian\names.txt" --source "Indian road-sign dataset class (to be verified)"
python -m training.train --data "$H\datasets\prepared\indian\data.yaml" --model $Base --epochs $Epochs --imgsz $Imgsz --batch $Batch --name $Name
if ($LASTEXITCODE -ne 0) { throw "Step failed: train" }
$best = "$H\experiments\runs\$Name\weights\best.pt"
if (-not (Test-Path $best)) { throw "training did not produce $best" }
Copy-Item $best "$H\models\sanketra_$Name.pt" -Force
$rep = Get-Content "$H\datasets\raw\indian\import_report.json" -Raw | ConvertFrom-Json
$note = "Trained on an Indian road-sign dataset ($($rep.classes) classes, $($rep.images_written) images), $Epochs epochs, base $Base."
if ($rep.machine_made_labels) { $note += " Boxes were proposed by a model, not hand-drawn." }
@{ note = $note; dataset = "Indian"; epochs = $Epochs } | ConvertTo-Json | Set-Content "$H\models\sanketra_$Name.json" -Encoding UTF8
Write-Host "The app picks your newest sanketra_*.pt automatically."
& "$PSScriptRoot\evaluate_gtsdb.ps1" -Name $Name -Imgsz $Imgsz -Data indian
Write-Host "Next: open the Signs page, fill meanings in $H\sign_registry.local.json for the new signs (official source), then restart the app."
