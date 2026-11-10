# Phase 5: train on real signs (GTSDB) and measure

## 1. Get the data
Source: the German Traffic Sign Detection Benchmark page, https://benchmark.ini.rub.de/gtsdb_dataset.html
Download **FullIJCNN2013.zip** (900 full images 1360x800 as .ppm, ground-truth `gt.txt`, ReadMe.txt) and extract it with a short destination path.
Read the benchmark's terms before using or sharing the data. The dataset is not part of this repository.

## 2. Check the category mapping
`configs/gtsdb_categories.json` maps GTSDB class ids to the benchmark's categories. Verify it against the ReadMe of your copy:
```powershell
python -m training.adapters.check_gtsdb_categories --src "C:\path\to\FullIJCNN2013"
```
It prints `OK: all 43 classes match the ReadMe categories` or lists every difference (nothing is changed automatically).

## 3. One command (stop the app server first with Ctrl+C: training needs the GPU memory)
```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\phase5_gtsdb.ps1
```
It looks for the extracted folder at `Downloads\FullIJCNN2013` (including a nested folder inside it). If you extracted elsewhere add `-Src "C:\your\path"`.
Tip: first run `.\scripts\phase5_gtsdb.ps1 -Epochs 30` to see the whole chain work, then the full 100 epochs.
Steps it runs: convert to YOLO (all 900 images, empty ones kept as negatives), group-level split (no neighbouring-frame leakage), train yolo11n (100 epochs, batch 8, 640), copy weights to `models/`, then latency, robustness and calibration experiments. Use `-Mode full` for all 43 classes (less data per class, expect lower per-class scores). Training on a 4 GB RTX 3050 should fit at batch 8; if you see an out-of-memory error use `-Batch 4`.

## 3b. Already trained? Re-run only the evaluations
```powershell
.\scripts\evaluate_gtsdb.ps1
```

## 4. Pipeline check first (2 minutes, synthetic data, not real signs)
```powershell
python scripts\make_toy_dataset.py --dst datasets\raw\toy --n 90
python -m training.prepare_dataset --src datasets\raw\toy --dst datasets\prepared\toy --names datasets\raw\toy\names.txt --group-regex "^(seq\d+)_"
python -m training.train --data datasets\prepared\toy\data.yaml --model yolo11n.pt --epochs 3 --imgsz 320 --name toy
```

## 5. What you can and cannot claim afterwards
- GTSDB is German signs. Results describe GTSDB's 3 challenge categories, not Indian signs.
- For Indian signs you still need a licensed annotated dataset plus your own photos (see DATASETS.md). Document exactly which classes the model was trained on.
- Report mAP@0.5, mAP@0.5:0.95, precision, recall and the robustness, calibration and latency files exactly as generated. Calibration may fail to improve on small data; report that honestly.
