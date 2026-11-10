# Troubleshooting
- `torch.cuda.is_available()` is False: update the NVIDIA driver, reinstall torch with the cu128 index URL, check `nvidia-smi`.
- Activate.ps1 blocked: `Set-ExecutionPolicy -Scope Process Bypass`.
- Webcam button disabled: another app holds the camera, or no camera; use a video instead.
- Upload fails "could not decode": re-encode to H.264 mp4 (`ffmpeg -i in.mov -c:v libx264 out.mp4`).
- Out of GPU memory: set Inference size 320/416, use yolo11n.
- Splash says backend not reachable: run `.\scripts\run.ps1` in another terminal.
- ByteTrack asks to install `lapx`: allow it or `pip install lapx`.
- Large RAM use: close browser tabs; the app itself stays under a few hundred MB plus the model.
- Smoke test says `+cpu` / `CUDA available: False` even though you have an NVIDIA GPU: you got CPU torch. Usually the Python version is too new for the CUDA wheels. Install Python 3.12 (`winget install Python.Python.3.12`), then re-run `.\scripts\setup.ps1` (it prefers 3.12 and tries cu128, cu126, cu124).
- Stream stays blank / startup very slow: look in the terminal for `warm-up took Xs` and `first inference of this stream took X ms`. Warm-up should be a few seconds. If the GPU path misbehaves, set `model.fp16: false` in `configs/default.yaml` and restart.
- Multi-second stall the first time an input shape is used: that was FP16 (measured, docs/PERFORMANCE.md). FP32 is now the default; leave `model.fp16: false`.
- Webcam opens your phone / 'Connecting to <phone>' / spinner: Windows exposes the phone as camera 0 (Phone Link 'Connected camera'). The Webcam button tries other cameras automatically; if it still picks the phone, disconnect the phone or force a camera in SanketraData\config.local.yaml (camera: index: 1).

## Windows becomes slow, hangs, or will not delete a folder
Re-creating a Python virtual environment (tens of thousands of files) inside an indexed folder such as Downloads can trigger Windows Search indexing and real-time antivirus scanning, which may freeze the PC or lock files. The cause was not confirmed by measurement; these habits avoid it:
- Keep one project folder and update it by copying changed files over it. `setup.ps1` keeps a working `.venv` (`-Rebuild` forces a new one).
- Keep the project and large datasets outside Downloads/Desktop/Documents/OneDrive where possible. If they must stay there, add the project folder and your data folder to the Windows Search and antivirus exclusion lists.
- Extract big zips with a short destination path (for example `tar -xf file.zip -C C:\ind`) and delete big folders with `rd /s /q "path"` instead of Explorer.
- Before training or after any hiccup run `.\scripts\doctor.ps1` (read-only): it checks the project location, data folder, disk space, the Python environment, GPU memory, port 8000 and stray python processes.
- Never start two heavy jobs at once (training, setup, dataset extraction). Running the app while training also slows both.
- Unit tests never open a real camera (`tests/conftest.py`); a hung camera driver can otherwise stall a test run for minutes.
