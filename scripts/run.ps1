# Starts API + built UI at http://127.0.0.1:8000   (stop with Ctrl+C)
. "$PSScriptRoot\_env.ps1"
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
