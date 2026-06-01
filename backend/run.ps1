$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    py -m venv .venv
}

.\.venv\Scripts\pip.exe install -q -r requirements.txt
.\.venv\Scripts\python.exe scripts\train.py
.\.venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8010
