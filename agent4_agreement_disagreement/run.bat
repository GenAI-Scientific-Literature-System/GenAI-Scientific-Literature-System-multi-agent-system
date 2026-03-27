@echo off
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [WARN] No venv found. Run setup.bat first.
)

echo [Agent 4] Starting server at http://localhost:5004
start "" http://localhost:5004
python server.py
