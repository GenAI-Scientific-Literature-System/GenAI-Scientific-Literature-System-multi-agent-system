@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
title MERLIN Server

echo.
echo  =============================================
echo   MERLIN  --  Starting API Server
echo  =============================================
echo.

:: ── Check venv exists ────────────────────────────────────────────────────
IF NOT EXIST venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment not found.
    echo         Run setup.bat first to install dependencies.
    echo.
    pause & exit /b 1
)

:: ── Activate venv ────────────────────────────────────────────────────────
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not activate venv.
    pause & exit /b 1
)
echo [OK] Virtual environment active.

:: ── Check .env ───────────────────────────────────────────────────────────
IF NOT EXIST .env (
    echo [ERROR] .env file not found. Run setup.bat first.
    pause & exit /b 1
)

:: ── Check API key is not still placeholder ────────────────────────────────
findstr /C:"your-mistral-api-key-here" .env >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo.
    echo  [WARNING] MISTRAL_API_KEY in .env is still the placeholder value.
    echo            Analysis calls will fail without a real key.
    echo            Get yours free at: https://console.mistral.ai
    echo.
)

:: ── Verify Flask is installed ─────────────────────────────────────────────
echo [INFO] Checking Flask installation...
python -c "import flask; print('[OK] Flask', flask.__version__)" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Flask not found. Run setup.bat again.
    pause & exit /b 1
)

:: ── Verify Mistral client is installed ───────────────────────────────────
echo [INFO] Checking mistralai installation...
python -c "import mistralai; print('[OK] mistralai installed')" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] mistralai not found. Run:  pip install mistralai
)

:: ── Start server ─────────────────────────────────────────────────────────
echo.
echo  =============================================
echo   Server starting on http://localhost:5000
echo   Press Ctrl+C to stop
echo  =============================================
echo.

python -m flask --app api/server.py run --host=0.0.0.0 --port=5000

:: ── If server exits ──────────────────────────────────────────────────────
echo.
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Server stopped with an error (exit code %ERRORLEVEL%).
    echo.
    echo  Common causes:
    echo    - Port 5000 already in use?  Change API_PORT in config.py
    echo    - Missing package?  Run setup.bat again.
    echo    - Syntax error?  Check the output above for traceback.
)
echo.
pause
