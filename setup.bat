@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
title MERLIN Setup

echo.
echo  =============================================
echo   MERLIN Setup  --  Windows Installer
echo   Epistemic Reasoning over Scientific Literature
echo  =============================================
echo.

:: ── Step 1: Check Python ──────────────────────────────────────────────────
echo [STEP 1/5] Checking Python...
python --version
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Python not found in PATH.
    echo         Download and install Python 3.10+ from https://python.org
    echo         Make sure to tick "Add Python to PATH" during install.
    echo.
    pause & exit /b 1
)
echo [OK] Python found.
echo.

:: ── Step 2: Create virtual environment ───────────────────────────────────
echo [STEP 2/5] Creating virtual environment...
IF EXIST venv (
    echo [INFO] venv already exists, skipping creation.
) ELSE (
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Could not create virtual environment.
        echo         Try:  python -m pip install virtualenv
        echo.
        pause & exit /b 1
    )
    echo [OK] Virtual environment created at .\venv\
)
echo.

:: ── Step 3: Activate venv ────────────────────────────────────────────────
echo [STEP 3/5] Activating virtual environment...
call venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not activate venv. Check your antivirus / execution policy.
    pause & exit /b 1
)
echo [OK] Virtual environment active.
echo.

:: ── Step 4: Upgrade pip (verbose) ────────────────────────────────────────
echo [STEP 4/5] Upgrading pip...
python -m pip install --upgrade pip
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] pip upgrade failed -- continuing anyway.
)
echo.

:: ── Step 5: Install dependencies (verbose) ───────────────────────────────
echo [STEP 5/5] Installing dependencies from requirements.txt...
echo            This may take 3-10 minutes on first run.
echo            You will see each package install below:
echo.
pip install -r requirements.txt --no-cache-dir
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] One or more packages failed to install.
    echo.
    echo  Common fixes:
    echo    - No internet?  Check your connection.
    echo    - Firewall/proxy?  Set:  set HTTPS_PROXY=http://yourproxy:port
    echo    - torch too large?  Edit requirements.txt and replace torch with:
    echo        torch==2.4.0+cpu  --index-url https://download.pytorch.org/whl/cpu
    echo    - Still failing?  Run:  pip install -r requirements.txt -v
    echo      and share the last 20 lines of output.
    echo.
    pause & exit /b 1
)
echo.
echo [OK] All dependencies installed successfully.
echo.

:: ── .env setup ───────────────────────────────────────────────────────────
IF NOT EXIST .env (
    copy .env.example .env >nul
    echo [INFO] Created .env from .env.example
    echo.
    echo  *** ACTION REQUIRED ***
    echo  Open .env in Notepad and replace:
    echo    MISTRAL_API_KEY=your-mistral-api-key-here
    echo  with your real key from https://console.mistral.ai
    echo.
) ELSE (
    echo [OK] .env file already exists.
    findstr /C:"your-mistral-api-key-here" .env >nul 2>&1
    IF %ERRORLEVEL% EQU 0 (
        echo.
        echo  [WARNING] MISTRAL_API_KEY is still the placeholder value in .env
        echo            Edit .env and add your real key before running!
        echo.
    )
)

echo  =============================================
echo   Setup complete!
echo.
echo   Next steps:
echo     1. Edit .env  -- add your MISTRAL_API_KEY
echo     2. Run run.bat to start the server
echo     3. Open http://localhost:5000 in your browser
echo  =============================================
echo.
pause
