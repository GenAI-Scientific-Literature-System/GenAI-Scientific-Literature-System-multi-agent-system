@echo off
SETLOCAL

echo.
echo =====================================================
echo   EPISTEMIC ENGINE — Setup Script (Windows)
echo   Agent 4: Hypothesis Compatibility
echo   Agent 5: Epistemic Boundary Analysis
echo =====================================================
echo.

REM ─── Check Python ───
python --version >NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found.

REM ─── Check pip ───
pip --version >NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip is not available. Please install pip.
    pause
    exit /b 1
)
echo [OK] pip found.

REM ─── Create virtual environment ───
IF NOT EXIST "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) ELSE (
    echo [OK] Virtual environment already exists.
)

REM ─── Activate venv ───
call venv\Scripts\activate.bat

REM ─── Install dependencies ───
echo [SETUP] Installing backend dependencies...
pip install -r backend\requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.

REM ─── Set API Key ───
echo.
IF "%MISTRAL_API_KEY%"=="" (
    echo [CONFIG] MISTRAL_API_KEY is not set.
    set /p MISTRAL_API_KEY="Enter your Mistral API key: "
    IF "%MISTRAL_API_KEY%"=="" (
        echo [WARN] No API key provided. You can set it later:
        echo        set MISTRAL_API_KEY=your_key_here
    ) ELSE (
        echo [OK] API key set for this session.
    )
) ELSE (
    echo [OK] MISTRAL_API_KEY is already set.
)

echo.
echo =====================================================
echo   Setup complete! Run:  run.bat
echo =====================================================
echo.
pause
ENDLOCAL
