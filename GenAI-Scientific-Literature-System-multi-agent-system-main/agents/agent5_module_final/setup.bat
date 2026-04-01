@echo off
echo ============================================================
echo   AGENT 5 — Uncertainty ^& Research Gap Analyst
echo   Setup Script
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found.
echo.

REM Create virtual environment
if not exist "venv" (
    echo [Setup] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

REM Activate venv
echo [Setup] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [Setup] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo [Setup] Installing dependencies...
pip install streamlit plotly pandas --quiet

echo.
echo [Setup] Creating data directories...
if not exist "data\output" mkdir data\output

echo.
echo [OK] Setup complete. Place your .txt paper files in data\ and run run.bat
