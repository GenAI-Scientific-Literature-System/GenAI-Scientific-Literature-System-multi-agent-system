@echo off
SETLOCAL

echo.
echo =====================================================
echo   EPISTEMIC ENGINE — Starting Server (Windows)
echo =====================================================
echo.

REM ─── Activate virtual environment ───
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

REM ─── Check API key ───
IF "%MISTRAL_API_KEY%"=="" (
    echo [WARN] MISTRAL_API_KEY is not set.
    set /p MISTRAL_API_KEY="Enter your Mistral API key (or press Enter to skip): "
)

REM ─── Start server ───
echo [START] Launching FastAPI server on http://localhost:8000
echo         Press Ctrl+C to stop.
echo.
echo  Open browser: http://localhost:8000
echo  API Docs:     http://localhost:8000/docs
echo  Health:       http://localhost:8000/health
echo.

cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

ENDLOCAL
