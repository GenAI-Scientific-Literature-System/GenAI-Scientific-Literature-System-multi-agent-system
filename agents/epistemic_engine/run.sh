#!/usr/bin/env bash
set -e

echo ""
echo "====================================================="
echo "  EPISTEMIC ENGINE — Starting Server (Linux/Mac)"
echo "====================================================="
echo ""

if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found. Run setup.sh first."
    exit 1
fi

source venv/bin/activate

if [ -z "$MISTRAL_API_KEY" ]; then
    echo "[WARN] MISTRAL_API_KEY is not set."
    read -p "Enter your Mistral API key (or Enter to skip): " KEY
    if [ -n "$KEY" ]; then
        export MISTRAL_API_KEY="$KEY"
    fi
fi

echo "[START] Launching FastAPI on http://localhost:8000"
echo "        Press Ctrl+C to stop."
echo ""
echo "  Browser:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Health:   http://localhost:8000/health"
echo ""

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
