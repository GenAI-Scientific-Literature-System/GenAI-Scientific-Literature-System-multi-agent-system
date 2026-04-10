#!/usr/bin/env bash
set -e

echo ""
echo "====================================================="
echo "  EPISTEMIC ENGINE — Setup Script (Linux/Mac)"
echo "  Agent 4: Hypothesis Compatibility"
echo "  Agent 5: Epistemic Boundary Analysis"
echo "====================================================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.9+ first."
    exit 1
fi
echo "[OK] Python3 found: $(python3 --version)"

# Create venv
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment exists."
fi

# Activate
source venv/bin/activate

# Install
echo "[SETUP] Installing backend dependencies..."
pip install --quiet -r backend/requirements.txt
echo "[OK] Dependencies installed."

# API Key
echo ""
if [ -z "$MISTRAL_API_KEY" ]; then
    echo "[CONFIG] MISTRAL_API_KEY is not set."
    read -p "Enter your Mistral API key: " MISTRAL_API_KEY
    if [ -z "$MISTRAL_API_KEY" ]; then
        echo "[WARN] No API key provided. Set it before running:"
        echo "       export MISTRAL_API_KEY=your_key_here"
    else
        export MISTRAL_API_KEY
        echo "[OK] API key set for this session."
        echo "     To persist, add to your ~/.bashrc or ~/.zshrc:"
        echo "     export MISTRAL_API_KEY=your_key_here"
    fi
else
    echo "[OK] MISTRAL_API_KEY is set."
fi

echo ""
echo "====================================================="
echo "  Setup complete! Run:  bash run.sh"
echo "====================================================="
echo ""
