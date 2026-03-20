#!/bin/bash
# MERLIN Run — Linux / macOS

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
fail() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

echo ""
echo "  ============================================="
echo "   MERLIN  —  Starting API Server"
echo "  ============================================="
echo ""

# ── venv check ────────────────────────────────────────────────────────────
if [ ! -f "venv/bin/activate" ]; then
    fail "Virtual environment not found. Run:  bash setup.sh"
fi

source venv/bin/activate
ok "Virtual environment active."

# ── .env check ────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    fail ".env not found. Run:  bash setup.sh"
fi

if grep -q "your-mistral-api-key-here" .env; then
    warn "MISTRAL_API_KEY is still the placeholder in .env"
    warn "Analysis calls will fail. Get your key: https://console.mistral.ai"
    echo ""
fi

# ── Dependency spot-checks ────────────────────────────────────────────────
info "Checking key packages..."

python3 -c "import flask; print('[OK] Flask ' + flask.__version__)" \
    || fail "Flask not found. Run: bash setup.sh"

python3 -c "import mistralai; print('[OK] mistralai installed')" \
    || warn "mistralai not found. Run: pip install mistralai"

python3 -c "import networkx; print('[OK] networkx ' + networkx.__version__)" \
    || warn "networkx not found. Run: pip install networkx"

python3 -c "import flask_cors; print('[OK] flask-cors installed')" \
    || warn "flask-cors not found. Run: pip install flask-cors"

echo ""
echo "  ============================================="
echo -e "   ${BOLD}Server starting on http://localhost:5000${NC}"
echo "   Press Ctrl+C to stop"
echo "  ============================================="
echo ""

# ── Launch ────────────────────────────────────────────────────────────────
python3 -m flask --app api/server.py run --host=0.0.0.0 --port=5000
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Server stopped with exit code $EXIT_CODE"
    echo ""
    echo "  Common causes:"
    echo "    Port 5000 in use?   Change API_PORT in config.py"
    echo "    Missing package?    Run bash setup.sh again"
    echo "    Crash / traceback?  Scroll up to read the error"
fi
