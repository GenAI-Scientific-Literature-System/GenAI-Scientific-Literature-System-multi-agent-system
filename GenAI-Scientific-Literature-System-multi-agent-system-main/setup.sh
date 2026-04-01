#!/bin/bash
# MERLIN Setup — Linux / macOS
# Verbose: every step prints what it's doing and whether it succeeded.

set -euo pipefail   # exit on any error, undefined var, or pipe failure

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

step()    { echo -e "\n${BOLD}${CYAN}[STEP $1]${NC} $2"; }
ok()      { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $1"; }
fail()    { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
info()    { echo -e "       $1"; }

echo ""
echo "  ============================================="
echo "   MERLIN Setup  —  Linux / macOS"
echo "   Epistemic Reasoning over Scientific Literature"
echo "  ============================================="
echo ""

# ── Step 1: Python check ──────────────────────────────────────────────────
step "1/5" "Checking Python version..."
if ! command -v python3 &>/dev/null; then
    fail "python3 not found. Install via: sudo apt install python3  (or brew install python)"
fi
PY_VER=$(python3 --version)
echo "       Found: $PY_VER"
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    fail "Python 3.10+ required. You have $PY_VER"
fi
ok "$PY_VER"

# ── Step 2: Virtual environment ───────────────────────────────────────────
step "2/5" "Creating virtual environment..."
if [ -d "venv" ]; then
    info "venv/ already exists — skipping creation."
    ok "Virtual environment exists."
else
    python3 -m venv venv
    ok "Virtual environment created at ./venv/"
fi

source venv/bin/activate
ok "Virtual environment activated."

# ── Step 3: Upgrade pip ───────────────────────────────────────────────────
step "3/5" "Upgrading pip..."
pip install --upgrade pip || warn "pip upgrade failed — continuing."
ok "pip up to date: $(pip --version)"

# ── Step 4: Install dependencies (verbose) ────────────────────────────────
step "4/5" "Installing dependencies from requirements.txt..."
info "This may take 3-10 minutes on first run."
info "You will see each package install below:\n"

if ! pip install -r requirements.txt --no-cache-dir; then
    echo ""
    fail "$(cat <<'ERRMSG'
One or more packages failed to install.

Common fixes:
  torch too large / slow?
    Edit requirements.txt, replace:
      torch==2.4.0
    with:
      torch==2.4.0+cpu  --index-url https://download.pytorch.org/whl/cpu

  Behind a proxy?
    export HTTPS_PROXY=http://yourproxy:port

  Still stuck? Run with more detail:
    pip install -r requirements.txt -v 2>&1 | tee install_log.txt
  Then check install_log.txt for the specific failure.
ERRMSG
)"
fi
ok "All dependencies installed."

# ── Step 5: .env setup ────────────────────────────────────────────────────
step "5/5" "Configuring environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    ok "Created .env from .env.example"
    echo ""
    echo -e "  ${YELLOW}*** ACTION REQUIRED ***${NC}"
    echo "  Open .env in a text editor and replace:"
    echo "    MISTRAL_API_KEY=your-mistral-api-key-here"
    echo "  with your real key from https://console.mistral.ai"
else
    ok ".env already exists."
    if grep -q "your-mistral-api-key-here" .env; then
        warn "MISTRAL_API_KEY is still the placeholder in .env — update it before running!"
    fi
fi

echo ""
echo "  ============================================="
echo "   Setup complete!"
echo ""
echo "   Next steps:"
echo "     1.  nano .env          — add your MISTRAL_API_KEY"
echo "     2.  bash run.sh        — start the server"
echo "     3.  Open http://localhost:5000 in your browser"
echo "  ============================================="
echo ""
