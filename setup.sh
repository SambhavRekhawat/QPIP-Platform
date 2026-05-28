#!/usr/bin/env bash
# =============================================================
# setup.sh  —  Bootstrap the Quantitative Portfolio Platform
# Run once after cloning the project:  bash setup.sh
# =============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Quantitative Portfolio Intelligence Platform        ║"
echo "║  Setup Script                                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Python version check ───────────────────────────────────
info "Checking Python version…"
PYTHON=$(command -v python3 || command -v python || error "Python not found")
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
info "Using Python $PY_VER at $PYTHON"

# ── 2. Create virtual environment ────────────────────────────
if [ ! -d "venv" ]; then
    info "Creating virtual environment…"
    $PYTHON -m venv venv
    success "Virtual environment created in ./venv"
else
    info "Virtual environment already exists, skipping."
fi

# ── 3. Activate venv ─────────────────────────────────────────
source venv/bin/activate
success "Virtual environment activated."

# ── 4. Upgrade pip & install dependencies ────────────────────
info "Installing Python dependencies (this takes ~2–3 minutes)…"
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
success "All dependencies installed."

# ── 5. Create .env if missing ────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env file created from .env.example"
    warn "👉  Open .env and fill in DB_PASSWORD (and NEWS_API_KEY if you have one)."
else
    info ".env already exists, skipping."
fi

# ── 6. Create directories ─────────────────────────────────────
mkdir -p logs exports ml_engine/saved_models
success "Log and export directories ready."

# ── 7. PostgreSQL check ──────────────────────────────────────
info "Checking PostgreSQL…"
if command -v psql &>/dev/null; then
    success "psql found: $(psql --version)"
else
    warn "psql not found in PATH. Make sure PostgreSQL is running."
    warn "Install guide: https://www.postgresql.org/download/"
fi

echo ""
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Edit .env and set your DB_PASSWORD"
echo "     ${YELLOW}nano .env${NC}"
echo ""
echo "  2. Create the PostgreSQL database:"
echo "     ${YELLOW}psql -U postgres -c \"CREATE DATABASE quant_platform;\"${NC}"
echo ""
echo "  3. Initialise database tables:"
echo "     ${YELLOW}python main.py --setup-db${NC}"
echo ""
echo "  4. Run the full analysis pipeline:"
echo "     ${YELLOW}python main.py --tickers RELIANCE TCS HDFCBANK INFY ICICIBANK${NC}"
echo ""
echo "  5. Launch the dashboard:"
echo "     ${YELLOW}streamlit run dashboard/app.py${NC}"
echo ""
echo -e "${CYAN}Dashboard will open at:  http://localhost:8501${NC}"
echo ""
