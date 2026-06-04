"""
run_missing_tables.py
---------------------
Creates any tables that are missing from your database but referenced
by the dashboard. Safe to run multiple times — uses CREATE TABLE IF NOT EXISTS.

Run from your project root:
    python run_missing_tables.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from database.connection import get_db_engine

TABLES = {

    # ── optimization_results ─────────────────────────────────────────────────
    # Columns match what save_optimization_result() actually inserts.
    "optimization_results": """
        CREATE TABLE IF NOT EXISTS optimization_results (
            id               SERIAL PRIMARY KEY,
            run_date         TIMESTAMP DEFAULT NOW(),
            method           VARCHAR(50),
            tickers          JSONB,
            weights          JSONB,
            current_weights  JSONB,
            expected_return  FLOAT,
            expected_vol     FLOAT,
            sharpe_ratio     FLOAT,
            improvement      FLOAT,
            constraints      JSONB,
            created_at       TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_optim_method
            ON optimization_results (method, run_date DESC);
    """,

    # ── liquidity_risk_results ────────────────────────────────────────────────
    "liquidity_risk_results": """
        CREATE TABLE IF NOT EXISTS liquidity_risk_results (
            id               SERIAL PRIMARY KEY,
            run_date         TIMESTAMP DEFAULT NOW(),
            portfolio_id     VARCHAR(100),
            tickers          JSONB,
            weights          JSONB,
            adv_scores       JSONB,
            liquidation_days JSONB,
            amihud_scores    JSONB,
            la_var_95        FLOAT,
            la_var_99        FLOAT,
            hhi_index        FLOAT,
            effective_n      FLOAT,
            alerts           JSONB,
            created_at       TIMESTAMP DEFAULT NOW()
        );
    """,

    # ── factor_risk_results ───────────────────────────────────────────────────
    "factor_risk_results": """
        CREATE TABLE IF NOT EXISTS factor_risk_results (
            id               SERIAL PRIMARY KEY,
            run_date         TIMESTAMP DEFAULT NOW(),
            portfolio_id     VARCHAR(100),
            factor_betas     JSONB,
            r_squared        FLOAT,
            alpha_annual     FLOAT,
            risk_pct         JSONB,
            return_contrib   JSONB,
            created_at       TIMESTAMP DEFAULT NOW()
        );
    """,

    # ── monte_carlo_results ───────────────────────────────────────────────────
    "monte_carlo_results": """
        CREATE TABLE IF NOT EXISTS monte_carlo_results (
            id               SERIAL PRIMARY KEY,
            run_date         TIMESTAMP DEFAULT NOW(),
            portfolio_id     VARCHAR(100),
            n_simulations    INT,
            holding_period   INT,
            var_95           FLOAT,
            cvar_95          FLOAT,
            var_99           FLOAT,
            cvar_99          FLOAT,
            created_at       TIMESTAMP DEFAULT NOW()
        );
    """,

    # ── var_results ───────────────────────────────────────────────────────────
    "var_results": """
        CREATE TABLE IF NOT EXISTS var_results (
            id               SERIAL PRIMARY KEY,
            run_date         TIMESTAMP DEFAULT NOW(),
            portfolio_id     VARCHAR(100),
            method           VARCHAR(50),
            confidence_level FLOAT,
            var_value        FLOAT,
            cvar_value       FLOAT,
            holding_period   INT,
            created_at       TIMESTAMP DEFAULT NOW()
        );
    """,

    # ── stress_test_results ───────────────────────────────────────────────────
    "stress_test_results": """
        CREATE TABLE IF NOT EXISTS stress_test_results (
            id               SERIAL PRIMARY KEY,
            run_date         TIMESTAMP DEFAULT NOW(),
            portfolio_id     VARCHAR(100),
            scenario         VARCHAR(100),
            market_shock     FLOAT,
            portfolio_loss   FLOAT,
            var_95           FLOAT,
            cvar_99          FLOAT,
            created_at       TIMESTAMP DEFAULT NOW()
        );
    """,
}


def main():
    engine = get_db_engine()
    created = []
    skipped = []

    with engine.begin() as conn:
        for table_name, ddl in TABLES.items():
            # Check if table already exists
            exists = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = :t)"
            ), {"t": table_name}).scalar()

            conn.execute(text(ddl))

            if exists:
                skipped.append(table_name)
                print(f"  ✓ {table_name} — already exists, skipped")
            else:
                created.append(table_name)
                print(f"  ✅ {table_name} — CREATED")

    print()
    print(f"Done. Created: {len(created)}, Already existed: {len(skipped)}")
    if created:
        print(f"New tables: {', '.join(created)}")


if __name__ == "__main__":
    main()
