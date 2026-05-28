"""
database/connection.py
----------------------
SQLAlchemy engine, session factory, and database initialization utilities.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from loguru import logger
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_URL, LOG_FILE, LOG_LEVEL, LOG_ROTATION

# ─── Logging Setup ───────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(LOG_FILE, rotation=LOG_ROTATION, level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")

# ─── SQLAlchemy Setup ─────────────────────────────────────────────────────────
Base = declarative_base()

def get_engine(echo: bool = False):
    """Create and return a SQLAlchemy engine with connection pooling."""
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=echo,
    )
    return engine


def get_session_factory(engine=None):
    """Return a sessionmaker bound to the given (or default) engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_session(engine=None):
    """Context manager that yields a database session and handles commit/rollback."""
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"Session error — rolled back: {exc}")
        raise
    finally:
        session.close()


def init_database(engine=None):
    """
    Create all tables defined in models.py if they don't already exist.
    Also creates the database schema migrations.
    """
    if engine is None:
        engine = get_engine()

    from database.models import Base as ModelBase
    try:
        ModelBase.metadata.create_all(engine)
        logger.info("✅ Database tables created / verified successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialise database: {e}")
        raise


def test_connection() -> bool:
    """Ping the database and return True if reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful.")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


# Module-level singletons (lazy initialisation)
_engine = None
_SessionLocal = None


def get_db_engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def get_db_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = get_session_factory(get_db_engine())
    return _SessionLocal()
