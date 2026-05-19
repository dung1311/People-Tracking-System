"""Database engine and session factory."""

import logging

from sqlmodel import Session, SQLModel, create_engine

from core.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(url=settings.DATABASE_URL)


def init_db():
    """Create all tables (safe to call multiple times)."""
    try:
        # Import all models so their metadata is registered
        import models  # noqa: F401

        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error("Cannot init database: %s", e)
        raise e


def get_session():
    with Session(engine) as session:
        yield session