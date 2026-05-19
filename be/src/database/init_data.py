"""Seed default admin user and default configs on first startup."""

import logging

from sqlmodel import Session, select

from core.config import settings
from core.security import hash_password
from models.user import User, UserRole
from models.tracking_config import TrackingConfig

logger = logging.getLogger(__name__)


def seed_default_data(engine):
    """Create default admin and configs if they don't exist."""
    with Session(engine) as session:
        _seed_admin(session)
        _seed_default_configs(session)


def _seed_admin(session: Session):
    existing = session.exec(
        select(User).where(User.username == settings.DEFAULT_ADMIN_USERNAME)
    ).first()
    if existing:
        return

    admin = User(
        username=settings.DEFAULT_ADMIN_USERNAME,
        email=settings.DEFAULT_ADMIN_EMAIL,
        full_name="System Administrator",
        hashed_password=hash_password(settings.DEFAULT_ADMIN_PASSWORD),
        role=UserRole.ADMIN,
    )
    session.add(admin)
    session.commit()
    logger.info("Created default admin user: %s", admin.username)


def _seed_default_configs(session: Session):
    from api.v1.endpoints.configs import DEFAULT_SCT_CONFIG, DEFAULT_MCT_CONFIG

    # SCT default
    existing_sct = session.exec(
        select(TrackingConfig).where(
            TrackingConfig.config_type == "sct",
            TrackingConfig.is_default == True,
        )
    ).first()
    if not existing_sct:
        sct = TrackingConfig(
            name="Default SCT Config",
            description="Built-in single camera tracking config",
            config_type="sct",
            config_data=DEFAULT_SCT_CONFIG,
            is_default=True,
        )
        session.add(sct)
        logger.info("Created default SCT config")

    # MCT default
    existing_mct = session.exec(
        select(TrackingConfig).where(
            TrackingConfig.config_type == "mct",
            TrackingConfig.is_default == True,
        )
    ).first()
    if not existing_mct:
        mct = TrackingConfig(
            name="Default MCT Config",
            description="Built-in multi-camera tracking config",
            config_type="mct",
            config_data=DEFAULT_MCT_CONFIG,
            is_default=True,
        )
        session.add(mct)
        logger.info("Created default MCT config")

    session.commit()
