import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session, select

from database.session import get_session
from models.tracking_config import TrackingConfig
from models.user import UserRole
from schemas.config import TrackingConfigCreate, TrackingConfigRead, TrackingConfigUpdate
from api.v1.deps import RoleChecker, get_current_active_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Role checkers
admin_only = Depends(RoleChecker([UserRole.ADMIN]))
operator_or_admin = Depends(RoleChecker([UserRole.ADMIN, UserRole.OPERATOR]))
any_user = Depends(get_current_active_user)

@router.post("/", response_model=TrackingConfigRead)
def create_config(
    *,
    session: Session = Depends(get_session),
    config: TrackingConfigCreate,
    current_user=operator_or_admin
):
    # If setting is_default=True, reset other defaults of same type
    if config.is_default:
        session.exec(
            f"UPDATE trackingconfig SET is_default = False WHERE config_type = '{config.config_type}'"
        )
    
    db_config = TrackingConfig.from_orm(config)
    session.add(db_config)
    session.commit()
    session.refresh(db_config)
    return db_config

@router.get("/", response_model=List[TrackingConfigRead])
def read_configs(
    *,
    session: Session = Depends(get_session),
    config_type: Optional[str] = Query(default=None, description="Filter by 'sct' or 'mct'"),
    current_user=any_user
):
    query = select(TrackingConfig)
    if config_type:
        query = query.where(TrackingConfig.config_type == config_type)
    configs = session.exec(query).all()
    return configs

@router.get("/defaults", response_model=List[TrackingConfigRead])
def read_default_configs(
    *,
    session: Session = Depends(get_session),
    current_user=any_user
):
    configs = session.exec(
        select(TrackingConfig).where(TrackingConfig.is_default == True)
    ).all()
    return configs

@router.get("/{config_id}", response_model=TrackingConfigRead)
def read_config(
    *,
    session: Session = Depends(get_session),
    config_id: int,
    current_user=any_user
):
    config = session.get(TrackingConfig, config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config profile not found")
    return config

@router.patch("/{config_id}", response_model=TrackingConfigRead)
def update_config(
    *,
    session: Session = Depends(get_session),
    config_id: int,
    config: TrackingConfigUpdate,
    current_user=operator_or_admin
):
    db_config = session.get(TrackingConfig, config_id)
    if not db_config:
        raise HTTPException(status_code=404, detail="Config profile not found")
    
    config_data = config.dict(exclude_unset=True)
    
    if config_data.get("is_default", False):
        session.exec(
            f"UPDATE trackingconfig SET is_default = False WHERE config_type = '{db_config.config_type}' AND id != {config_id}"
        )
        
    for key, value in config_data.items():
        setattr(db_config, key, value)
        
    db_config.updated_at = datetime.utcnow()
    session.add(db_config)
    session.commit()
    session.refresh(db_config)
    return db_config

@router.delete("/{config_id}")
def delete_config(
    *,
    session: Session = Depends(get_session),
    config_id: int,
    current_user=admin_only
):
    config = session.get(TrackingConfig, config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config profile not found")
        
    if config.is_default:
        raise HTTPException(
            status_code=400, 
            detail="Cannot delete a default configuration profile. Assign another default profile first."
        )
        
    session.delete(config)
    session.commit()
    return {"ok": True}

@router.post("/validate")
def validate_config(
    config_data: dict,
    current_user=operator_or_admin
):
    # Simply check if the dictionary contains valid tracking configuration structure
    # For SCT, it should have DETECTOR and TRACKER
    # For MCT, it should have MATCHING, GLOBAL_TRACK
    issues = []
    
    # Generic validation
    if not config_data:
        return {"valid": False, "errors": ["Configuration data is empty"]}
        
    # Standard format validation
    is_sct = "DETECTOR" in config_data or "TRACKER" in config_data
    is_mct = "MATCHING" in config_data or "GLOBAL_TRACK" in config_data
    
    if not is_sct and not is_mct:
        issues.append("Config does not appear to be a valid SCT (requires DETECTOR/TRACKER) or MCT (requires MATCHING/GLOBAL_TRACK) format.")
        
    return {
        "valid": len(issues) == 0,
        "errors": issues
    }
