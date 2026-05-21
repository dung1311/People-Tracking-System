from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from database.session import get_session
from models.user import User, UserRole
from schemas.user import UserRead, UserUpdate
from api.v1.deps import RoleChecker
from core.security import get_password_hash

router = APIRouter()

# Protect all routes in this router for ADMIN only
admin_dependency = Depends(RoleChecker([UserRole.ADMIN]))

@router.get("/", response_model=List[UserRead], dependencies=[admin_dependency])
def read_users(db: Session = Depends(get_session)):
    users = db.exec(select(User)).all()
    return users

@router.patch("/{user_id}", response_model=UserRead, dependencies=[admin_dependency])
def update_user(
    user_id: int,
    user_in: UserUpdate,
    db: Session = Depends(get_session)
):
    db_user = db.get(User, user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    update_data = user_in.dict(exclude_unset=True)
    if "password" in update_data:
        hashed_password = get_password_hash(update_data["password"])
        db_user.hashed_password = hashed_password
        del update_data["password"]
        
    for key, value in update_data.items():
        setattr(db_user, key, value)
        
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.delete("/{user_id}", dependencies=[admin_dependency])
def delete_user(
    user_id: int,
    db: Session = Depends(get_session)
):
    db_user = db.get(User, user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    db.delete(db_user)
    db.commit()
    return {"ok": True}
