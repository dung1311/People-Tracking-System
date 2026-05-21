from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select

from database.session import get_session
from core.security import create_access_token, verify_password, get_password_hash
from models.user import User, UserRole
from schemas.auth import Token
from schemas.user import UserCreate, UserRead
from api.v1.deps import get_current_active_user, RoleChecker

router = APIRouter()

@router.post("/login", response_model=Token)
def login(
    db: Session = Depends(get_session),
    form_data: OAuth2PasswordRequestForm = Depends()
):
    user = db.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password",
        )
    elif not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    
    access_token = create_access_token(subject=user.username)
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.post("/register", response_model=UserRead)
def register(
    user_in: UserCreate,
    db: Session = Depends(get_session),
    current_user: User = Depends(RoleChecker([UserRole.ADMIN]))
):
    existing_user = db.exec(select(User).where(
        (User.username == user_in.username) | (User.email == user_in.email)
    )).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered",
        )
    
    db_user = User(
        username=user_in.username,
        email=user_in.email,
        role=user_in.role,
        hashed_password=get_password_hash(user_in.password),
        is_active=True
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/me", response_model=UserRead)
def get_me(current_user: User = Depends(get_current_active_user)):
    return current_user
