"""Authentication endpoints: login, register, me."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from api.v1.deps import get_current_user, require_admin
from core.security import create_access_token, hash_password, verify_password
from database.session import get_session
from models.user import User, UserRole
from schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from schemas.user import UserRead

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == body.username)).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return TokenResponse(access_token=token)


@router.post("/register", response_model=UserRead, status_code=201)
def register(
    body: RegisterRequest,
    session: Session = Depends(get_session),
    admin: User = Depends(require_admin),
):
    """Register a new user. Admin only."""
    # Check duplicates
    existing = session.exec(
        select(User).where(
            (User.username == body.username) | (User.email == body.email)
        )
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already exists",
        )
    # Validate role
    try:
        role = UserRole(body.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {body.role}. Must be one of: admin, operator, viewer",
        )
    user = User(
        username=body.username,
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        role=role,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.get("/me", response_model=UserRead)
def get_me(user: User = Depends(get_current_user)):
    return user
