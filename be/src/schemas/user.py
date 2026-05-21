from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from models.user import UserRole

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[UserRole] = UserRole.VIEWER

class UserRead(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
