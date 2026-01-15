from models.camera import CameraBase

class CameraCreate(CameraBase):
    pass

class CameraRead(CameraBase):
    id: int
    
class CameraUpdate(CameraBase):
    name: str | None = None
    source: str | None = None
    description: str | None = None
    is_active: bool | None = None
    config_path: str | None = None
