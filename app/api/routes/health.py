from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter()

@router.get("/health")
def health():
    settings = get_settings()
    return {"status": "ok", "version": settings.app_version}
