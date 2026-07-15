"""
Enterprise Configuration Module
-------------------------------
Manages all application settings via Pydantic Settings.
Reads values from environment variables (12-factor app compliant).
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure dirs exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class AppSettings(BaseSettings):
    """Application configuration."""

    # FastAPI
    app_name: str = Field(default="Lego3D API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    uvicorn_host: str = Field(default="0.0.0.0", alias="UVICORN_HOST")
    uvicorn_port: int = Field(default=8000, alias="UVICORN_PORT")
    uvicorn_workers: int = Field(default=1, alias="UVICORN_WORKERS")

    # Storage
    uploads_dir: Path = Field(default=UPLOADS_DIR, alias="UPLOADS_DIR")
    outputs_dir: Path = Field(default=OUTPUTS_DIR, alias="OUTPUTS_DIR")
    max_upload_size_mb: int = Field(default=20, alias="MAX_UPLOAD_SIZE_MB")

    # ML Model
    model_id: str = Field(default="stabilityai/TripoSR", alias="MODEL_ID")
    model_config_name: str = Field(default="config.yaml", alias="MODEL_CONFIG_NAME")
    model_weight_name: str = Field(default="model.ckpt", alias="MODEL_WEIGHT_NAME")
    device: str = Field(default="cuda:0", alias="DEVICE")
    chunk_size: int = Field(default=8192, alias="CHUNK_SIZE")
    mc_resolution: int = Field(default=256, alias="MC_RESOLUTION")
    remove_bg: bool = Field(default=True, alias="REMOVE_BG")
    foreground_ratio: float = Field(default=0.85, alias="FOREGROUND_RATIO")

    # LEGO Conversion
    lego_target_max_studs: int = Field(default=40, alias="LEGO_TARGET_MAX_STUDS")
    lego_hollow: bool = Field(default=False, alias="LEGO_HOLLOW")

    # Redis / Celery
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    celery_broker_url: str = Field(default="redis://localhost:6379/0", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
