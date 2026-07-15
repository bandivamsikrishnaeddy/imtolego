"""
Pydantic Models / Schemas
-------------------------
Defines request bodies, response payloads, and job tracking models.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class LegoModelFormat(str, Enum):
    OBJ = "obj"
    GLB = "glb"
    LDR = "ldr"      # LDraw format (text-based LEGO model)
    MPD = "mpd"      # Multi-part data (LEGO)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for this job.")
    status: JobStatus = Field(default=JobStatus.PENDING)
    message: str = Field(default="Upload accepted. Processing will begin shortly.")


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    input_filename: Optional[str] = None
    progress_pct: int = Field(default=0, ge=0, le=100)
    current_stage: Optional[str] = None
    output_files: dict = Field(default_factory=dict)
    error_message: Optional[str] = None


class ConversionRequest(BaseModel):
    lego_format: LegoModelFormat = Field(default=LegoModelFormat.LDR)
    mc_resolution: Optional[int] = Field(default=None, ge=32, le=512)
    remove_bg: Optional[bool] = None
    foreground_ratio: Optional[float] = Field(default=None, ge=0.5, le=1.0)


class HealthCheck(BaseModel):
    status: str = "ok"
    version: str
    model_loaded: bool
    device: str
