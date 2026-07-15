"""
FastAPI Application
-------------------
Provides enterprise-grade REST endpoints for image upload, job tracking,
and asset download.
"""

import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.models.schemas import (
    ConversionRequest,
    HealthCheck,
    JobState,
    JobStatus,
    UploadResponse,
)
from app.pipeline.orchestrator import LegoPipeline

# --------------------------------------------------------------------------- #
# Logging & Settings
# --------------------------------------------------------------------------- #
settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger()

# --------------------------------------------------------------------------- #
# Job Store (simple in-memory; replace with Redis in production)
# --------------------------------------------------------------------------- #
_jobs: Dict[str, JobState] = {}
_executor = ThreadPoolExecutor(max_workers=settings.uvicorn_workers)


def _new_job_id() -> str:
    return str(uuid.uuid4())[:8]


def _persist_upload(file: UploadFile) -> Path:
    """Save an uploaded file to disk, return the resolved path."""
    suffix = Path(file.filename or "upload").suffix.lower()
    dest = settings.uploads_dir / f"{_new_job_id()}{suffix}"
    with open(dest, "wb") as buf:
        for chunk in iter(lambda: file.file.read(8192), b""):
            buf.write(chunk)
    return dest


def _run_pipeline_background(job_id: str, image_path: Path, params: dict) -> None:
    """Run pipeline in worker thread and update job registry."""
    logger.info("bg_pipeline_start", job_id=job_id)
    job = _jobs[job_id]
    job.status = JobStatus.PROCESSING
    job.updated_at = datetime.utcnow()

    try:
        pipeline = LegoPipeline()
        outputs = pipeline.run(job_id, image_path, params=params)
        job.output_files = {k: str(v) for k, v in outputs.items()}
        job.status = JobStatus.COMPLETED
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error_message = str(exc)
        logger.error("bg_pipeline_failed", job_id=job_id, error=str(exc))
    finally:
        job.updated_at = datetime.utcnow()
        job.progress_pct = 100 if job.status == JobStatus.COMPLETED else 0


# --------------------------------------------------------------------------- #
# App Factory
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Upload an image -> generate a 3D mesh + LEGO model.",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ---------------------------------------------------------------------- #
    # Routes
    # ---------------------------------------------------------------------- #
    @app.get("/health", response_model=HealthCheck)
    async def health() -> HealthCheck:
        import torch
        return HealthCheck(
            status="ok",
            version=settings.app_version,
            model_loaded=True,  # lazy-loaded on first request
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    @app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
    async def upload_image(
        file: UploadFile = File(..., description="Input image (PNG/JPEG)"),
        remove_bg: Optional[bool] = Form(None),
        foreground_ratio: Optional[float] = Form(None),
        mc_resolution: Optional[int] = Form(None),
        lego_format: Optional[str] = Form("ldr"),
        hollow: bool = Form(False),
    ) -> UploadResponse:
        """
        Accept an image and kick-off background processing.
        Returns a job ID for status polling.
        """
        # Validate extension
        allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in allowed:
            raise HTTPException(400, f"Unsupported file type {suffix}. Use {allowed}")

        # Validate size (fast check via content-length header if available)
        content_length = file.size
        if content_length and content_length > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(413, "File too large")

        # Persist file
        image_path = _persist_upload(file)

        # Register job
        job_id = _new_job_id()
        now = datetime.utcnow()
        _jobs[job_id] = JobState(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            input_filename=file.filename,
        )

        # Prepare params
        params: dict = {"hollow": hollow}
        if remove_bg is not None:
            params["remove_bg"] = remove_bg
        if foreground_ratio is not None:
            params["foreground_ratio"] = foreground_ratio
        if mc_resolution is not None:
            params["mc_resolution"] = mc_resolution

        # Kick off background work
        _executor.submit(_run_pipeline_background, job_id, image_path, params)

        logger.info("upload_accepted", job_id=job_id, filename=file.filename)
        return UploadResponse(job_id=job_id, status=JobStatus.PENDING)

    @app.get("/jobs/{job_id}", response_model=JobState)
    async def get_job(job_id: str) -> JobState:
        """Poll job status and retrieve output file paths when complete."""
        if job_id not in _jobs:
            raise HTTPException(404, "Job not found")
        return _jobs[job_id]

    @app.get("/download/{job_id}/{asset}")
    async def download_asset(job_id: str, asset: str) -> FileResponse:
        """
        Download a generated asset by job_id and asset key.
        Asset keys: mesh_obj, mesh_glb, ldr, manifest
        """
        if job_id not in _jobs:
            raise HTTPException(404, "Job not found")
        job = _jobs[job_id]
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(400, "Job not yet complete")

        filepath = job.output_files.get(asset)
        if not filepath:
            raise HTTPException(404, f"Asset '{asset}' not found for this job")

        path = Path(filepath)
        if not path.exists():
            raise HTTPException(404, "File no longer on disk")

        media_types = {
            "ldr": "text/plain",
            "manifest": "text/plain",
            "mesh_obj": "model/obj",
            "mesh_glb": "model/gltf-binary",
        }
        mt = media_types.get(asset, "application/octet-stream")
        return FileResponse(path, media_type=mt, filename=path.name)

    return app


app = create_app()
