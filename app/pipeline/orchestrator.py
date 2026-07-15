"""
End-to-End Pipeline Orchestrator
--------------------------------
Encapsulates the entire workflow:
  Image Upload -> Background Removal -> TripoSR Inference
  -> Mesh Extraction -> LEGO Conversion -> Asset Packaging
"""

import shutil
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rembg
import structlog
import torch
import trimesh
from PIL import Image

# TripoSR imports (from the copied module)
from tsr.system import TSR
from tsr.utils import remove_background as tsr_remove_bg, resize_foreground

from app.core.config import get_settings
from app.lego.converter import LegoConverter
from app.models.schemas import JobState, JobStatus

logger = structlog.get_logger()
settings = get_settings()


# --------------------------------------------------------------------------- #
# Event / Callback interface for progress tracking
# --------------------------------------------------------------------------- #
class PipelineObserver(ABC):
    @abstractmethod
    def on_stage_change(self, job_id: str, stage: str, progress_pct: int) -> None:
        ...

    @abstractmethod
    def on_complete(self, job_id: str, outputs: Dict[str, Path]) -> None:
        ...

    @abstractmethod
    def on_failure(self, job_id: str, error: str) -> None:
        ...


class DefaultPipelineObserver(PipelineObserver):
    """Default observer that simply logs events."""

    def on_stage_change(self, job_id: str, stage: str, progress_pct: int) -> None:
        logger.info("pipeline.stage_change", job_id=job_id, stage=stage, progress=progress_pct)

    def on_complete(self, job_id: str, outputs: Dict[str, Path]) -> None:
        logger.info("pipeline.complete", job_id=job_id, outputs={k: str(v) for k, v in outputs.items()})

    def on_failure(self, job_id: str, error: str) -> None:
        logger.error("pipeline.failed", job_id=job_id, error=error)


# --------------------------------------------------------------------------- #
# Service singletons (heavyweight, kept alive)
# --------------------------------------------------------------------------- #
class MLService:
    """Encapsulates TripoSR model loading and inference."""

    _instance: Optional["MLService"] = None
    _model: Optional[TSR] = None
    _rembg_session: Any = None
    _device: str = "cpu"

    def __new__(cls) -> "MLService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        if self._model is not None:
            return
        cfg = get_settings()
        # Auto-select best available device: cuda > mps > cpu
        if torch.cuda.is_available():
            self._device = cfg.device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        logger.info("loading_tripo_model", model_id=cfg.model_id, device=self._device)
        self._model = TSR.from_pretrained(
            cfg.model_id,
            config_name=cfg.model_config_name,
            weight_name=cfg.model_weight_name,
        )
        self._model.renderer.set_chunk_size(cfg.chunk_size)
        self._model.to(self._device)
        self._rembg_session = rembg.new_session()
        logger.info("model_loaded")

    @property
    def model(self) -> TSR:
        if self._model is None:
            raise RuntimeError("MLService not loaded. Call .load() first.")
        return self._model

    @property
    def rembg_session(self) -> Any:
        return self._rembg_session

    @property
    def device(self) -> str:
        return self._device


def get_ml_service() -> MLService:
    svc = MLService()
    svc.load()
    return svc


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class LegoPipeline:
    """
    Responsible for orchestrating the full image -> LEGO conversion.
    Designed to be called inside a FastAPI endpoint or a Celery task.
    """

    def __init__(self, observer: Optional[PipelineObserver] = None):
        self.observer = observer or DefaultPipelineObserver()
        self.ml = get_ml_service()
        self.lego = LegoConverter(
            target_max_studs=settings.lego_target_max_studs,
            hollow=settings.lego_hollow,
        )

    def _notify(self, job_id: str, stage: str, progress_pct: int) -> None:
        self.observer.on_stage_change(job_id, stage, progress_pct)

    def run(self, job_id: str, image_path: Path, params: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Execute the full pipeline synchronously.
        Returns a dict of generated file paths.
        """
        params = params or {}
        output_dir = settings.outputs_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ---------------------------------------------------------------- #
            # 1. Load & preprocess image
            # ---------------------------------------------------------------- #
            self._notify(job_id, "preprocessing", 5)
            do_remove_bg = params.get("remove_bg", settings.remove_bg)
            fg_ratio = params.get("foreground_ratio", settings.foreground_ratio)
            image = self._preprocess(image_path, do_remove_bg, fg_ratio)
            logger.info("image_preprocessed", job_id=job_id, size=image.size)

            # ---------------------------------------------------------------- #
            # 2. TripoSR inference -> scene code
            # ---------------------------------------------------------------- #
            self._notify(job_id, "inference", 20)
            with torch.no_grad():
                scene_codes = self.ml.model([image], device=self.ml.device)
            logger.info("inference_complete", job_id=job_id)

            # ---------------------------------------------------------------- #
            # 3. Extract mesh (with vertex colours)
            # ---------------------------------------------------------------- #
            self._notify(job_id, "mesh_extraction", 40)
            mc_res = params.get("mc_resolution", settings.mc_resolution)
            meshes = self.ml.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_res)
            mesh: trimesh.Trimesh = meshes[0]
            mesh_path = output_dir / "mesh.obj"
            mesh.export(str(mesh_path))
            logger.info("mesh_extracted", job_id=job_id, vertices=len(mesh.vertices), faces=len(mesh.faces))

            # Also export GLB for web preview
            glb_path = output_dir / "mesh.glb"
            mesh.export(str(glb_path))

            # ---------------------------------------------------------------- #
            # 4. LEGO conversion
            # ---------------------------------------------------------------- #
            self._notify(job_id, "lego_conversion", 70)
            lego_files = self.lego.convert(mesh, output_dir)
            logger.info("lego_conversion_complete", job_id=job_id, files={k: str(v) for k, v in lego_files.items()})

            # ---------------------------------------------------------------- #
            # 5. Package / manifest
            # ---------------------------------------------------------------- #
            self._notify(job_id, "packaging", 90)
            manifest = {
                "mesh_obj": mesh_path,
                "mesh_glb": glb_path,
                **lego_files,
            }
            # Write a simple results.json for programmatic consumers
            import json
            results_meta = {k: str(v.resolve()) for k, v in manifest.items()}
            (output_dir / "results.json").write_text(json.dumps(results_meta, indent=2))

            self._notify(job_id, "complete", 100)
            self.observer.on_complete(job_id, manifest)
            return manifest

        except Exception as exc:  # pragma: no cover
            logger.exception("pipeline_failed", job_id=job_id, error=str(exc))
            self.observer.on_failure(job_id, str(exc))
            raise

    # ----------------------------------------------------------------------- #
    # Preprocessing helpers
    # ----------------------------------------------------------------------- #
    def _preprocess(self, image_path: Path, do_remove_bg: bool, fg_ratio: float) -> Image.Image:
        pil = Image.open(image_path).convert("RGBA")
        if do_remove_bg:
            pil = tsr_remove_bg(pil, self.ml.rembg_session)
            pil = resize_foreground(pil, fg_ratio)
            # Composite onto neutral grey background
            arr = np.array(pil).astype(np.float32) / 255.0
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
            pil = Image.fromarray((arr * 255.0).astype(np.uint8))
        else:
            if pil.mode == "RGBA":
                arr = np.array(pil).astype(np.float32) / 255.0
                arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
                pil = Image.fromarray((arr * 255.0).astype(np.uint8))
            else:
                pil = pil.convert("RGB")
        return pil
