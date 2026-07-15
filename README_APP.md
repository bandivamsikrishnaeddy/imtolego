# ImToLego – Enterprise Image-to-LEGO Pipeline

A production-grade system that turns **one image** into:
1.  A high quality **3D mesh** (via TripoSR)
2.  A **LEGO brick model** (.ldr file compatible with BrickLink Studio / LeoCAD)
3.  A **bill-of-materials** (parts manifest)

Built with **FastAPI**, **TripoSR**, and a custom **voxel → brick packing** engine.

`tsr/` is TripoSR's model code (MIT licensed, see `tsr/LICENSE`), vendored
directly into this repo rather than pulled in as a separate clone/submodule.

---

## Project Layout

```
imtolego/
├── app/
│   ├── api/server.py          # FastAPI app (upload / jobs / download / health)
│   ├── core/
│   │   ├── config.py          # Pydantic settings (env-driven)
│   │   └── logging.py         # structured JSON logging
│   ├── lego/
│   │   └── converter.py       # mesh → voxels → standard LEGO bricks → LDraw
│   ├── models/
│   │   └── schemas.py         # pydantic request / response models
│   └── pipeline/
│       └── orchestrator.py     # full pipeline w/ progress callbacks
├── tsr/                       # TripoSR model code (vendored)
├── tests/
│   └── test_smoke.py
├── scripts/
│   └── run.sh                 # production server bootstrap
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── uploads/                   # temporary uploaded images
├── outputs/                   # generated meshes + LEGO files
├── environment.yml            # Conda environment spec
└── requirements.txt
```

---

## Quick Start

### 1. Prerequisites

- Conda (miniconda or anaconda)
- macOS / Linux (Windows via WSL2)
- ~10 GB disk space for model & environment

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate imtolego
```

### 3. Run the API Server

```bash
./scripts/run.sh
```

or directly with uvicorn:

```bash
export PYTHONPATH=$(pwd)
python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

### 4. Test with cURL

```bash
# Upload an image and get a job_id
curl -X POST "http://localhost:8000/upload" \
     -F "file=@/path/to/your/image.png" \
     -F "remove_bg=true" \
     -F "foreground_ratio=0.85" \
     -F "mc_resolution=256"
# → {"job_id":"a1b2c3d4","status":"pending"}

# Poll job status
curl "http://localhost:8000/jobs/a1b2c3d4"

# Download outputs when completed
curl -o model.ldr    "http://localhost:8000/download/a1b2c3d4/ldr"
curl -o mesh.glb     "http://localhost:8000/download/a1b2c3d4/mesh_glb"
curl -o manifest.txt "http://localhost:8000/download/a1b2c3d4/manifest"
```

---

## Pipeline Flow

```
User Image
    │
    ▼
Background Removal (rembg)
    │
    ▼
TripoSR Inference  ──→  Scene Code (neural field)
    │
    ▼
Marching Cubes  ──→  Mesh (.obj / .glb)
    │
    ▼
LEGO Voxelization  ──→  Occupancy Grid
    │
    ▼
Greedy Brick Packing  ──→  Standard Bricks (1x1, 1x2, 2x2, 2x4 …)
    │
    ▼
LDraw Export (.ldr)  +  Parts Manifest (bill of materials)
```

### Supported Brick Types

| Size   | Height |
|--------|--------|
| 1×1    | plate  |
| 1×2    | plate  |
| 2×2    | plate  |
| 1×3    | plate  |
| 2×3    | plate  |
| 1×4    | plate  |
| 2×4    | plate  |
| 1×6    | plate  |
| 2×6    | plate  |

*(Brick heights are generated with 3× plate = 1 brick where applicable.)*

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Service & model readiness |
| POST | `/upload` | Accept image + optional params, returns `job_id` |
| GET  | `/jobs/{job_id}` | Poll status & output files |
| GET  | `/download/{job_id}/{asset}` | Download asset (`ldr`, `manifest`, `mesh_obj`, `mesh_glb`) |
| GET  | `/metrics` | Prometheus metrics |

### Upload Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | PNG/JPG input |
| `remove_bg` | bool | `true` | Auto-remove background |
| `foreground_ratio` | float | `0.85` | Foreground crop ratio |
| `mc_resolution` | int | `256` | Marching cubes grid resolution (`32`–`512`) |
| `hollow` | bool | `false` | Create hollow shell instead of solid brick mass |

---

## Docker (optional)

```bash
cd docker
docker-compose up --build
```

- API exposed on `localhost:8000`
- Volumes: `uploads/` and `outputs/` mounted from host
- GPU support configured via `deploy.resources` (Linux + nvidia-docker required)

---

## Configuration

All settings are driven by environment variables (see `.env.example`):

```bash
UVICORN_PORT=8000
DEVICE=cpu                     # or cuda:0 / mps
MODEL_ID=stabilityai/TripoSR
MC_RESOLUTION=256
LEGO_VOXEL_SCALE=0.02
LEGO_WALL_THICKNESS=1
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | TripoSR (Stability AI / Tripo AI) |
| API   | FastAPI + Uvicorn |
| 3D    | trimesh, torchmcubes |
| LEGO  | Custom greedy packer + LDraw spec writer |
| Infra | Docker, Prometheus metrics, structured logging (structlog) |

---

## License

MIT – derived from the original TripoSR license.
