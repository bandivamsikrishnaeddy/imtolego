"""
Integration / Smoke Tests
--------------------------
Tests the FastAPI endpoints and the LEGO conversion engine without requiring
a full TripoSR model download (uses a simple cube for lego tests).
"""

import io

import numpy as np
import pytest
import trimesh
from PIL import Image

from app.api.server import create_app
from app.lego.converter import LegoConverter
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestLegoConverter:
    def test_cube_conversion(self, tmp_path):
        """Ensure a simple cube becomes bricks and an LDraw file."""
        mesh = trimesh.creation.box(extents=[40, 30, 24])
        # add colours
        n = len(mesh.vertices)
        mesh.visual.vertex_colors = np.column_stack([
            np.full(n, 200), np.full(n, 50), np.full(n, 50), np.full(n, 255)
        ])
        converter = LegoConverter(target_max_studs=10)
        results = converter.convert(mesh, tmp_path)
        assert "ldr" in results
        assert "manifest" in results
        assert results["ldr"].exists()
        text = results["ldr"].read_text()
        assert "1" in text  # LDraw brick lines start with "1"

    def test_hollow_shell(self, tmp_path):
        mesh = trimesh.creation.box(extents=[40, 40, 40])
        n = len(mesh.vertices)
        mesh.visual.vertex_colors = np.column_stack([
            np.full(n, 200), np.full(n, 200), np.full(n, 200), np.full(n, 255)
        ])
        converter = LegoConverter(target_max_studs=8, hollow=True)
        out = converter.convert(mesh, tmp_path)
        assert out["ldr"].exists()


class TestUploadEndpoint:
    def test_upload_wrong_extension(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("bad.exe", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_upload_accepts_png(self, client):
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        resp = client.post(
            "/upload",
            files={"file": ("test.png", buf, "image/png")},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"
