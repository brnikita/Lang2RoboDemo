"""End-to-end pipeline tests using real room photos.

Requires:
    - E2E_PHOTOS_DIR env var pointing to a directory with 3+ room photos
    - OPENROUTER_API_KEY env var for Claude API calls
    - Running with: pytest -m e2e --timeout=300

Skipped automatically if photos or API key are not available.
Tests run in strict order — each depends on previous steps.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

_PHOTOS_DIR = os.environ.get("E2E_PHOTOS_DIR", "")
_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

_skip_no_photos = pytest.mark.skipif(
    not _PHOTOS_DIR or not Path(_PHOTOS_DIR).exists(),
    reason="E2E_PHOTOS_DIR not set or directory missing",
)
_skip_no_api_key = pytest.mark.skipif(
    not _API_KEY,
    reason="OPENROUTER_API_KEY not set",
)


def _get_photo_files() -> list[Path]:
    """Collect image files from the photos directory."""
    if not _PHOTOS_DIR:
        return []
    photos_dir = Path(_PHOTOS_DIR)
    if not photos_dir.exists():
        return []
    return sorted(f for f in photos_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS)


def _project_dir(project_id: str) -> Path:
    """Get project data directory."""
    from backend.app.core.config import get_settings

    return get_settings().DATA_DIR / "projects" / project_id


@pytest.mark.e2e
@_skip_no_photos
@_skip_no_api_key
class TestFullPipeline:
    """Sequential end-to-end pipeline test.

    All tests share state via class attributes.
    Must run in order: upload → calibrate → recommend → build → simulate → iterate.
    """

    project_id: str = ""
    _client: TestClient | None = None

    @classmethod
    def _get_client(cls) -> TestClient:
        """Get or create TestClient singleton."""
        if cls._client is None:
            import backend.app.core.claude as claude_mod

            claude_mod._client = None  # noqa: SLF001
            cls._client = TestClient(app)
        return cls._client

    def test_01_upload_photos(self) -> None:
        """QA-1: Upload photos and run reconstruction."""
        photos = _get_photo_files()
        assert len(photos) >= 3, f"Need 3+ photos, got {len(photos)}"

        client = self._get_client()
        files = [("photos", (p.name, p.read_bytes(), "image/jpeg")) for p in photos[:16]]
        resp = client.post("/api/capture", files=files)
        assert resp.status_code == 200, f"Upload failed: {resp.text}"

        data = resp.json()
        TestFullPipeline.project_id = data["project_id"]
        assert data["dimensions"]["width_m"] > 0

    def test_02_reconstruction_artifacts(self) -> None:
        """QA-1: Reconstruction produces pointcloud and status."""
        pdir = _project_dir(self.project_id)
        assert (pdir / "reconstruction" / "pointcloud.ply").exists()
        assert (pdir / "status.json").exists()

        status = json.loads((pdir / "status.json").read_text())
        assert status["current_phase"] == "upload"

    def test_03_calibrate(self) -> None:
        """QA-2: Calibrate scale and run vision analysis."""
        client = self._get_client()
        resp = client.post(
            f"/api/capture/{self.project_id}/calibrate",
            json={
                "point_a": [0.0, 0.0, 0.0],
                "point_b": [1.0, 0.0, 0.0],
                "real_distance_m": 0.9,
            },
        )
        assert resp.status_code == 200, f"Calibrate failed: {resp.text}"

        space = resp.json()
        assert space["dimensions"]["width_m"] > 0
        assert space["dimensions"]["area_m2"] > 0

    def test_04_space_model_saved(self) -> None:
        """QA-2: SpaceModel JSON written to project directory."""
        path = _project_dir(self.project_id) / "space_model.json"
        assert path.exists()

    def test_05_generate_recommendation(self) -> None:
        """QA-3: Claude generates a valid recommendation."""
        client = self._get_client()
        resp = client.post(
            "/api/recommend",
            json={
                "project_id": self.project_id,
                "scenario": (
                    "Small workshop. Robot picks parts from intake tray, "
                    "moves to camera inspection, places on work table."
                ),
            },
        )
        assert resp.status_code == 200, f"Recommend failed: {resp.text}"

        rec = resp.json()
        assert len(rec["equipment"]) > 0
        assert len(rec["workflow_steps"]) > 0

    def test_06_recommendation_saved(self) -> None:
        """QA-3: Recommendation JSON saved."""
        path = _project_dir(self.project_id) / "recommendation" / "recommendation.json"
        assert path.exists()

    def test_07_build_scene(self) -> None:
        """QA-4: Scene builds with real equipment models."""
        client = self._get_client()
        resp = client.post(
            f"/api/projects/{self.project_id}/build-scene",
        )
        assert resp.status_code == 200, f"Build failed: {resp.text}"

        data = resp.json()
        assert data["valid"] is True
        assert data["equipment_count"] > 0

    def test_08_scene_has_joints(self) -> None:
        """QA-4: Scene contains articulated joints and actuators."""
        import mujoco

        scene = _project_dir(self.project_id) / "scenes" / "v1.xml"
        assert scene.exists()

        model = mujoco.MjModel.from_xml_path(str(scene))
        assert model.njnt > 0, f"No joints (njnt={model.njnt})"
        assert model.nu > 0, f"No actuators (nu={model.nu})"
        assert model.nsite > 0, f"No sites (nsite={model.nsite})"

    def test_09_run_simulation(self) -> None:
        """QA-5: Simulation runs without 500 error."""
        client = self._get_client()
        resp = client.post(
            f"/api/projects/{self.project_id}/simulate",
        )
        assert resp.status_code == 200, f"Simulate failed: {resp.text}"

        result = resp.json()
        assert result["metrics"]["cycle_time_s"] > 0

    def test_10_no_ee_site_errors(self) -> None:
        """QA-5: No EE site resolution errors in results."""
        path = _project_dir(self.project_id) / "simulations" / "latest.json"
        result = json.loads(path.read_text())
        for step in result["steps"]:
            error = step.get("error") or ""
            assert "No EE site" not in error, f"EE error: {error}"

    def test_11_run_optimization(self) -> None:
        """QA-7: Optimization loop completes."""
        client = self._get_client()
        resp = client.post(
            f"/api/projects/{self.project_id}/iterate",
            json={"max_iterations": 2},
        )
        assert resp.status_code == 200, f"Iterate failed: {resp.text}"

        data = resp.json()
        assert data["iterations_run"] >= 1

    def test_12_multiple_scene_versions(self) -> None:
        """QA-7: Iteration creates multiple scene versions."""
        scenes_dir = _project_dir(self.project_id) / "scenes"
        xmls = sorted(scenes_dir.glob("v*.xml"))
        assert len(xmls) >= 2, f"Only {len(xmls)} versions"

    def test_13_dashboard_lists_project(self) -> None:
        """QA-8: Project appears in dashboard API."""
        client = self._get_client()
        resp = client.get("/api/projects")
        assert resp.status_code == 200

        ids = [p["id"] for p in resp.json()]
        assert self.project_id in ids

    def test_14_project_detail_restores(self) -> None:
        """QA-8: Project detail returns full state."""
        client = self._get_client()
        resp = client.get(f"/api/projects/{self.project_id}")
        assert resp.status_code == 200

        detail = resp.json()
        assert detail["status"]["id"] == self.project_id
        assert detail["dimensions"] is not None
        assert detail["recommendation"] is not None
        assert detail["sim_result"] is not None
