"""Tests for Claude Vision scene analysis."""

from backend.app.models.space import (
    Dimensions,
    ExistingEquipment,
    SceneAnalysis,
    Zone,
)
from backend.app.services.vision import _extract_json, build_space_model


class TestExtractJson:
    """Tests for JSON extraction from Claude responses."""

    def test_plain_json(self) -> None:
        text = '{"zones": [], "existing_equipment": [], "doors": [], "windows": []}'
        result = _extract_json(text)
        assert result.startswith("{")

    def test_json_in_code_block(self) -> None:
        text = '```json\n{"zones": []}\n```'
        result = _extract_json(text)
        assert result == '{"zones": []}'

    def test_json_in_generic_code_block(self) -> None:
        text = '```\n{"zones": []}\n```'
        result = _extract_json(text)
        assert result == '{"zones": []}'


class TestBuildSpaceModel:
    """Tests for SpaceModel composition."""

    def test_merge_reconstruction_and_analysis(
        self, sample_reconstruction,
    ) -> None:
        analysis = SceneAnalysis(
            zones=[
                Zone(
                    name="main_area",
                    polygon=[(0, 0), (6, 0), (6, 5), (0, 5)],
                    area_m2=30.0,
                )
            ],
            existing_equipment=[
                ExistingEquipment(
                    name="printer_1",
                    category="printer",
                    position=(1.0, 2.0, 0.85),
                    confidence=0.9,
                )
            ],
        )
        space = build_space_model(sample_reconstruction, analysis)

        assert space.dimensions.width_m == 6.0
        assert len(space.zones) == 1
        assert space.zones[0].name == "main_area"
        assert len(space.existing_equipment) == 1
        assert space.reconstruction == sample_reconstruction
