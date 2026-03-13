"""Tests for prompt loader."""

import pytest

from backend.app.core.prompts import load_prompt


class TestPromptLoader:
    """Tests for loading system prompt templates."""

    def test_load_existing_prompt(self) -> None:
        text = load_prompt("vision_analysis")
        assert len(text) > 0

    def test_load_with_extension(self) -> None:
        text = load_prompt("vision_analysis.md")
        assert len(text) > 0

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_prompt("does_not_exist")
