"""Tests for application configuration."""

from backend.app.core.config import Settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_defaults(self) -> None:
        settings = Settings(OPENROUTER_API_KEY="test-key")
        assert settings.OPENROUTER_MODEL == "anthropic/claude-sonnet-4.6"
        assert settings.MAX_ITERATIONS == 5

    def test_custom_model(self) -> None:
        settings = Settings(
            OPENROUTER_API_KEY="test-key",
            OPENROUTER_MODEL="anthropic/claude-opus-4-20250514",
        )
        assert "opus" in settings.OPENROUTER_MODEL

    def test_paths_exist(self) -> None:
        settings = Settings(OPENROUTER_API_KEY="test-key")
        assert settings.DATA_DIR.name == "data"
        assert settings.KNOWLEDGE_BASE_DIR.name == "knowledge-base"
        assert settings.PROMPTS_DIR.name == "prompts"
