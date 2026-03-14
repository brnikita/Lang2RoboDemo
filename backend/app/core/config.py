"""Application configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings

__all__ = ["Settings", "get_settings"]

_ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Args:
        OPENROUTER_API_KEY: API key for OpenRouter.
        OPENROUTER_MODEL: Default model identifier on OpenRouter.
        OPENROUTER_VISION_MODEL: Model for vision analysis (falls back to OPENROUTER_MODEL).
        OPENROUTER_PLANNING_MODEL: Model for planning/iteration (falls back to OPENROUTER_MODEL).
        OPENROUTER_BASE_URL: OpenRouter API base URL.
        DATA_DIR: Directory for project data.
        MODELS_DIR: Directory for cached equipment models.
        KNOWLEDGE_BASE_DIR: Directory for equipment catalog JSONs.
        PROMPTS_DIR: Directory for system prompt templates.
        MAX_ITERATIONS: Maximum number of improvement iterations.
    """

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "anthropic/claude-sonnet-4.6"
    OPENROUTER_VISION_MODEL: str = ""
    OPENROUTER_PLANNING_MODEL: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 5173

    DATA_DIR: Path = _ROOT_DIR / "data"
    MODELS_DIR: Path = _ROOT_DIR / "models"
    KNOWLEDGE_BASE_DIR: Path = _ROOT_DIR / "knowledge-base"
    PROMPTS_DIR: Path = _ROOT_DIR / "prompts"

    MAX_ITERATIONS: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def vision_model(self) -> str:
        """Resolve model for vision analysis tasks."""
        return self.OPENROUTER_VISION_MODEL or self.OPENROUTER_MODEL

    @property
    def planning_model(self) -> str:
        """Resolve model for planning and iteration tasks."""
        return self.OPENROUTER_PLANNING_MODEL or self.OPENROUTER_MODEL


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached application settings singleton.

    Returns:
        Application settings instance.
    """
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings
