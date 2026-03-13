"""Application configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings

__all__ = ["Settings", "get_settings"]

_ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Args:
        OPENROUTER_API_KEY: API key for OpenRouter.
        OPENROUTER_MODEL: Model identifier on OpenRouter.
        OPENROUTER_BASE_URL: OpenRouter API base URL.
        DATA_DIR: Directory for project data.
        MODELS_DIR: Directory for cached equipment models.
        KNOWLEDGE_BASE_DIR: Directory for equipment catalog JSONs.
        PROMPTS_DIR: Directory for system prompt templates.
        MAX_ITERATIONS: Maximum number of improvement iterations.
    """

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "anthropic/claude-sonnet-4-20250514"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    DATA_DIR: Path = _ROOT_DIR / "data"
    MODELS_DIR: Path = _ROOT_DIR / "models"
    KNOWLEDGE_BASE_DIR: Path = _ROOT_DIR / "knowledge-base"
    PROMPTS_DIR: Path = _ROOT_DIR / "prompts"

    MAX_ITERATIONS: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


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
