"""Prompt loader for system prompt templates."""

from backend.app.core.config import get_settings

__all__ = ["load_prompt"]


def load_prompt(name: str) -> str:
    """Load a system prompt template from the prompts directory.

    Args:
        name: Prompt file name (with or without .md extension).

    Returns:
        Prompt text content.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    settings = get_settings()
    path = settings.PROMPTS_DIR / name
    if not path.suffix:
        path = path.with_suffix(".md")
    return path.read_text(encoding="utf-8")
