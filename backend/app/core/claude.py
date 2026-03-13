"""OpenRouter Claude API client using OpenAI-compatible format."""

import base64
import logging
from pathlib import Path

import httpx

from backend.app.core.config import get_settings

__all__ = ["ClaudeClient", "get_claude_client"]

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


class ClaudeClient:
    """Async client for Claude API via OpenRouter.

    Args:
        api_key: OpenRouter API key.
        base_url: OpenRouter base URL.
        model: Model identifier.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def send_message(
        self,
        system: str,
        messages: list[dict],
        model: str | None = None,
    ) -> str:
        """Send a text message and return the response text.

        Args:
            system: System prompt.
            messages: Conversation messages in OpenAI format.
            model: Model override (uses default if None).

        Returns:
            Response text content.

        Raises:
            httpx.HTTPStatusError: On API errors after retries.
        """
        return await self._request(system, messages, model)

    async def send_vision_message(
        self,
        system: str,
        images: list[Path],
        text: str,
        model: str | None = None,
    ) -> str:
        """Send images + text and return the response text.

        Args:
            system: System prompt.
            images: List of image file paths.
            text: Text message to accompany images.
            model: Model override (uses default if None).

        Returns:
            Response text content.

        Raises:
            httpx.HTTPStatusError: On API errors after retries.
        """
        content = _build_vision_content(images, text)
        messages = [{"role": "user", "content": content}]
        return await self._request(system, messages, model)

    async def _request(
        self,
        system: str,
        messages: list[dict],
        model: str | None,
    ) -> str:
        """Execute API request with retries.

        Args:
            system: System prompt.
            messages: Conversation messages.
            model: Model override.

        Returns:
            Response text content.
        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or self._model,
            "messages": [{"role": "system", "content": system}, *messages],
        }

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(
                    url, json=payload, headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                return _extract_text(data)
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                logger.warning(
                    "Claude API attempt %d/%d failed: %s",
                    attempt + 1, _MAX_RETRIES + 1, exc,
                )

        raise last_error  # type: ignore[misc]


def _build_vision_content(
    images: list[Path], text: str,
) -> list[dict]:
    """Build multimodal content array for vision request.

    Args:
        images: Image file paths to encode.
        text: Accompanying text message.

    Returns:
        List of content blocks (image_url + text).
    """
    content: list[dict] = []
    for image_path in images:
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        suffix = image_path.suffix.lstrip(".").lower()
        mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
    content.append({"type": "text", "text": text})
    return content


def _extract_text(response_data: dict) -> str:
    """Extract text content from OpenAI-format response.

    Args:
        response_data: Parsed JSON response.

    Returns:
        Text content from the first choice.

    Raises:
        ValueError: If response format is unexpected.
    """
    choices = response_data.get("choices", [])
    if not choices:
        raise ValueError("Empty choices in API response")
    return choices[0]["message"]["content"]


_client: ClaudeClient | None = None


def get_claude_client() -> ClaudeClient:
    """Return cached Claude client singleton.

    Returns:
        Configured ClaudeClient instance.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        settings = get_settings()
        _client = ClaudeClient(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            model=settings.OPENROUTER_MODEL,
        )
    return _client
