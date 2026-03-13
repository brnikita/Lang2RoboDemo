"""Tests for OpenRouter Claude client."""

import os

import pytest

from backend.app.core.claude import ClaudeClient


@pytest.mark.e2e
class TestClaudeClientIntegration:
    """Integration tests for Claude client (requires OPENROUTER_API_KEY)."""

    @pytest.fixture
    def client(self) -> ClaudeClient:
        """Create a Claude client with real API key."""
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")
        return ClaudeClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-sonnet-4-20250514",
        )

    @pytest.mark.asyncio
    async def test_simple_message(self, client: ClaudeClient) -> None:
        response = await client.send_message(
            system="You are a helpful assistant. Reply in one word.",
            messages=[{"role": "user", "content": "Say hello."}],
        )
        assert len(response) > 0
        await client.close()
