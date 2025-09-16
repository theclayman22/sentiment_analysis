"""OpenAI analyzer placeholder."""

from typing import Any

from .base_analyzer import BaseAnalyzer


class OpenAIAnalyzer(BaseAnalyzer):
    """Analyzer using OpenAI models."""

    def analyze(self, text: str) -> Any:
        return {"provider": "openai", "text": text}
