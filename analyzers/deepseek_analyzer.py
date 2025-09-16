"""DeepSeek analyzer placeholder."""

from typing import Any

from .base_analyzer import BaseAnalyzer


class DeepSeekAnalyzer(BaseAnalyzer):
    """Analyzer using DeepSeek models."""

    def analyze(self, text: str) -> Any:
        return {"provider": "deepseek", "text": text}
