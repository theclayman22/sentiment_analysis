"""VADER analyzer placeholder."""

from typing import Any

from .base_analyzer import BaseAnalyzer


class VaderAnalyzer(BaseAnalyzer):
    """Analyzer using VADER sentiment analysis."""

    def analyze(self, text: str) -> Any:
        return {"provider": "vader", "text": text}
