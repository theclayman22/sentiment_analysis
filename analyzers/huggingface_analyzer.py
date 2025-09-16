"""HuggingFace analyzer placeholder."""

from typing import Any

from .base_analyzer import BaseAnalyzer


class HuggingFaceAnalyzer(BaseAnalyzer):
    """Analyzer using HuggingFace models."""

    def analyze(self, text: str) -> Any:
        return {"provider": "huggingface", "text": text}
