"""Ekman emotion classifier."""

from typing import Any


class EkmanAnalyzer:
    """Identify Ekman emotions in text."""

    def analyze(self, text: str) -> Any:
        return {"emotion": "neutral", "text": text}
