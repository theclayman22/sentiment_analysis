"""Valence analysis model."""

from typing import Any


class ValenceAnalyzer:
    """Classify text as positive, neutral or negative."""

    def analyze(self, text: str) -> Any:
        return {"valence": "neutral", "text": text}
