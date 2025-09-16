"""Keyword-basierter OpenAI Analyzer als Fallback."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from analyzers.base_analyzer import AnalysisResult, BaseAnalyzer
from config.emotion_mappings import EKMAN_EMOTIONS

_POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "happy",
    "joy",
    "love",
    "awesome",
    "fantastic",
    "wonderful",
    "pleasant",
    "delight",
}

_NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "sad",
    "hate",
    "angry",
    "horrible",
    "disgust",
    "fear",
    "upset",
    "annoyed",
}


class OpenAIAnalyzer(BaseAnalyzer):
    """Ein heuristischer Analyzer, der OpenAI-Modelle simuliert."""

    def __init__(
        self, api_config: Optional[Any] = None, model_name: str = "apt-5-nano"
    ) -> None:
        super().__init__(model_name=model_name, api_config=api_config)
        self._api_key_available = bool(getattr(api_config, "primary_key", None))

    def analyze_single(self, text: str, analysis_type: str, **kwargs) -> AnalysisResult:
        tokens = self._tokenize(text)
        metadata = {
            "provider": "openai",
            "model": self.model_name,
            "api_key_available": self._api_key_available,
        }

        start_time = time.time()
        if analysis_type == "valence":
            scores = self._analyze_valence(tokens)
        elif analysis_type == "ekman":
            scores = self._analyze_ekman(tokens)
        elif analysis_type == "emotion_arc":
            scores = self._analyze_emotion_arc(tokens)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

        processing_time = time.time() - start_time
        return AnalysisResult(
            text=text,
            model=self.model_name,
            analysis_type=analysis_type,
            scores=scores,
            processing_time=processing_time,
            metadata=metadata,
        )

    def analyze_batch(self, texts: List[str], analysis_type: str, **kwargs) -> List[AnalysisResult]:
        return [self.analyze_single(text, analysis_type, **kwargs) for text in texts]

    def is_available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # interne Hilfsfunktionen
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in text.split() if token.strip()]

    def _normalise(self, scores: Dict[str, float]) -> Dict[str, float]:
        total = sum(scores.values())
        if total <= 0:
            uniform = 1.0 / len(scores) if scores else 0.0
            return {key: uniform for key in scores}
        return {key: value / total for key, value in scores.items()}

    def _analyze_valence(self, tokens: Iterable[str]) -> Dict[str, float]:
        token_list = list(tokens)
        counter = Counter(token_list)
        positive_hits = sum(counter[word] for word in _POSITIVE_WORDS)
        negative_hits = sum(counter[word] for word in _NEGATIVE_WORDS)
        neutral_hits = max(0, len(token_list) - (positive_hits + negative_hits))

        scores = {
            "positive": positive_hits + 1.0,
            "negative": negative_hits + 1.0,
            "neutral": neutral_hits + 1.0,
        }
        return self._normalise(scores)

    def _analyze_ekman(self, tokens: Iterable[str]) -> Dict[str, float]:
        token_list = list(tokens)
        counter = Counter(token_list)
        scores: Dict[str, float] = {}
        for emotion, data in EKMAN_EMOTIONS.items():
            synonyms = {emotion.lower(), *[syn.lower() for syn in data["synonyms"]]}
            hits = sum(counter[word] for word in synonyms)
            scores[emotion] = float(hits + 1.0)
        return self._normalise(scores)

    def _analyze_emotion_arc(self, tokens: Iterable[str]) -> Dict[str, float]:
        valence = self._analyze_valence(tokens)
        happiness = valence.get("positive", 0.0) + 0.5 * valence.get("neutral", 0.0)
        return {"happiness": max(0.0, min(1.0, happiness))}


__all__ = ["OpenAIAnalyzer"]
