"""
VADER Sentiment Analyzer
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from analyzers.base_analyzer import AnalysisResult, BaseAnalyzer


class VADERAnalyzer(BaseAnalyzer):
    """VADER Sentiment Analyzer"""

    def __init__(self, api_config: Any | None = None) -> None:
        super().__init__("vader", api_config)
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_single(self, text: str, analysis_type: str, **kwargs) -> AnalysisResult:
        """Analysiert einen einzelnen Text"""
        metadata = {"provider": "vader"}

        try:
            start_time = time.time()

            if analysis_type == "valence":
                scores = self._analyze_valence(text, **kwargs)
            elif analysis_type == "ekman":
                scores = {
                    emotion: 0.1
                    for emotion in [
                        "joy",
                        "surprise",
                        "fear",
                        "anger",
                        "disgust",
                        "sadness",
                        "contempt",
                    ]
                }
            elif analysis_type == "emotion_arc":
                scores = self._analyze_happiness(text, **kwargs)
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

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Error analysing text with VADER: %s", exc)
            return AnalysisResult(
                text=text,
                model=self.model_name,
                analysis_type=analysis_type,
                scores={},
                processing_time=0.0,
                metadata=metadata,
                error=str(exc),
            )

    def analyze_batch(self, texts: List[str], analysis_type: str, **kwargs) -> List[AnalysisResult]:
        """Analysiert eine Liste von Texten schnell"""
        return [self.analyze_single(text, analysis_type, **kwargs) for text in texts]

    def is_available(self) -> bool:
        """VADER ist immer verfügbar"""
        return True

    def _analyze_valence(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Valence mit VADER"""
        scores = self.analyzer.polarity_scores(text)

        positive = float(scores.get("pos", 0.0))
        negative = float(scores.get("neg", 0.0))
        neutral = float(scores.get("neu", 0.0))

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        }

    def _analyze_happiness(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Happiness für Emotion Arc"""
        scores = self.analyzer.polarity_scores(text)
        compound = float(scores.get("compound", 0.0))
        happiness = (compound + 1) / 2

        return {"happiness": happiness}


# Rückwärtskompatibilität zum bisherigen Namen
class VaderAnalyzer(VADERAnalyzer):
    """Alias für VADERAnalyzer für bestehende Importe"""

    pass
