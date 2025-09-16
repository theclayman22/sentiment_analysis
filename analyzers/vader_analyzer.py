"""
VADER analyzer implementation.
"""

from typing import Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .base_analyzer import BaseAnalyzer, AnalysisResult


class VaderAnalyzer(BaseAnalyzer):
    """Analyzer using VADER sentiment analysis."""

    def __init__(self) -> None:
        super().__init__(model_name="vader", api_config=None)
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze_single(self, text: str, analysis_type: str, **kwargs) -> AnalysisResult:
        def _compute_scores() -> Dict[str, float]:
            return self._analyzer.polarity_scores(text)

        scores, processing_time = self._measure_time(_compute_scores)
        return AnalysisResult(
            text=text,
            model=self.model_name,
            analysis_type=analysis_type,
            scores=scores,
            processing_time=processing_time,
            metadata={"provider": "vader"},
        )

    def analyze_batch(self, texts: List[str], analysis_type: str, **kwargs) -> List[AnalysisResult]:
        return [self.analyze_single(text, analysis_type, **kwargs) for text in texts]

    def is_available(self) -> bool:
        return True
