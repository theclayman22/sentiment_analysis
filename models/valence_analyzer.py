"""Valence-Analyse Koordinator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from analyzers.base_analyzer import AnalysisResult
from analyzers.deepseek_analyzer import DeepSeekAnalyzer
from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from analyzers.openai_analyzer import OpenAIAnalyzer
from analyzers.vader_analyzer import VADERAnalyzer
from config.settings import Settings
from utils.api_manager import APIManager


class ValenceAnalyzer:
    """Koordiniert Valence-Analysen über alle verfügbaren Modelle."""

    def __init__(self) -> None:
        self.api_manager = APIManager()
        self.analyzers: Dict[str, object] = {}
        self._initialize_analyzers()

    def _initialize_analyzers(self) -> None:
        """Initialisiert alle verfügbaren Analyzer mit Fallback-Logik."""
        try:
            config = self.api_manager.get_api_config("openai_reasoning")
            if getattr(config, "primary_key", None):
                self.analyzers["apt-5-nano"] = OpenAIAnalyzer(config)
        except Exception:  # pragma: no cover - defensive initialisation
            pass

        try:
            config = self.api_manager.get_api_config("deepseek")
            if getattr(config, "primary_key", None):
                self.analyzers["deepseek-chat"] = DeepSeekAnalyzer(config)
        except Exception:  # pragma: no cover - defensive initialisation
            pass

        try:
            config = self.api_manager.get_api_config("huggingface")
            if getattr(config, "primary_key", None):
                for model_name in [
                    "facebook/bart-large",
                    "FacebookAI/roberta-base",
                    "siebert/sentiment-roberta-large-english",
                ]:
                    self.analyzers[model_name] = HuggingFaceAnalyzer(model_name, config)
        except Exception:  # pragma: no cover - defensive initialisation
            pass

        # VADER ist immer verfügbar
        self.analyzers["vader"] = VADERAnalyzer()

    def analyze_single(
        self, text: str, models: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, AnalysisResult]:
        """Analysiert einen Text mit den angegebenen Modellen."""
        available_models = models or list(self.analyzers.keys())
        selected_models = [name for name in available_models if name in self.analyzers]

        if not selected_models:
            return {}

        max_workers = min(Settings.MAX_WORKERS, len(selected_models)) or 1
        results: Dict[str, AnalysisResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for model_name in selected_models:
                analyzer = self.analyzers[model_name]
                if hasattr(analyzer, "is_available") and analyzer.is_available():  # type: ignore[attr-defined]
                    future = executor.submit(
                        analyzer.analyze_single, text, "valence", **kwargs  # type: ignore[attr-defined]
                    )
                    futures[future] = model_name

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result(timeout=Settings.REQUEST_TIMEOUT)
                    results[model_name] = result
                except Exception as exc:  # pragma: no cover - defensive runtime guard
                    results[model_name] = AnalysisResult(
                        text=text,
                        model=model_name,
                        analysis_type="valence",
                        scores={},
                        processing_time=0.0,
                        error=str(exc),
                    )

        return results

    def analyze_batch(
        self, texts: List[str], models: Optional[List[str]] = None, **kwargs
    ) -> List[Dict[str, AnalysisResult]]:
        """Analysiert eine Liste von Texten sequenziell."""
        return [self.analyze_single(text, models, **kwargs) for text in texts]

    def aggregate_scores(self, results: Dict[str, AnalysisResult]) -> Dict[str, float]:
        """Aggregiert Valence-Scores über mehrere Modelle."""
        aggregates = {"positive": [], "negative": [], "neutral": []}

        for result in results.values():
            if result.error:
                continue
            for key in aggregates:
                if key in result.scores:
                    aggregates[key].append(float(result.scores[key]))

        aggregated_scores = {key: (sum(values) / len(values) if values else 0.0) for key, values in aggregates.items()}
        total = sum(aggregated_scores.values())
        if total > 0:
            aggregated_scores = {key: value / total for key, value in aggregated_scores.items()}
        else:
            uniform = 1.0 / len(aggregated_scores)
            aggregated_scores = {key: uniform for key in aggregated_scores}

        return aggregated_scores

    def analyze(
        self, text: str, models: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Kompatibilitätsmethode für bestehende Aufrufe."""
        detailed_results = self.analyze_single(text, models, **kwargs)
        aggregated = self.aggregate_scores(detailed_results) if detailed_results else {
            "positive": 1 / 3,
            "negative": 1 / 3,
            "neutral": 1 / 3,
        }
        return {"valence": aggregated, "details": detailed_results}

    def get_available_models(self) -> List[str]:
        """Gibt alle verfügbaren Modelle zurück."""
        return [name for name, analyzer in self.analyzers.items() if analyzer.is_available()]  # type: ignore[attr-defined]
