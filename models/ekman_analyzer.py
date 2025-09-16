"""Ekman-Emotionen Analyse Koordinator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

from analyzers.base_analyzer import AnalysisResult
from analyzers.deepseek_analyzer import DeepSeekAnalyzer
from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from analyzers.openai_analyzer import OpenAIAnalyzer
from config.emotion_mappings import EKMAN_EMOTIONS
from config.settings import Settings
from utils.api_manager import APIManager


class EkmanAnalyzer:
    """Koordiniert Ekman-Emotionen Analysen mit Synonym-Clustering."""

    def __init__(self) -> None:
        self.api_manager = APIManager()
        self.analyzers: Dict[str, object] = {}
        self._initialize_analyzers()

    def _initialize_analyzers(self) -> None:
        """Initialisiert verfügbare Analyzer für Ekman-Emotionen."""
        try:
            config = self.api_manager.get_api_config("openai_reasoning")
            if getattr(config, "primary_key", None):
                self.analyzers["gpt-5-nano"] = OpenAIAnalyzer(config)
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
                for model_name in ["facebook/bart-large", "FacebookAI/roberta-base"]:
                    self.analyzers[model_name] = HuggingFaceAnalyzer(model_name, config)
        except Exception:  # pragma: no cover - defensive initialisation
            pass

    def analyze_single(
        self, text: str, models: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, AnalysisResult]:
        """Analysiert einen Text mit ausgewählten Modellen."""
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
                        analyzer.analyze_single, text, "ekman", **kwargs  # type: ignore[attr-defined]
                    )
                    futures[future] = model_name

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result(timeout=Settings.REQUEST_TIMEOUT + 15)
                    results[model_name] = self._apply_synonym_clustering(result)
                except Exception as exc:  # pragma: no cover - defensive runtime guard
                    results[model_name] = AnalysisResult(
                        text=text,
                        model=model_name,
                        analysis_type="ekman",
                        scores={emotion: 0.0 for emotion in EKMAN_EMOTIONS},
                        processing_time=0.0,
                        error=str(exc),
                    )

        return results

    def analyze_batch(
        self, texts: List[str], models: Optional[List[str]] = None, **kwargs
    ) -> List[Dict[str, AnalysisResult]]:
        """Analysiert mehrere Texte sequenziell."""
        return [self.analyze_single(text, models, **kwargs) for text in texts]

    def _apply_synonym_clustering(self, result: AnalysisResult) -> AnalysisResult:
        """Wendet Synonym-Clustering auf Analyseergebnisse an."""
        if result.error:
            return result

        enhanced_scores: Dict[str, float] = {}
        for emotion_key, emotion_data in EKMAN_EMOTIONS.items():
            base_score = float(result.scores.get(emotion_key, 0.0))
            synonym_boost = 0.0

            for synonym in emotion_data["synonyms"]:
                if synonym in result.scores and synonym not in EKMAN_EMOTIONS:
                    synonym_boost += float(result.scores[synonym]) * 0.3

            combined_score = base_score + synonym_boost
            enhanced_scores[emotion_key] = max(0.0, min(1.0, combined_score))

        metadata = dict(result.metadata or {})
        metadata.update({
            "original_scores": result.scores,
            "synonym_clustering_applied": True,
        })

        return AnalysisResult(
            text=result.text,
            model=result.model,
            analysis_type=result.analysis_type,
            scores=enhanced_scores,
            processing_time=result.processing_time,
            metadata=metadata,
            error=result.error,
        )

    def get_available_models(self) -> List[str]:
        """Gibt verfügbare Modelle zurück."""
        return [name for name, analyzer in self.analyzers.items() if analyzer.is_available()]  # type: ignore[attr-defined]

    def get_aggregated_scores(self, results: Dict[str, AnalysisResult]) -> Dict[str, float]:
        """Aggregiert Scores über alle Modelle hinweg."""
        emotion_scores: Dict[str, List[float]] = {emotion: [] for emotion in EKMAN_EMOTIONS}

        for result in results.values():
            if result.error:
                continue
            for emotion, score in result.scores.items():
                if emotion in emotion_scores:
                    emotion_scores[emotion].append(float(score))

        aggregated: Dict[str, float] = {}
        for emotion, values in emotion_scores.items():
            aggregated[emotion] = float(np.mean(values)) if values else 0.0

        return aggregated
