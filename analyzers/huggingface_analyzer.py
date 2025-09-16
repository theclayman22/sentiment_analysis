"""
HuggingFace Modelle Analyzer (BART, RoBERTa, SiEBERT)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
from transformers import pipeline

from analyzers.base_analyzer import AnalysisResult, BaseAnalyzer
from config.emotion_mappings import EKMAN_EMOTIONS, get_all_emotion_terms


class HuggingFaceAnalyzer(BaseAnalyzer):
    """HuggingFace Modelle Analyzer"""

    _FILL_MASK_MODELS = {
        "facebook/bart-large",
        "FacebookAI/roberta-base",
    }
    _SENTIMENT_MODELS = {
        "siebert/sentiment-roberta-large-english",
    }
    _DEFAULT_MODEL = "facebook/bart-large"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_config: Optional[Any] = None,
    ) -> None:
        super().__init__(model_name or self._DEFAULT_MODEL, api_config)
        self.pipeline: Optional[Any] = None
        self.tokenizer = None
        self._fallback_active = False
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialisiert das HuggingFace Modell"""
        try:
            device = 0 if torch.cuda.is_available() else -1

            pipeline_kwargs: Dict[str, Any] = {}
            token = getattr(self.api_config, "primary_key", None)
            if token:
                # transformers >= 4.37 verwendet den Parameter `token`
                pipeline_kwargs["token"] = token

            if self.model_name in self._FILL_MASK_MODELS:
                self.pipeline = pipeline(
                    task="fill-mask",
                    model=self.model_name,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device=device,
                    **pipeline_kwargs,
                )
            elif self.model_name in self._SENTIMENT_MODELS:
                self.pipeline = pipeline(
                    task="sentiment-analysis",
                    model=self.model_name,
                    device=device,
                    **pipeline_kwargs,
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Error initializing model %s: %s", self.model_name, exc)
            self.pipeline = None
            self._fallback_active = True

    def analyze_single(self, text: str, analysis_type: str, **kwargs) -> AnalysisResult:
        """Analysiert einen einzelnen Text"""
        metadata = {
            "provider": "huggingface",
            "model": self.model_name,
            "pipeline_available": bool(self.pipeline),
        }

        try:
            start_time = time.time()

            if analysis_type == "valence":
                scores = self._analyze_valence(text, **kwargs)
            elif analysis_type == "ekman":
                scores = self._analyze_ekman(text, **kwargs)
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
            self.logger.error("Error during analysis: %s", exc)
            fallback_scores = self._fallback_scores(analysis_type)
            return AnalysisResult(
                text=text,
                model=self.model_name,
                analysis_type=analysis_type,
                scores=fallback_scores,
                processing_time=0.0,
                metadata=metadata,
                error=str(exc),
            )

    def analyze_batch(self, texts: List[str], analysis_type: str, **kwargs) -> List[AnalysisResult]:
        """Analysiert eine Liste von Texten"""
        return [self.analyze_single(text, analysis_type, **kwargs) for text in texts]

    def is_available(self) -> bool:
        """Prüft, ob der Analyzer verfügbar ist"""
        return bool(self.pipeline) or self._fallback_active

    def _analyze_valence(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Valence mit HuggingFace Modellen"""
        target_emotions = ["positive", "negative", "neutral"]

        if self.model_name in self._SENTIMENT_MODELS and self.pipeline is not None:
            result = self.pipeline(text)
            if isinstance(result, list) and result:
                sentiment = result[0]
                label = sentiment.get("label", "").upper()
                score = float(sentiment.get("score", 0.0))
                if label == "POSITIVE":
                    return {
                        "positive": score,
                        "negative": max(0.0, 1 - score),
                        "neutral": 0.1,
                    }
                if label == "NEGATIVE":
                    return {
                        "positive": max(0.0, 1 - score),
                        "negative": score,
                        "neutral": 0.1,
                    }

        if self.model_name in self._FILL_MASK_MODELS:
            return self._fill_mask_analysis(text, target_emotions)

        return self._fallback_distribution(target_emotions)

    def _analyze_ekman(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Ekman-Emotionen mit Fill-Mask"""
        target_emotions = list(EKMAN_EMOTIONS.keys())

        if self.model_name in self._SENTIMENT_MODELS:
            return {emotion: 0.1 for emotion in target_emotions}

        if self.model_name in self._FILL_MASK_MODELS:
            return self._fill_mask_analysis(text, target_emotions)

        return self._fallback_distribution(target_emotions)

    def _analyze_happiness(self, text: str, **kwargs) -> Dict[str, float]:
        """Analysiert Happiness für Emotion Arc"""
        target_emotions = ["happiness"]

        if self.model_name in self._SENTIMENT_MODELS and self.pipeline is not None:
            result = self.pipeline(text)
            if isinstance(result, list) and result:
                sentiment = result[0]
                label = sentiment.get("label", "").upper()
                score = float(sentiment.get("score", 0.0))
                happiness = score if label == "POSITIVE" else max(0.0, 1 - score)
                return {"happiness": happiness}

        if self.model_name in self._FILL_MASK_MODELS:
            return self._fill_mask_analysis(text, target_emotions)

        return self._fallback_distribution(target_emotions)

    def _fill_mask_analysis(self, text: str, target_emotions: List[str]) -> Dict[str, float]:
        """Führt Fill-Mask Analyse für Emotionen durch"""
        if self.pipeline is None:
            return self._fallback_distribution(target_emotions)

        try:
            masked_text = f"This text makes me feel <mask>. {text}"

            emotion_terms: Dict[str, List[str]] = {}
            for emotion in target_emotions:
                emotion_terms[emotion] = [term.lower() for term in get_all_emotion_terms(emotion)]

            predictions = self.pipeline(masked_text, top_k=50)

            if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
                # Einige Versionen liefern eine Liste von Listen
                predictions = predictions[0]

            emotion_scores = {emotion: 0.0 for emotion in target_emotions}

            for pred in predictions:
                token = str(pred.get("token_str", "")).strip().lower()
                score = float(pred.get("score", 0.0))

                for emotion, terms in emotion_terms.items():
                    if token in terms:
                        emotion_scores[emotion] += score

            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {key: value / total_score for key, value in emotion_scores.items()}
            else:
                emotion_scores = self._fallback_distribution(target_emotions)

            return emotion_scores

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Error in fill-mask analysis: %s", exc)
            return self._fallback_distribution(target_emotions)

    def _fallback_distribution(self, target_emotions: List[str]) -> Dict[str, float]:
        if not target_emotions:
            return {}
        weight = 1.0 / len(target_emotions)
        return {emotion: weight for emotion in target_emotions}

    def _fallback_scores(self, analysis_type: str) -> Dict[str, float]:
        if analysis_type == "valence":
            return self._fallback_distribution(["positive", "negative", "neutral"])
        if analysis_type == "ekman":
            return self._fallback_distribution(list(EKMAN_EMOTIONS.keys()))
        if analysis_type == "emotion_arc":
            return {"happiness": 0.5}
        return {}
