"""DeepSeek-based analyzer that queries the API when available."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from analyzers.base_analyzer import AnalysisResult, BaseAnalyzer
from config.emotion_mappings import EKMAN_EMOTIONS


class DeepSeekAnalyzer(BaseAnalyzer):
    """Analyzer that delegates emotion detection to DeepSeek models."""

    def __init__(
        self, api_config: Optional[Any] = None, model_name: str = "deepseek-chat"
    ) -> None:
        super().__init__(model_name=model_name, api_config=api_config)
        self._api_key = getattr(self.api_config, "primary_key", "") or ""
        self._base_url = getattr(self.api_config, "base_url", "") or "https://api.deepseek.com"
        self._timeout = getattr(self.api_config, "timeout", 30)
        self._session, self._endpoint = self._initialize_session()
        self._api_key_available = self._session is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_single(self, text: str, analysis_type: str, **kwargs) -> AnalysisResult:
        metadata = {
            "provider": "deepseek",
            "model": self.model_name,
            "api_key_available": self._api_key_available,
        }

        start_time = time.time()
        if analysis_type == "valence":
            scores, available = self._analyze_valence(text)
        elif analysis_type == "ekman":
            scores, available = self._analyze_ekman(text)
        elif analysis_type == "emotion_arc":
            valence_scores, available = self._analyze_valence(text)
            scores = {"happiness": self._derive_happiness(valence_scores)}
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

        metadata["analysis_available"] = available
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
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_session(self) -> Tuple[Optional[requests.Session], Optional[str]]:
        if not self._api_key or not self._base_url:
            return None, None

        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
        )
        endpoint = f"{self._base_url.rstrip('/')}/chat/completions"
        return session, endpoint

    def _analyze_valence(self, text: str) -> Tuple[Dict[str, float], bool]:
        labels = ["positive", "negative", "neutral"]
        return self._generate_scores(text, labels)

    def _analyze_ekman(self, text: str) -> Tuple[Dict[str, float], bool]:
        labels = list(EKMAN_EMOTIONS.keys())
        return self._generate_scores(text, labels)

    def _generate_scores(
        self, text: str, labels: List[str]
    ) -> Tuple[Dict[str, float], bool]:
        raw_scores = self._request_model_scores(text, labels)
        if raw_scores is None:
            return self._zero_scores(labels), False

        sanitized = self._sanitize_scores(raw_scores, labels)
        if sanitized is None:
            return self._zero_scores(labels), False

        normalised = self._normalise_scores(sanitized)
        return normalised, True

    def _derive_happiness(self, valence_scores: Dict[str, float]) -> float:
        positive = max(0.0, valence_scores.get("positive", 0.0))
        neutral = max(0.0, valence_scores.get("neutral", 0.0))
        happiness = positive + 0.4 * neutral
        return max(0.0, min(1.0, happiness))

    def _request_model_scores(
        self, text: str, labels: List[str]
    ) -> Optional[Dict[str, float]]:
        if self._session is None or self._endpoint is None:
            return None

        prompt = self._build_prompt(text, labels)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an assistant that returns only JSON objects with emotion probabilities.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        try:
            response = self._session.post(
                self._endpoint,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - defensive
            self.logger.error("DeepSeek API request failed: %s", exc)
            return None

        try:
            data = response.json()
        except ValueError:  # pragma: no cover - defensive
            self.logger.error("DeepSeek API returned a non-JSON payload")
            return None

        message_content = self._extract_message_content(data)
        if not message_content:
            return None

        try:
            parsed = json.loads(message_content)
        except json.JSONDecodeError:
            self.logger.error("Failed to decode DeepSeek response as JSON: %s", message_content)
            return None

        if not isinstance(parsed, dict):
            return None

        return {label: float(parsed.get(label, 0.0)) for label in labels}

    def _build_prompt(self, text: str, labels: List[str]) -> str:
        label_list = ", ".join(labels)
        return (
            "Estimate probability scores for the following emotions: "
            f"{label_list}. Return a JSON object with numeric probabilities between 0 and 1 that sum to 1. "
            "Do not include explanations."
            f" Text: ```{text}```"
        )

    def _extract_message_content(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        return ""

    def _sanitize_scores(
        self, raw_scores: Dict[str, float], labels: List[str]
    ) -> Optional[Dict[str, float]]:
        sanitized: Dict[str, float] = {}
        total = 0.0
        for label in labels:
            value = raw_scores.get(label, 0.0)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = 0.0
            if numeric < 0:
                numeric = 0.0
            sanitized[label] = numeric
            total += numeric

        if total <= 0:
            return None
        return sanitized

    def _normalise_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        total = sum(scores.values())
        if total <= 0:
            return {key: 0.0 for key in scores}
        return {key: value / total for key, value in scores.items()}

    def _zero_scores(self, labels: List[str]) -> Dict[str, float]:
        return {label: 0.0 for label in labels}


__all__ = ["DeepSeekAnalyzer"]
