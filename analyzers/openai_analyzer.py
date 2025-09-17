"""OpenAI-based analyzer that retrieves probabilities from the API when possible."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from analyzers.base_analyzer import AnalysisResult, BaseAnalyzer
from config.emotion_mappings import EKMAN_EMOTIONS


class OpenAIAnalyzer(BaseAnalyzer):
    """Analyzer that delegates emotion detection to OpenAI models."""

    def __init__(
        self, api_config: Optional[Any] = None, model_name: str = "gpt-5-nano"
    ) -> None:
        super().__init__(model_name=model_name, api_config=api_config)
        self._client = self._initialize_client()
        self._api_key_available = self._client is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_single(self, text: str, analysis_type: str, **kwargs) -> AnalysisResult:
        metadata = {
            "provider": "openai",
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
    def _initialize_client(self) -> Optional[OpenAI]:
        api_key = getattr(self.api_config, "primary_key", None)
        if not api_key:
            return None

        base_url = getattr(self.api_config, "base_url", None)
        try:
            if base_url:
                return OpenAI(api_key=api_key, base_url=base_url)
            return OpenAI(api_key=api_key)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Failed to initialize OpenAI client: %s", exc)
            return None

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

        return sanitized, True

    def _derive_happiness(self, valence_scores: Dict[str, float]) -> float:
        positive = max(0.0, valence_scores.get("positive", 0.0))
        neutral = max(0.0, valence_scores.get("neutral", 0.0))
        happiness = positive + 0.5 * neutral
        return max(0.0, min(1.0, happiness))

    def _request_model_scores(
        self, text: str, labels: List[str]
    ) -> Optional[Dict[str, float]]:
        if self._client is None:
            return None

        prompt = self._build_prompt(text, labels)
        try:
            response = self._client.responses.create(
                model=self.model_name,
                input=prompt,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("OpenAI API request failed: %s", exc)
            return None

        content = self._extract_text_from_response(response)
        if not content:
            return None

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            self.logger.error("Failed to decode OpenAI response as JSON: %s", content)
            return None

        if not isinstance(data, dict):
            return None

        return {label: float(data.get(label, 0.0)) for label in labels}

    def _build_prompt(self, text: str, labels: List[str]) -> str:
        label_list = ", ".join(labels)
        return (
            "You are an assistant that performs emotion intensity estimation.\n"
            f"Return a JSON object with independent intensity scores for the following labels: {label_list}.\n"
            "Each intensity must be a numeric value between 0 and 1; the scores do not need to sum to 1.\n"
            "Respond with JSON only without additional commentary.\n"
            f"Text: ```{text}```"
        )

    def _extract_text_from_response(self, response: Any) -> str:
        direct_text = getattr(response, "output_text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text.strip()

        response_dict = self._to_dict(response)
        if not response_dict:
            return ""

        output_text = response_dict.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = response_dict.get("output")
        if isinstance(output_items, list):
            parts: List[str] = []
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if isinstance(content, dict):
                        text_value = content.get("text")
                        if isinstance(text_value, str):
                            parts.append(text_value)
            if parts:
                joined = "".join(parts).strip()
                if joined:
                    return joined

        choices = response_dict.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

        return ""

    def _to_dict(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "model_dump"):
            try:
                data = response.model_dump()
                if isinstance(data, dict):
                    return data
            except Exception:  # pragma: no cover - defensive
                return {}
        if hasattr(response, "dict"):
            try:
                data = response.dict()
                if isinstance(data, dict):
                    return data
            except Exception:  # pragma: no cover - defensive
                return {}
        if isinstance(response, dict):
            return response
        return {}

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
            elif numeric > 1:
                numeric = 1.0
            sanitized[label] = numeric
            total += numeric

        if total <= 0:
            return None
        return sanitized

    def _zero_scores(self, labels: List[str]) -> Dict[str, float]:
        return {label: 0.0 for label in labels}


__all__ = ["OpenAIAnalyzer"]
