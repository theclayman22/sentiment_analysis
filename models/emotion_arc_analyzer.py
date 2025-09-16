"""Emotion Arc Analyzer - Happiness Tracking über Textverlauf."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

try:  # pragma: no cover - optional dependency fallback
    from scipy.signal import find_peaks, savgol_filter
except Exception:  # pragma: no cover - SciPy optional
    find_peaks = None
    savgol_filter = None

try:  # pragma: no cover - optional dependency fallback
    from scipy.stats import linregress
except Exception:  # pragma: no cover - SciPy optional
    linregress = None

from analyzers.base_analyzer import AnalysisResult
from analyzers.deepseek_analyzer import DeepSeekAnalyzer
from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from analyzers.openai_analyzer import OpenAIAnalyzer
from analyzers.vader_analyzer import VADERAnalyzer
from config.settings import Settings
from utils.api_manager import APIManager
from utils.text_processor import TextProcessor


class EmotionArcAnalyzer:
    """Analysiert emotionale Bögen (Happiness-Tracking) über Textverlauf."""

    STORY_ARCHETYPES: Dict[str, Dict[str, str]] = {
        "rags_to_riches": {
            "name": "Rags to Riches",
            "description": "Aufstieg vom Unglück zum Glück",
            "pattern": "monotonic_rise",
        },
        "tragedy": {
            "name": "Tragedy",
            "description": "Fall vom Glück ins Unglück",
            "pattern": "monotonic_fall",
        },
        "man_in_hole": {
            "name": "Man in a Hole",
            "description": "Glück → Unglück → Glück",
            "pattern": "valley",
        },
        "icarus": {
            "name": "Icarus",
            "description": "Unglück → Glück → Unglück",
            "pattern": "peak",
        },
        "cinderella": {
            "name": "Cinderella",
            "description": "Komplex mit mehreren Wendungen (aufwärts)",
            "pattern": "rise_fall_rise",
        },
        "oedipus": {
            "name": "Oedipus",
            "description": "Komplex mit mehreren Wendungen (abwärts)",
            "pattern": "fall_rise_fall",
        },
    }

    def __init__(self) -> None:
        self.api_manager = APIManager()
        self.text_processor = TextProcessor()
        self.analyzers: Dict[str, object] = {}
        self._initialize_analyzers()

    def _initialize_analyzers(self) -> None:
        """Initialisiert verfügbare Analyzer für Emotion Arc."""
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
                self.analyzers["siebert/sentiment-roberta-large-english"] = HuggingFaceAnalyzer(
                    "siebert/sentiment-roberta-large-english", config
                )
        except Exception:  # pragma: no cover - defensive initialisation
            pass

        self.analyzers["vader"] = VADERAnalyzer()

    def analyze_arc(
        self, text: str, model: str = "apt-5-nano", n_segments: int = 20, **kwargs
    ) -> Dict[str, Any]:
        """Analysiert den emotionalen Bogen eines Textes."""
        segments = self.text_processor.extract_segments_for_arc(text, n_segments)
        if not segments:
            return {"error": "Konnte Text nicht segmentieren"}

        analyzer = self.analyzers.get(model) or self.analyzers.get("vader")
        if analyzer is None:
            return {"error": f"Modell {model} nicht verfügbar"}

        if not analyzer.is_available():  # type: ignore[attr-defined]
            return {"error": f"Modell {model} nicht verfügbar"}

        max_workers = min(Settings.MAX_WORKERS, len(segments)) or 1
        segment_scores = [0.5] * len(segments)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(analyzer.analyze_single, segment, "emotion_arc", **kwargs): idx  # type: ignore[attr-defined]
                for idx, segment in enumerate(segments)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result(timeout=Settings.REQUEST_TIMEOUT)
                    if not result.error and "happiness" in result.scores:
                        segment_scores[idx] = float(result.scores["happiness"])
                except Exception:
                    segment_scores[idx] = 0.5

        happiness_scores = np.array(segment_scores, dtype=float)
        arc_data = self._analyze_emotional_arc(happiness_scores, segments)

        return {
            "segments": segments,
            "happiness_scores": happiness_scores.tolist(),
            "arc_analysis": arc_data,
            "model_used": model if analyzer is not None else "vader",
            "n_segments": len(segments),
        }

    def _analyze_emotional_arc(
        self, happiness_scores: np.ndarray, segments: List[str]
    ) -> Dict[str, Any]:
        """Analysiert die emotionale Arc und klassifiziert das Muster."""
        if len(happiness_scores) >= 5:
            smoothed_scores = self._smooth_scores(happiness_scores)
        else:
            smoothed_scores = happiness_scores

        min_score, max_score = float(smoothed_scores.min()), float(smoothed_scores.max())
        if max_score - min_score > 0:
            normalized_scores = (smoothed_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.full_like(smoothed_scores, 0.5)

        features = self._extract_arc_features(normalized_scores)
        archetype, confidence = self._classify_archetype(normalized_scores, features)
        key_moments = self._find_key_moments(normalized_scores, segments)

        return {
            "raw_scores": happiness_scores.tolist(),
            "smoothed_scores": smoothed_scores.tolist(),
            "normalized_scores": normalized_scores.tolist(),
            "features": features,
            "archetype": archetype,
            "confidence": confidence,
            "key_moments": key_moments,
        }

    def _smooth_scores(self, scores: np.ndarray) -> np.ndarray:
        """Glättet Scores mit Savitzky-Golay oder gleitendem Mittel."""
        if savgol_filter is not None and len(scores) >= 5:
            window_length = min(len(scores) // 2, 7)
            window_length = max(3, window_length)
            if window_length % 2 == 0:
                window_length += 1
            try:
                return savgol_filter(scores, window_length, 2)  # type: ignore[no-any-return]
            except Exception:  # pragma: no cover - fallback auf gleitendes Mittel
                pass

        kernel_size = max(3, min(len(scores), 5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        padded = np.pad(scores, (pad, pad), mode="edge")
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[: len(scores)]

    def _extract_arc_features(self, scores: np.ndarray) -> Dict[str, Any]:
        """Extrahiert Features aus dem emotionalen Bogen."""
        if len(scores) < 3:
            return {"error": "Zu wenige Datenpunkte"}

        x = np.arange(len(scores))

        if linregress is not None and len(scores) > 1:
            slope, _, r_value, _, _ = linregress(x, scores)  # type: ignore[assignment]
            r_squared = float(r_value**2)
        elif len(scores) > 1:
            coeffs = np.polyfit(x, scores, 1)
            slope = float(coeffs[0])
            predicted = np.polyval(coeffs, x)
            ss_res = float(np.sum((scores - predicted) ** 2))
            ss_tot = float(np.sum((scores - scores.mean()) ** 2))
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            slope = 0.0
            r_squared = 0.0

        peaks = self._detect_extrema(scores, kind="peak")
        valleys = self._detect_extrema(scores, kind="valley")
        volatility = float(np.std(np.diff(scores))) if len(scores) > 1 else 0.0

        segment = max(1, len(scores) // 5)
        start_pos = self._classify_position(float(scores[:segment].mean()))
        end_pos = self._classify_position(float(scores[-segment:].mean()))

        return {
            "trend": "positive" if slope > 0.01 else "negative" if slope < -0.01 else "neutral",
            "slope": float(slope),
            "r_squared": r_squared,
            "n_peaks": int(len(peaks)),
            "n_valleys": int(len(valleys)),
            "volatility": volatility,
            "start_position": start_pos,
            "end_position": end_pos,
            "peak_positions": peaks.tolist(),
            "valley_positions": valleys.tolist(),
        }

    def _detect_extrema(self, scores: np.ndarray, kind: str) -> np.ndarray:
        """Findet Peaks oder Valleys im Score-Verlauf."""
        if find_peaks is not None:
            if kind == "peak":
                indices, _ = find_peaks(scores, prominence=0.15)
            else:
                indices, _ = find_peaks(-scores, prominence=0.15)
            return indices.astype(int)

        indices: List[int] = []
        comparator = (lambda a, b: a > b) if kind == "peak" else (lambda a, b: a < b)
        for idx in range(1, len(scores) - 1):
            if comparator(scores[idx], scores[idx - 1]) and comparator(scores[idx], scores[idx + 1]):
                indices.append(idx)
        return np.array(indices, dtype=int)

    def _classify_position(self, value: float) -> str:
        """Klassifiziert Position als low/medium/high."""
        if value < 0.33:
            return "low"
        if value < 0.67:
            return "medium"
        return "high"

    def _classify_archetype(
        self, scores: np.ndarray, features: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Klassifiziert den Archetyp basierend auf Features."""
        if "error" in features:
            return "unknown", 0.0

        archetype_scores: Dict[str, float] = {}

        if (
            features["trend"] == "positive"
            and features["n_peaks"] + features["n_valleys"] <= 1
            and features["start_position"] in {"low", "medium"}
            and features["end_position"] == "high"
        ):
            archetype_scores["rags_to_riches"] = 0.8 + 0.2 * features["r_squared"]

        if (
            features["trend"] == "negative"
            and features["n_peaks"] + features["n_valleys"] <= 1
            and features["start_position"] == "high"
            and features["end_position"] in {"low", "medium"}
        ):
            archetype_scores["tragedy"] = 0.8 + 0.2 * features["r_squared"]

        if (
            features["n_valleys"] == 1
            and features["n_peaks"] <= 1
            and features["start_position"] != "low"
            and features["end_position"] != "low"
        ):
            archetype_scores["man_in_hole"] = 0.7

        if (
            features["n_peaks"] == 1
            and features["n_valleys"] <= 1
            and features["start_position"] != "high"
            and features["end_position"] != "high"
        ):
            archetype_scores["icarus"] = 0.7

        if (
            features["n_peaks"] + features["n_valleys"] >= 2
            and features["trend"] == "positive"
            and features["end_position"] == "high"
        ):
            archetype_scores["cinderella"] = 0.6

        if (
            features["n_peaks"] + features["n_valleys"] >= 2
            and features["trend"] == "negative"
            and features["end_position"] == "low"
        ):
            archetype_scores["oedipus"] = 0.6

        if not archetype_scores:
            return "unknown", 0.0

        best_archetype = max(archetype_scores, key=archetype_scores.get)
        return best_archetype, float(archetype_scores[best_archetype])

    def _find_key_moments(self, scores: np.ndarray, segments: List[str]) -> List[Dict[str, Any]]:
        """Findet Schlüsselmomente (Peaks und Valleys) im Text."""
        moments: List[Dict[str, Any]] = []
        peak_indices = self._detect_extrema(scores, kind="peak")
        valley_indices = self._detect_extrema(scores, kind="valley")

        for idx in peak_indices:
            if idx < len(segments):
                preview = segments[idx]
                if len(preview) > 100:
                    preview = preview[:100] + "..."
                moments.append(
                    {
                        "type": "peak",
                        "position": int(idx),
                        "happiness": float(scores[idx]),
                        "text_preview": preview,
                    }
                )

        for idx in valley_indices:
            if idx < len(segments):
                preview = segments[idx]
                if len(preview) > 100:
                    preview = preview[:100] + "..."
                moments.append(
                    {
                        "type": "valley",
                        "position": int(idx),
                        "happiness": float(scores[idx]),
                        "text_preview": preview,
                    }
                )

        moments.sort(key=lambda item: item["position"])
        return moments

    def create_arc_visualization(self, arc_data: Dict[str, Any]) -> go.Figure:
        """Erstellt Plotly-Visualisierung des emotionalen Bogens."""
        if "error" in arc_data:
            fig = go.Figure()
            fig.add_annotation(
                text="Fehler bei der Arc-Analyse",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
            return fig

        analysis = arc_data["arc_analysis"]
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(analysis["raw_scores"]))),
                y=analysis["raw_scores"],
                mode="lines+markers",
                name="Raw Happiness",
                line=dict(color="lightblue", width=1),
                marker=dict(size=4),
                opacity=0.5,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(analysis["smoothed_scores"]))),
                y=analysis["smoothed_scores"],
                mode="lines",
                name="Emotional Arc",
                line=dict(color="blue", width=3),
            )
        )

        for moment in analysis["key_moments"]:
            fig.add_annotation(
                x=moment["position"],
                y=moment["happiness"],
                text="📈" if moment["type"] == "peak" else "📉",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green" if moment["type"] == "peak" else "red",
                ax=0,
                ay=-30 if moment["type"] == "peak" else 30,
            )

        archetype_info = ""
        archetype_key = analysis.get("archetype")
        if archetype_key and archetype_key in self.STORY_ARCHETYPES and archetype_key != "unknown":
            archetype_name = self.STORY_ARCHETYPES[archetype_key]["name"]
            confidence = float(analysis.get("confidence", 0.0)) * 100
            archetype_info = f" - {archetype_name} ({confidence:.0f}% Konfidenz)"

        fig.update_layout(
            title=f"Emotionaler Bogen{archetype_info}",
            xaxis_title="Text-Progression →",
            yaxis_title="Happiness Level",
            yaxis=dict(range=[0, 1]),
            height=500,
            hovermode="x unified",
        )

        return fig

    def get_available_models(self) -> List[str]:
        """Gibt verfügbare Modelle für Emotion Arc zurück."""
        return [name for name, analyzer in self.analyzers.items() if analyzer.is_available()]  # type: ignore[attr-defined]
