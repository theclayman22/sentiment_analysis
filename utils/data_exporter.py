"""Utilities to convert and export analysis results."""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from analyzers.base_analyzer import AnalysisResult

try:  # pragma: no cover - optional dependency
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit is optional during tests
    st = None  # type: ignore


class DataExporter:
    """Create export artefacts for sentiment analysis results."""

    def results_to_dataframe(
        self, results: List[Dict[str, AnalysisResult]], analysis_type: str
    ) -> pd.DataFrame:
        """Convert a nested result structure into a flat :class:`DataFrame`."""

        rows: List[Dict[str, Any]] = []

        for index, text_results in enumerate(results, start=1):
            base_row = {
                "text_id": index,
                "text": "",
                "text_length": 0,
                "analysis_type": analysis_type,
            }

            for result in text_results.values():
                if result.text:
                    base_row["text"] = result.text
                    base_row["text_length"] = len(result.text)
                    break

            for model_name, result in text_results.items():
                scores = result.scores or {}
                row = {
                    **base_row,
                    "model": model_name,
                    "processing_time": result.processing_time,
                    "error": result.error or "",
                }

                if analysis_type == "valence":
                    row.update(
                        {
                            "positive": scores.get("positive", 0.0),
                            "negative": scores.get("negative", 0.0),
                            "neutral": scores.get("neutral", 0.0),
                        }
                    )
                elif analysis_type == "ekman":
                    row.update(
                        {
                            "joy": scores.get("joy", 0.0),
                            "surprise": scores.get("surprise", 0.0),
                            "fear": scores.get("fear", 0.0),
                            "anger": scores.get("anger", 0.0),
                            "disgust": scores.get("disgust", 0.0),
                            "sadness": scores.get("sadness", 0.0),
                            "contempt": scores.get("contempt", 0.0),
                        }
                    )
                elif analysis_type == "emotion_arc":
                    row["happiness"] = scores.get("happiness", 0.0)

                rows.append(row)

        return pd.DataFrame(rows)

    def arc_to_dataframe(self, arc_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert emotion arc data to a :class:`DataFrame`."""

        if not arc_data or "error" in arc_data:
            return pd.DataFrame()

        segments = arc_data.get("segments", [])
        happiness_scores = arc_data.get("happiness_scores", [])
        analysis = arc_data.get("arc_analysis", {})

        rows: List[Dict[str, Any]] = []
        for index, (segment, happiness) in enumerate(zip(segments, happiness_scores), start=1):
            rows.append(
                {
                    "segment_id": index,
                    "segment_text": segment,
                    "happiness_score": happiness,
                    "model_used": arc_data.get("model_used", ""),
                    "archetype": analysis.get("archetype", ""),
                    "confidence": analysis.get("confidence", 0.0),
                }
            )

        df = pd.DataFrame(rows)

        key_moments = analysis.get("key_moments", []) or []
        for moment in key_moments:
            position = moment.get("position")
            if position is not None and 0 <= position < len(df):
                df.loc[df["segment_id"] == position + 1, "key_moment"] = moment.get("type", "")

        return df

    def export_to_csv(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """Serialise ``df`` to CSV and return the textual payload."""

        if df.empty:
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.csv"

        buffer = io.StringIO()
        df.to_csv(buffer, index=False, encoding="utf-8")
        buffer.seek(0)
        return buffer.getvalue()

    def export_to_excel(self, df: pd.DataFrame, filename: Optional[str] = None) -> bytes:
        """Serialise ``df`` to an Excel workbook and return the bytes payload."""

        if df.empty:
            return b""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.xlsx"

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)
        buffer.seek(0)
        return buffer.getvalue()

    def export_to_json(
        self, results: List[Dict[str, AnalysisResult]], filename: Optional[str] = None
    ) -> str:
        """Serialise analysis results to JSON."""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.json"

        json_payload: List[Dict[str, Any]] = []
        for index, text_results in enumerate(results, start=1):
            entry = {
                "text_id": index,
                "models": {},
            }
            for model_name, result in text_results.items():
                entry["models"][model_name] = {
                    "text": result.text,
                    "model": result.model,
                    "analysis_type": result.analysis_type,
                    "scores": result.scores,
                    "processing_time": result.processing_time,
                    "metadata": result.metadata,
                    "error": result.error,
                }
            json_payload.append(entry)

        return json.dumps(json_payload, indent=2, ensure_ascii=False)

    def create_download_button(
        self, data: Any, filename: str, mime_type: str, label: str
    ):
        """Create a Streamlit download button if Streamlit is available."""

        if st is None:  # pragma: no cover - Streamlit not installed during tests
            raise RuntimeError("Streamlit ist nicht installiert. Download-Button kann nicht erstellt werden.")

        return st.download_button(label=label, data=data, file_name=filename, mime=mime_type)


def export_csv(path: Path, rows: Iterable[dict]) -> None:
    """Backwards compatible helper to export dictionaries to CSV files."""

    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")
