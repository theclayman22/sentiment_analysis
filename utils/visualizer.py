"""Visualisation helpers for sentiment analysis results."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyzers.base_analyzer import AnalysisResult
from config.emotion_mappings import get_emotion_display_name


class SentimentVisualizer:
    """Create Plotly figures for different sentiment analysis use cases."""

    def __init__(self, language: str = "DE") -> None:
        self.language = language
        self.colors = {
            "positive": "#2E8B57",
            "negative": "#DC143C",
            "neutral": "#708090",
            "joy": "#FFD700",
            "surprise": "#FF69B4",
            "fear": "#4B0082",
            "anger": "#FF4500",
            "disgust": "#9ACD32",
            "sadness": "#4682B4",
            "contempt": "#8B4513",
        }

    def create_valence_comparison(self, results: Dict[str, AnalysisResult]) -> go.Figure:
        """Create a grouped bar chart comparing valence scores across models."""

        models: List[str] = []
        positive_scores: List[float] = []
        negative_scores: List[float] = []
        neutral_scores: List[float] = []

        for result in results.values():
            if result.error:
                continue
            models.append(result.model)
            positive_scores.append(result.scores.get("positive", 0.0))
            negative_scores.append(result.scores.get("negative", 0.0))
            neutral_scores.append(result.scores.get("neutral", 0.0))

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name=get_emotion_display_name("positive", self.language),
                x=models,
                y=positive_scores,
                marker_color=self.colors["positive"],
            )
        )
        fig.add_trace(
            go.Bar(
                name=get_emotion_display_name("negative", self.language),
                x=models,
                y=negative_scores,
                marker_color=self.colors["negative"],
            )
        )
        fig.add_trace(
            go.Bar(
                name=get_emotion_display_name("neutral", self.language),
                x=models,
                y=neutral_scores,
                marker_color=self.colors["neutral"],
            )
        )

        fig.update_layout(
            title="Valence-Vergleich zwischen Modellen",
            xaxis_title="Modelle",
            yaxis_title="Score",
            barmode="group",
            yaxis=dict(range=[0, 1]),
            height=400,
        )
        return fig

    def create_ekman_radar_chart(self, results: Dict[str, AnalysisResult]) -> go.Figure:
        """Create a radar chart visualising Ekman emotion scores."""

        emotions = ["joy", "surprise", "fear", "anger", "disgust", "sadness", "contempt"]
        emotion_labels = [get_emotion_display_name(emotion, self.language) for emotion in emotions]

        fig = go.Figure()
        for result in results.values():
            if result.error:
                continue
            scores = [result.scores.get(emotion, 0.0) for emotion in emotions]
            fig.add_trace(
                go.Scatterpolar(
                    r=scores,
                    theta=emotion_labels,
                    fill="toself",
                    name=result.model,
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Ekman-Emotionen Vergleich",
            height=500,
        )
        return fig

    def create_batch_overview(
        self, results: List[Dict[str, AnalysisResult]], analysis_type: str
    ) -> go.Figure:
        """Create an overview figure for batched analyses."""

        if analysis_type == "valence":
            return self._create_valence_batch_overview(results)
        if analysis_type == "ekman":
            return self._create_ekman_batch_overview(results)
        return go.Figure()

    def _create_valence_batch_overview(
        self, results: List[Dict[str, AnalysisResult]]
    ) -> go.Figure:
        """Create stacked line charts for batched valence analyses."""

        text_ids: List[int] = []
        positive_scores: List[float] = []
        negative_scores: List[float] = []
        models: List[str] = []

        for index, text_results in enumerate(results, start=1):
            for result in text_results.values():
                if result.error:
                    continue
                text_ids.append(index)
                positive_scores.append(result.scores.get("positive", 0.0))
                negative_scores.append(result.scores.get("negative", 0.0))
                models.append(result.model)

        if not text_ids:
            return go.Figure()

        df = pd.DataFrame(
            {
                "Text ID": text_ids,
                "Positive": positive_scores,
                "Negative": negative_scores,
                "Model": models,
            }
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Positive Scores", "Negative Scores"),
            shared_xaxes=True,
        )

        for model in df["Model"].unique():
            model_data = df[df["Model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data["Text ID"],
                    y=model_data["Positive"],
                    mode="lines+markers",
                    name=f"{model} (Positive)",
                    line=dict(color=self.colors["positive"]),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        for model in df["Model"].unique():
            model_data = df[df["Model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data["Text ID"],
                    y=model_data["Negative"],
                    mode="lines+markers",
                    name=f"{model} (Negative)",
                    line=dict(color=self.colors["negative"]),
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(title="Valence Scores über alle Texte", height=600, xaxis_title="Text ID")
        return fig

    def _create_ekman_batch_overview(
        self, results: List[Dict[str, AnalysisResult]]
    ) -> go.Figure:
        """Create multi subplot overview for Ekman emotions."""

        emotions = ["joy", "anger", "fear", "sadness"]
        titles = [get_emotion_display_name(emotion, self.language) for emotion in emotions]
        fig = make_subplots(rows=2, cols=2, subplot_titles=titles)

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for idx, emotion in enumerate(emotions):
            row, col = positions[idx]
            text_ids: List[int] = []
            emotion_scores: List[float] = []
            models: List[str] = []

            for index, text_results in enumerate(results, start=1):
                for result in text_results.values():
                    if result.error:
                        continue
                    text_ids.append(index)
                    emotion_scores.append(result.scores.get(emotion, 0.0))
                    models.append(result.model)

            if not text_ids:
                continue

            df = pd.DataFrame({"Text ID": text_ids, "Score": emotion_scores, "Model": models})
            for model in df["Model"].unique():
                model_data = df[df["Model"] == model]
                fig.add_trace(
                    go.Scatter(
                        x=model_data["Text ID"],
                        y=model_data["Score"],
                        mode="lines+markers",
                        name=model,
                        line=dict(color=self.colors.get(emotion, "#000000")),
                        showlegend=(idx == 0),
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(title="Ekman-Emotionen über alle Texte", height=600)
        return fig

    def create_model_performance_comparison(
        self, results: List[Dict[str, AnalysisResult]]
    ) -> go.Figure:
        """Create a figure comparing processing time and error rate across models."""

        if not results:
            return go.Figure()

        model_times: Dict[str, List[float]] = {}
        model_errors: Dict[str, int] = {}

        for text_results in results:
            for model_name, result in text_results.items():
                model_times.setdefault(model_name, []).append(result.processing_time)
                if result.error:
                    model_errors[model_name] = model_errors.get(model_name, 0) + 1
                else:
                    model_errors.setdefault(model_name, model_errors.get(model_name, 0))

        models = list(model_times.keys())
        avg_times = [float(np.mean(model_times[model])) if model_times[model] else 0.0 for model in models]
        error_rates = [model_errors.get(model, 0) / max(len(results), 1) * 100 for model in models]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Durchschnittliche Verarbeitungszeit", "Fehlerrate (%)"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        fig.add_trace(
            go.Bar(x=models, y=avg_times, name="Verarbeitungszeit (s)", marker_color="lightblue"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=models, y=error_rates, name="Fehlerrate (%)", marker_color="lightcoral"),
            row=1,
            col=2,
        )

        fig.update_layout(title="Modell-Performance Vergleich", height=400, showlegend=False)
        return fig
