"""Streamlit entrypoint for the Sentiment Analysis Toolkit."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import streamlit as st

from analyzers.base_analyzer import AnalysisResult
from config.languages import get_text
from config.settings import Settings
from models.ekman_analyzer import EkmanAnalyzer
from models.emotion_arc_analyzer import EmotionArcAnalyzer
from models.valence_analyzer import ValenceAnalyzer
from ui.main_content import MainContentUI
from ui.results_display import ResultsDisplayUI
from ui.sidebar import SidebarUI


def _get_valence_analyzer() -> ValenceAnalyzer:
    if "valence_analyzer" not in st.session_state:
        st.session_state["valence_analyzer"] = ValenceAnalyzer()
    return st.session_state["valence_analyzer"]


def _get_ekman_analyzer() -> EkmanAnalyzer:
    if "ekman_analyzer" not in st.session_state:
        st.session_state["ekman_analyzer"] = EkmanAnalyzer()
    return st.session_state["ekman_analyzer"]


def _get_emotion_arc_analyzer() -> EmotionArcAnalyzer:
    if "emotion_arc_analyzer" not in st.session_state:
        st.session_state["emotion_arc_analyzer"] = EmotionArcAnalyzer()
    return st.session_state["emotion_arc_analyzer"]


def _extract_analysis_kwargs(settings: Dict[str, Any]) -> Dict[str, Any]:
    keys = {"reasoning_effort", "verbosity", "timeout", "batch_size"}
    return {key: settings[key] for key in keys if key in settings}


def _run_emotion_arc_analysis(
    texts: List[str], settings: Dict[str, Any]
) -> List[Dict[str, AnalysisResult]]:
    analyzer = _get_emotion_arc_analyzer()
    n_segments = settings.get("n_segments", 20)
    selected_models = settings.get("selected_models", [])
    available_models = [
        model for model in selected_models if model in getattr(analyzer, "analyzers", {})
    ]
    if not available_models:
        available_models = list(getattr(analyzer, "analyzers", {}).keys())
    if not available_models:
        available_models = ["vader"]

    kwargs = _extract_analysis_kwargs(settings)
    results: List[Dict[str, AnalysisResult]] = []

    for text in texts:
        text_results: Dict[str, AnalysisResult] = {}
        for model in available_models:
            start_time = time.time()
            arc_data = analyzer.analyze_arc(
                text,
                model=model,
                n_segments=n_segments,
                **kwargs,
            )
            processing_time = time.time() - start_time

            if arc_data.get("error"):
                text_results[model] = AnalysisResult(
                    text=text,
                    model=model,
                    analysis_type="emotion_arc",
                    scores={},
                    processing_time=processing_time,
                    metadata=arc_data,
                    error=arc_data.get("error"),
                )
                continue

            happiness_scores = arc_data.get("happiness_scores", [])
            average_happiness = (
                sum(happiness_scores) / len(happiness_scores) if happiness_scores else 0.0
            )

            text_results[model] = AnalysisResult(
                text=text,
                model=model,
                analysis_type="emotion_arc",
                scores={"happiness": float(average_happiness)},
                processing_time=processing_time,
                metadata=arc_data,
            )
        results.append(text_results)

    return results


def _run_analysis(texts: List[str], settings: Dict[str, Any]) -> List[Dict[str, AnalysisResult]]:
    analysis_type = settings.get("analysis_type", "valence")
    selected_models = settings.get("selected_models", [])
    kwargs = _extract_analysis_kwargs(settings)

    if analysis_type == "valence":
        analyzer = _get_valence_analyzer()
        return analyzer.analyze_batch(texts, selected_models, **kwargs)
    if analysis_type == "ekman":
        analyzer = _get_ekman_analyzer()
        return analyzer.analyze_batch(texts, selected_models, **kwargs)
    if analysis_type == "emotion_arc":
        return _run_emotion_arc_analysis(texts, settings)
    return []


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Sentiment Analysis Toolkit", layout="wide")

    sidebar = SidebarUI()
    settings = sidebar.render()
    language = settings.get("language", Settings.DEFAULT_LANGUAGE)

    main_content = MainContentUI(language)
    main_content.render_header()
    input_data = main_content.render_input_section()

    should_run = main_content.render_analysis_button(input_data, settings)

    if should_run and input_data.get("valid"):
        with st.spinner(get_text_message(language)):
            texts = input_data.get("texts", [])
            results = _run_analysis(texts, settings)
        st.session_state["analysis_results"] = results
        st.session_state["analysis_metadata"] = {
            "analysis_type": settings.get("analysis_type", "valence"),
            "selected_models": settings.get("selected_models", []),
            "benchmark_mode": settings.get("benchmark_mode", False),
            **{key: value for key, value in settings.items() if key not in {"language"}},
        }

    stored_results = st.session_state.get("analysis_results", [])
    stored_metadata = st.session_state.get("analysis_metadata", {})

    if stored_results:
        results_ui = ResultsDisplayUI(language)
        analysis_type = stored_metadata.get("analysis_type", "valence")
        results_ui.render_results_section(stored_results, analysis_type, stored_metadata)


def get_text_message(language: str) -> str:
    """Return a language specific progress message."""
    return get_text("analyzing", language)


if __name__ == "__main__":
    main()
