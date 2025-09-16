"""Streamlit sidebar components for the Sentiment Analysis Toolkit."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from config.languages import get_text
from config.settings import Settings


class SidebarUI:
    """Render and manage all sidebar controls."""

    def __init__(self) -> None:
        self.language = st.session_state.get("language", Settings.DEFAULT_LANGUAGE)

    def render(self) -> Dict[str, Any]:
        """Render the full sidebar and return the collected settings."""
        settings: Dict[str, Any] = {}

        with st.sidebar:
            settings["language"] = self._render_language_selector()
            self.language = settings["language"]

            st.divider()
            settings["analysis_type"] = self._render_analysis_type_selector()

            st.divider()
            model_settings = self._render_model_selector(settings["analysis_type"])
            settings.update(model_settings)

            st.divider()
            advanced_settings = self._render_advanced_settings(
                settings["analysis_type"], settings.get("selected_models", [])
            )
            settings.update(advanced_settings)

            st.divider()
            self._render_info_section()

        return settings

    def _render_language_selector(self) -> str:
        """Render the language selector and update the session state if needed."""
        st.subheader("🌐 " + get_text("language_settings", self.language))

        languages = Settings.SUPPORTED_LANGUAGES
        current_language = st.session_state.get("language", Settings.DEFAULT_LANGUAGE)
        try:
            index = languages.index(current_language)
        except ValueError:
            index = 0

        language = st.selectbox(
            "Language / Sprache",
            options=languages,
            index=index,
            key="language_selector",
        )

        if language != current_language:
            st.session_state["language"] = language
            self.language = language
            st.rerun()

        return language

    def _render_analysis_type_selector(self) -> str:
        """Render the analysis type selector."""
        st.subheader("🎯 " + get_text("analysis_type", self.language))

        analysis_options = {
            "valence": get_text("valence_results", self.language),
            "ekman": get_text("ekman_results", self.language),
            "emotion_arc": get_text("emotion_arc_results", self.language),
        }

        return st.selectbox(
            get_text("analysis_type", self.language),
            options=list(analysis_options.keys()),
            format_func=lambda key: analysis_options.get(key, key.title()),
            key="analysis_type_selector",
        )

    def _render_model_selector(self, analysis_type: str) -> Dict[str, Any]:
        """Render the model selection widget depending on the analysis type."""
        st.subheader("🤖 " + get_text("model_selection", self.language))

        available_models = self._get_available_models_for_analysis(analysis_type)
        st.session_state.setdefault("benchmark_mode", False)
        benchmark_mode = st.checkbox(
            get_text("benchmark_mode", self.language),
            help=get_text("benchmark_description", self.language),
            key="benchmark_mode",
        )

        if benchmark_mode:
            selected_models = available_models
            if available_models:
                st.info(f"📊 {len(available_models)} Modelle ausgewählt für Benchmark")
            else:
                st.error("Keine Modelle für diesen Analyse-Typ verfügbar")
        else:
            if available_models:
                default_index = 0
                if "selected_models" in st.session_state:
                    try:
                        default_model = st.session_state["selected_models"][0]
                        default_index = available_models.index(default_model)
                    except (IndexError, ValueError):
                        default_index = 0

                selected_model = st.selectbox(
                    get_text("single_model", self.language),
                    options=available_models,
                    index=default_index,
                    format_func=lambda name: Settings.MODELS[name].display_name,
                    key="single_model_selector",
                )
                selected_models = [selected_model]
            else:
                st.error("Keine Modelle für diesen Analyse-Typ verfügbar")
                selected_models = []

        st.session_state["selected_models"] = selected_models

        return {
            "benchmark_mode": benchmark_mode,
            "selected_models": selected_models,
        }

    def _render_advanced_settings(
        self, analysis_type: str, selected_models: List[str]
    ) -> Dict[str, Any]:
        """Render advanced configuration controls."""
        settings: Dict[str, Any] = {}

        with st.expander("⚙️ Erweiterte Einstellungen"):
            if analysis_type == "emotion_arc":
                settings["n_segments"] = st.slider(
                    "Anzahl Segmente für Arc-Analyse",
                    min_value=10,
                    max_value=50,
                    value=20,
                    help="Mehr Segmente = detailliertere Analyse, aber langsamere Verarbeitung",
                )

            if any(model == "apt-5-nano" for model in selected_models):
                settings["reasoning_effort"] = st.selectbox(
                    "OpenAI Reasoning Effort",
                    options=["minimal", "low", "medium", "high"],
                    index=0,
                    help="Höhere Werte = bessere Qualität, aber langsamere Verarbeitung",
                )
                settings["verbosity"] = st.selectbox(
                    "OpenAI Verbosity",
                    options=["low", "medium", "high"],
                    index=0,
                )

            settings["batch_size"] = st.slider(
                "Batch-Größe",
                min_value=1,
                max_value=Settings.MAX_BATCH_SIZE,
                value=min(Settings.DEFAULT_BATCH_SIZE, Settings.MAX_BATCH_SIZE),
                help="Anzahl Texte die parallel verarbeitet werden",
            )

            settings["timeout"] = st.slider(
                "Timeout (Sekunden)",
                min_value=10,
                max_value=120,
                value=Settings.REQUEST_TIMEOUT,
                help="Maximale Wartezeit pro Anfrage",
            )

        return settings

    def _render_info_section(self) -> None:
        """Render the informational help section."""
        with st.expander("ℹ️ Information"):
            st.markdown(
                """
                **Sentiment Analysis Toolkit**

                **Verfügbare Modelle:**
                - 🧠 OpenAI GPT-5 Nano (apt-5-nano)
                - 🤖 DeepSeek Chat
                - 🤗 HuggingFace BART Large
                - 🤗 HuggingFace RoBERTa Base
                - 🤗 SiEBERT (Sentiment RoBERTa)
                - 📊 VADER (Klassisches Lexikon)

                **Analyse-Typen:**
                - **Valence**: Positiv/Negativ/Neutral
                - **Ekman**: 7 Basis-Emotionen mit Synonymen
                - **Emotion Arc**: Happiness-Tracking über Textverlauf

                **Features:**
                - Benchmark-Modus für Modell-Vergleiche
                - Batch-Verarbeitung für multiple Texte
                - CSV/Excel/JSON Export
                - Interaktive Visualisierungen
                """
            )

    def _get_available_models_for_analysis(self, analysis_type: str) -> List[str]:
        """Return the model identifiers supporting ``analysis_type``."""
        state_key_map = {
            "valence": "available_valence_models",
            "ekman": "available_ekman_models",
            "emotion_arc": "available_emotion_arc_models",
        }

        session_key = state_key_map.get(analysis_type)
        if session_key and session_key in st.session_state:
            session_models = st.session_state.get(session_key, [])
            return self._filter_models_by_capability(session_models, analysis_type)

        fallback: List[str] = []
        for model_name, model_config in Settings.MODELS.items():
            if analysis_type == "valence" and model_config.supports_valence:
                fallback.append(model_name)
            elif analysis_type == "ekman" and model_config.supports_ekman:
                fallback.append(model_name)
            elif analysis_type == "emotion_arc" and model_config.supports_emotion_arc:
                fallback.append(model_name)

        return fallback

    def _filter_models_by_capability(
        self, models: List[str], analysis_type: str
    ) -> List[str]:
        filtered: List[str] = []
        for model_name in models:
            model_config = Settings.MODELS.get(model_name)
            if not model_config:
                continue

            if analysis_type == "valence" and model_config.supports_valence:
                filtered.append(model_name)
            elif analysis_type == "ekman" and model_config.supports_ekman:
                filtered.append(model_name)
            elif analysis_type == "emotion_arc" and model_config.supports_emotion_arc:
                filtered.append(model_name)

        return filtered
