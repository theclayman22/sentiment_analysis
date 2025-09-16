"""
Hauptanwendung - Sentiment Analysis Toolkit
Streamlit App für umfassende Sentiment- und Emotionsanalyse
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

# Pfad für lokale Imports hinzufügen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import der Module
from ui.sidebar import SidebarUI
from ui.main_content import MainContentUI
from ui.results_display import ResultsDisplayUI
from models.valence_analyzer import ValenceAnalyzer
from models.ekman_analyzer import EkmanAnalyzer
from models.emotion_arc_analyzer import EmotionArcAnalyzer
from config.languages import get_text
from utils.data_exporter import DataExporter
from utils.visualizer import SentimentVisualizer

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalysisApp:
    """Hauptanwendungsklasse für das Sentiment Analysis Toolkit"""

    def __init__(self) -> None:
        self.setup_page_config()
        self.initialize_session_state()

        # UI-Komponenten
        self.sidebar_ui = SidebarUI()
        self.main_content_ui: MainContentUI | None = None
        self.results_ui: ResultsDisplayUI | None = None

        # Analyzer
        self.valence_analyzer: Optional[ValenceAnalyzer] = None
        self.ekman_analyzer: Optional[EkmanAnalyzer] = None
        self.emotion_arc_analyzer: Optional[EmotionArcAnalyzer] = None

        # Visualizer
        self.visualizer: SentimentVisualizer | None = None

    def setup_page_config(self) -> None:
        """Konfiguriert Streamlit Seitenlayout"""
        st.set_page_config(
            page_title="Sentiment Analysis Toolkit",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                "Get Help": "https://github.com/your-repo/sentiment-toolkit",
                "Report a bug": "https://github.com/your-repo/sentiment-toolkit/issues",
                "About": (
                    """
                # Sentiment Analysis Toolkit
                Professionelle Sentiment- und Emotionsanalyse mit verschiedenen KI-Modellen.

                **Features:**
                - Valence-Analyse (Positiv/Negativ/Neutral)
                - Ekman-Emotionen mit Synonym-Clustering
                - Emotion Arc (Happiness-Tracking)
                - Benchmark-Modus für Modell-Vergleiche
                - Batch-Verarbeitung & Export
                """
                ),
            },
        )

    def initialize_session_state(self) -> None:
        """Initialisiert Session State Variablen"""
        if "language" not in st.session_state:
            st.session_state.language = "DE"

        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results: List[Dict[str, Any]] = []

        if "arc_results" not in st.session_state:
            st.session_state.arc_results: Dict[str, Any] = {}

        if "current_analysis_type" not in st.session_state:
            st.session_state.current_analysis_type = "valence"

        if "analysis_metadata" not in st.session_state:
            st.session_state.analysis_metadata = {}

    def initialize_components(self) -> None:
        """Initialisiert alle Komponenten basierend auf aktueller Sprache"""
        language = st.session_state.get("language", "DE")

        # UI-Komponenten
        self.main_content_ui = MainContentUI(language)
        self.results_ui = ResultsDisplayUI(language)
        self.visualizer = SentimentVisualizer(language)

        # Analyzer (Lazy Loading)
        if self.valence_analyzer is None:
            self.valence_analyzer = self._get_or_create_analyzer(
                "valence_analyzer", ValenceAnalyzer
            )

        if self.ekman_analyzer is None:
            self.ekman_analyzer = self._get_or_create_analyzer(
                "ekman_analyzer", EkmanAnalyzer
            )

        if self.emotion_arc_analyzer is None:
            self.emotion_arc_analyzer = self._get_or_create_analyzer(
                "emotion_arc_analyzer", EmotionArcAnalyzer
            )

        self._update_available_models_state()

    def run(self) -> None:
        """Hauptschleife der Anwendung"""
        try:
            # Komponenten initialisieren
            self.initialize_components()

            if not self.main_content_ui or not self.results_ui:
                st.error("UI-Komponenten konnten nicht initialisiert werden")
                return

            # Header rendern
            self.main_content_ui.render_header()

            # Sidebar rendern und Einstellungen holen
            settings = self.sidebar_ui.render()

            # Hauptinhalt rendern
            self.render_main_content(settings)

        except Exception as exc:  # pragma: no cover - Streamlit runtime guard
            st.error("Ein unerwarteter Fehler ist aufgetreten:")
            st.code(traceback.format_exc())
            logger.error("Unerwarteter Fehler in der Hauptschleife: %s", exc)

    def render_main_content(self, settings: Dict[str, Any]) -> None:
        """Rendert den Hauptinhalt basierend auf Einstellungen"""
        if not self.main_content_ui or not self.results_ui:
            return

        # Input-Sektion
        input_data = self.main_content_ui.render_input_section()

        # Analyse-Button und -durchführung
        if self.main_content_ui.render_analysis_button(input_data, settings):
            self.run_analysis(input_data, settings)

        # Ergebnisse anzeigen wenn vorhanden
        if (
            st.session_state.analysis_results
            and st.session_state.current_analysis_type != "emotion_arc"
        ):
            self.results_ui.render_results_section(
                st.session_state.analysis_results,
                st.session_state.current_analysis_type,
                st.session_state.analysis_metadata,
            )

        # Emotion Arc Ergebnisse separat behandeln
        if (
            st.session_state.arc_results
            and st.session_state.current_analysis_type == "emotion_arc"
        ):
            self.render_emotion_arc_results(st.session_state.arc_results, settings)

    def run_analysis(self, input_data: Dict[str, Any], settings: Dict[str, Any]) -> None:
        """Führt die Sentiment-Analyse durch"""
        texts = input_data.get("texts", [])
        analysis_type = settings.get("analysis_type", "valence")
        selected_models = settings.get("selected_models", [])
        language = st.session_state.get("language", "DE")

        if not input_data.get("valid") or not texts:
            st.error(get_text("error_no_text", language))
            return

        if not selected_models:
            st.error("Keine Texte oder Modelle ausgewählt")
            return

        # Session State aktualisieren
        st.session_state.current_analysis_type = analysis_type
        st.session_state.analysis_results = []
        st.session_state.arc_results = {}

        try:
            with st.spinner(get_text("analyzing", language)):
                if analysis_type == "emotion_arc":
                    arc_data = self.run_emotion_arc_analysis(texts, settings)
                    if arc_data:
                        st.session_state.analysis_metadata = {
                            "analysis_type": analysis_type,
                            "selected_models": selected_models,
                            "benchmark_mode": False,
                            **{key: value for key, value in settings.items() if key != "language"},
                        }
                else:
                    results = self.run_standard_analysis(texts, settings)
                    if results is not None:
                        st.session_state.analysis_metadata = {
                            "analysis_type": analysis_type,
                            "selected_models": selected_models,
                            "benchmark_mode": settings.get("benchmark_mode", False),
                            **{key: value for key, value in settings.items() if key != "language"},
                        }
        except Exception as exc:  # pragma: no cover - runtime guard
            st.error(f"Fehler bei der Analyse: {str(exc)}")
            logger.error("Fehler bei der Analyse: %s", exc)
            st.code(traceback.format_exc())

    def run_standard_analysis(
        self, texts: List[str], settings: Dict[str, Any]
    ) -> List[Dict[str, Any]] | None:
        """Führt Standard-Analyse (Valence oder Ekman) durch"""
        if not self.main_content_ui:
            return None

        analysis_type = settings.get("analysis_type")
        selected_models = settings.get("selected_models", [])

        analyzer: ValenceAnalyzer | EkmanAnalyzer | None
        if analysis_type == "valence":
            analyzer = self.valence_analyzer
        elif analysis_type == "ekman":
            analyzer = self.ekman_analyzer
        else:
            st.error(f"Unbekannter Analyse-Typ: {analysis_type}")
            return None

        if analyzer is None:
            st.error(f"Analyzer für {analysis_type} nicht verfügbar")
            return None

        # Progress-Anzeige
        total_tasks = max(len(texts) * max(len(selected_models), 1), 1)
        progress_bar, status_text = self.main_content_ui.render_progress_section(total_tasks)

        results: List[Dict[str, Any]] = []
        current_task = 0

        try:
            for index, text in enumerate(texts):
                if settings.get("benchmark_mode", False):
                    model_kwargs = self._get_model_specific_kwargs(settings)
                    text_results = analyzer.analyze_single(  # type: ignore[attr-defined]
                        text,
                        models=selected_models,
                        **model_kwargs,
                    )
                    current_task += len(selected_models)
                else:
                    model_name = selected_models[0]
                    model_kwargs = self._get_model_specific_kwargs(settings)
                    text_results = analyzer.analyze_single(  # type: ignore[attr-defined]
                        text,
                        models=[model_name],
                        **model_kwargs,
                    )
                    current_task += 1

                results.append(text_results)

                # Progress aktualisieren
                self.main_content_ui.update_progress(
                    progress_bar,
                    status_text,
                    current_task,
                    total_tasks,
                    f"Text {index + 1}/{len(texts)} analysiert",
                )

        except Exception as exc:  # pragma: no cover - runtime guard
            st.error(f"Fehler während der Analyse: {str(exc)}")
            logger.error("Fehler während der Standard-Analyse: %s", exc)
            return None

        # Ergebnisse speichern
        st.session_state.analysis_results = results

        # Erfolgsmeldung
        progress_bar.progress(1.0)
        status_text.text("✅ " + get_text("analysis_complete", st.session_state.language))
        st.success(f"🎉 {len(texts)} Texte erfolgreich analysiert!")

        return results

    def run_emotion_arc_analysis(
        self, texts: List[str], settings: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Führt Emotion Arc Analyse durch"""
        if not texts:
            st.error("Keine Texte für Emotion Arc Analyse verfügbar")
            return None

        if len(texts) != 1:
            st.error("Emotion Arc Analyse unterstützt nur einen Text pro Durchlauf")
            return None

        if not self.main_content_ui:
            return None

        if self.emotion_arc_analyzer is None:
            st.error("Emotion Arc Analyzer nicht verfügbar")
            return None

        text = texts[0]
        selected_models = settings.get("selected_models", [])

        if not selected_models:
            st.error("Kein Modell für Emotion Arc ausgewählt")
            return None

        model = selected_models[0]
        n_segments = settings.get("n_segments", 20)

        # Progress-Anzeige
        progress_bar, status_text = self.main_content_ui.render_progress_section(n_segments)

        try:
            status_text.text("Starte Emotion Arc Analyse...")

            arc_result = self.emotion_arc_analyzer.analyze_arc(
                text=text,
                model=model,
                n_segments=n_segments,
                **self._get_model_specific_kwargs(settings),
            )

            if arc_result.get("error"):
                st.error(f"Fehler bei der Arc-Analyse: {arc_result['error']}")
                return None

            arc_result["original_text"] = text
            arc_result["selected_model"] = model

            # Ergebnisse speichern
            st.session_state.arc_results = arc_result

            # Progress abschließen
            progress_bar.progress(1.0)
            status_text.text("✅ " + get_text("analysis_complete", st.session_state.language))
            st.success("🎉 Emotional Arc für Text erfolgreich analysiert!")

            return arc_result

        except Exception as exc:  # pragma: no cover - runtime guard
            st.error(f"Fehler bei der Emotion Arc Analyse: {str(exc)}")
            logger.error("Fehler bei der Emotion Arc Analyse: %s", exc)
            return None

    def render_emotion_arc_results(
        self, arc_data: Dict[str, Any], settings: Dict[str, Any]
    ) -> None:
        """Rendert Emotion Arc Ergebnisse"""
        st.divider()
        st.subheader("📈 " + get_text("emotion_arc_results", st.session_state.language))

        if "error" in arc_data:
            st.error(f"Fehler: {arc_data['error']}")
            return

        original_text = arc_data.get("original_text", "")
        if original_text:
            snippet = original_text[:400]
            suffix = "..." if len(original_text) > 400 else ""
            st.text_area(
                "Text",
                value=f"{snippet}{suffix}",
                height=160,
                disabled=True,
                key="emotion_arc_text_display",
            )

        # Arc-Visualisierung
        if self.emotion_arc_analyzer:
            try:
                figure = self.emotion_arc_analyzer.create_arc_visualization(arc_data)
                st.plotly_chart(figure, width="stretch")
            except Exception as exc:  # pragma: no cover - runtime guard
                st.error(f"Fehler bei der Visualisierung: {str(exc)}")
                logger.error("Fehler bei der Arc-Visualisierung: %s", exc)

        analysis = arc_data.get("arc_analysis", {})
        if not analysis:
            return

        # Arc-Details in Expander
        with st.expander("📊 Arc-Analyse Details", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                archetype = analysis.get("archetype", "unknown")
                if (
                    archetype != "unknown"
                    and self.emotion_arc_analyzer
                    and archetype in self.emotion_arc_analyzer.STORY_ARCHETYPES
                ):
                    archetype_info = self.emotion_arc_analyzer.STORY_ARCHETYPES[archetype]
                    st.metric(
                        get_text("arc_pattern", st.session_state.language),
                        archetype_info.get("name", archetype),
                    )
                else:
                    st.metric(get_text("arc_pattern", st.session_state.language), "Unbekannt")

            with col2:
                confidence = analysis.get("confidence", 0.0)
                st.metric(
                    get_text("arc_confidence", st.session_state.language),
                    f"{float(confidence) * 100:.1f}%",
                )

            with col3:
                key_moments = analysis.get("key_moments", []) or []
                st.metric(
                    get_text("key_moments", st.session_state.language),
                    len(key_moments),
                )

        # Key Moments Details
        key_moments = analysis.get("key_moments", []) or []
        if key_moments:
            with st.expander("🎭 " + get_text("key_moments", st.session_state.language)):
                for moment in key_moments:
                    moment_type = moment.get("type")
                    if moment_type == "peak":
                        label = "📈 Peak"
                    elif moment_type == "valley":
                        label = "📉 Valley"
                    else:
                        label = "" if moment_type is None else str(moment_type).title()
                    position = moment.get("position")
                    position_display = position + 1 if isinstance(position, int) else "-"
                    st.write(f"**{label}** (Position {position_display})")
                    happiness = moment.get("happiness")
                    if happiness is not None:
                        st.write(f"Happiness: {float(happiness):.3f}")
                    text_preview = moment.get("text_preview") or ""
                    if text_preview:
                        st.write(f"Text: *{text_preview}*")
                    st.divider()

        # Export für Arc-Daten
        exporter = DataExporter()
        arc_df = exporter.arc_to_dataframe(arc_data)
        if arc_df.empty:
            st.info("Keine Arc-Daten zum Exportieren verfügbar")
        else:
            csv_data = exporter.export_to_csv(arc_df)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_arc_{timestamp}.csv"

            st.download_button(
                label=get_text("export_csv", st.session_state.language),
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                width="stretch",
            )

    def _get_or_create_analyzer(
        self, cache_key: str, factory: Callable[[], Any]
    ) -> Optional[Any]:
        cached = st.session_state.get(cache_key)
        if cached is not None:
            return cached

        try:
            analyzer = factory()
        except Exception as exc:  # pragma: no cover - defensive initialisation
            logger.error("Fehler beim Initialisieren von %s: %s", cache_key, exc)
            return None

        st.session_state[cache_key] = analyzer
        return analyzer

    def _update_available_models_state(self) -> None:
        """Synchronisiert verfügbare Modelle mit dem Session State."""
        st.session_state["available_valence_models"] = (
            self.valence_analyzer.get_available_models()
            if self.valence_analyzer is not None
            else []
        )
        st.session_state["available_ekman_models"] = (
            self.ekman_analyzer.get_available_models()
            if self.ekman_analyzer is not None
            else []
        )
        st.session_state["available_emotion_arc_models"] = (
            self.emotion_arc_analyzer.get_available_models()
            if self.emotion_arc_analyzer is not None
            else []
        )

    def _get_model_specific_kwargs(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Holt modell-spezifische Parameter aus den Einstellungen"""
        kwargs: Dict[str, Any] = {}

        # OpenAI-spezifische Parameter
        if settings.get("reasoning_effort"):
            kwargs["reasoning_effort"] = settings["reasoning_effort"]

        if settings.get("verbosity"):
            kwargs["verbosity"] = settings["verbosity"]

        # Timeout & Batch-Size
        if settings.get("timeout"):
            kwargs["timeout"] = settings["timeout"]

        if settings.get("batch_size"):
            kwargs["batch_size"] = settings["batch_size"]

        return kwargs


def main() -> None:
    """Hauptfunktion - Entry Point der Anwendung"""
    try:
        app = SentimentAnalysisApp()
        app.run()
    except Exception as exc:  # pragma: no cover - runtime guard
        st.error("Kritischer Fehler beim Starten der Anwendung:")
        st.code(traceback.format_exc())
        logger.critical("Kritischer Fehler beim App-Start: %s", exc)


if __name__ == "__main__":
    main()
