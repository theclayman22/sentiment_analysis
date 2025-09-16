"""Components to present analysis results within Streamlit."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from analyzers.base_analyzer import AnalysisResult
from config.emotion_mappings import get_emotion_display_name
from config.languages import get_text
from utils.data_exporter import DataExporter
from utils.visualizer import SentimentVisualizer


class ResultsDisplayUI:
    """Handle presentation of analysis outcomes to the user."""

    def __init__(self, language: str = "DE") -> None:
        self.language = language
        self.visualizer = SentimentVisualizer(language)
        self.exporter = DataExporter()

    def render_results_section(
        self,
        results: List[Dict[str, AnalysisResult]],
        analysis_type: str,
        settings: Dict[str, Any],
    ) -> None:
        """Render the entire results area for the current analysis."""
        st.divider()
        st.subheader("📊 " + get_text("results", self.language))

        if not results:
            st.warning("Keine Ergebnisse verfügbar")
            return

        self._render_results_overview(results, settings)

        if analysis_type == "valence":
            self._render_valence_results(results, settings)
        elif analysis_type == "ekman":
            self._render_ekman_results(results, settings)
        elif analysis_type == "emotion_arc":
            self._render_emotion_arc_results(results, settings)

        self._render_export_section(results, analysis_type)

    def _render_results_overview(
        self,
        results: List[Dict[str, AnalysisResult]],
        settings: Dict[str, Any],
    ) -> None:
        """Show a quick overview of processed texts and performance metrics."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Texte analysiert", len(results))

        with col2:
            model_count = len(settings.get("selected_models", []))
            st.metric("Modelle verwendet", model_count)

        with col3:
            total_time = 0.0
            count = 0
            for text_results in results:
                for result in text_results.values():
                    if result.processing_time:
                        total_time += result.processing_time
                        count += 1
            average_time = total_time / count if count else 0.0
            st.metric("⌀ Zeit pro Text", f"{average_time:.2f}s")

        with col4:
            error_count = 0
            total_count = 0
            for text_results in results:
                for result in text_results.values():
                    total_count += 1
                    if result.error:
                        error_count += 1
            error_rate = (error_count / total_count * 100) if total_count else 0.0
            st.metric("Fehlerrate", f"{error_rate:.1f}%")

    def _render_valence_results(
        self, results: List[Dict[str, AnalysisResult]], settings: Dict[str, Any]
    ) -> None:
        """Render result views for valence analyses."""
        st.markdown("### " + get_text("valence_results", self.language))

        if settings.get("benchmark_mode"):
            self._render_valence_benchmark(results)
        else:
            self._render_valence_single(results)

    def _render_valence_benchmark(
        self, results: List[Dict[str, AnalysisResult]]
    ) -> None:
        """Render comparison charts for each text in benchmark mode."""
        for index, text_results in enumerate(results):
            with st.expander(f"📄 Text {index + 1}", expanded=index < 3):
                first_result = next(iter(text_results.values()))
                text_preview = first_result.text or ""
                if text_preview:
                    snippet = text_preview[:200]
                    suffix = "..." if len(text_preview) > 200 else ""
                    st.text_area(
                        "Text",
                        value=f"{snippet}{suffix}",
                        height=100,
                        disabled=True,
                        key=f"valence_text_{index}",
                    )

                fig = self.visualizer.create_valence_comparison(text_results)
                st.plotly_chart(fig, use_container_width=True)
                self._render_valence_table(text_results, f"valence_table_{index}")

    def _render_valence_single(
        self, results: List[Dict[str, AnalysisResult]]
    ) -> None:
        """Render valence results for a single selected model."""
        if len(results) > 1:
            fig = self.visualizer.create_batch_overview(results, "valence")
            st.plotly_chart(fig, use_container_width=True)

        for index, text_results in enumerate(results):
            expanded = len(results) == 1
            with st.expander(f"📄 Text {index + 1}", expanded=expanded):
                first_result = next(iter(text_results.values()))
                text_preview = first_result.text or ""
                if text_preview:
                    snippet = text_preview[:300]
                    suffix = "..." if len(text_preview) > 300 else ""
                    st.text_area(
                        "Text",
                        value=f"{snippet}{suffix}",
                        height=120,
                        disabled=True,
                        key=f"single_valence_text_{index}",
                    )
                self._render_valence_table(text_results, f"single_valence_table_{index}")

    def _render_valence_table(
        self, text_results: Dict[str, AnalysisResult], key: str
    ) -> None:
        """Render a table summarising valence scores for ``text_results``."""
        table_data: List[Dict[str, Any]] = []

        for result in text_results.values():
            if result.error:
                table_data.append(
                    {
                        "Modell": result.model,
                        "Positiv": "❌",
                        "Negativ": "❌",
                        "Neutral": "❌",
                        "Zeit (s)": f"{result.processing_time:.2f}",
                        "Fehler": (result.error[:50] + "...") if len(result.error) > 50 else result.error,
                    }
                )
            else:
                table_data.append(
                    {
                        "Modell": result.model,
                        "Positiv": f"{result.scores.get('positive', 0):.3f}",
                        "Negativ": f"{result.scores.get('negative', 0):.3f}",
                        "Neutral": f"{result.scores.get('neutral', 0):.3f}",
                        "Zeit (s)": f"{result.processing_time:.2f}",
                        "Fehler": "",
                    }
                )

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, key=key)

    def _render_ekman_results(
        self, results: List[Dict[str, AnalysisResult]], settings: Dict[str, Any]
    ) -> None:
        """Render result views for Ekman emotion analyses."""
        st.markdown("### " + get_text("ekman_results", self.language))

        if settings.get("benchmark_mode"):
            for index, text_results in enumerate(results):
                with st.expander(f"📄 Text {index + 1}", expanded=index < 2):
                    first_result = next(iter(text_results.values()))
                    text_preview = first_result.text or ""
                    if text_preview:
                        snippet = text_preview[:200]
                        suffix = "..." if len(text_preview) > 200 else ""
                        st.text_area(
                            "Text",
                            value=f"{snippet}{suffix}",
                            height=100,
                            disabled=True,
                            key=f"ekman_text_{index}",
                        )

                    fig = self.visualizer.create_ekman_radar_chart(text_results)
                    st.plotly_chart(fig, use_container_width=True)
                    self._render_ekman_table(text_results, f"ekman_table_{index}")
        else:
            if len(results) > 1:
                fig = self.visualizer.create_batch_overview(results, "ekman")
                st.plotly_chart(fig, use_container_width=True)

            for index, text_results in enumerate(results):
                expanded = len(results) == 1
                with st.expander(f"📄 Text {index + 1}", expanded=expanded):
                    first_result = next(iter(text_results.values()))
                    text_preview = first_result.text or ""
                    if text_preview:
                        snippet = text_preview[:300]
                        suffix = "..." if len(text_preview) > 300 else ""
                        st.text_area(
                            "Text",
                            value=f"{snippet}{suffix}",
                            height=120,
                            disabled=True,
                            key=f"single_ekman_text_{index}",
                        )
                    self._render_ekman_table(text_results, f"single_ekman_table_{index}")

    def _render_ekman_table(
        self, text_results: Dict[str, AnalysisResult], key: str
    ) -> None:
        """Render a detailed table for Ekman emotion scores."""
        emotions = ["joy", "surprise", "fear", "anger", "disgust", "sadness", "contempt"]
        table_data: List[Dict[str, Any]] = []

        for result in text_results.values():
            row: Dict[str, Any] = {"Modell": result.model}
            if result.error:
                for emotion in emotions:
                    row[get_emotion_display_name(emotion, self.language)] = "❌"
                row["Zeit (s)"] = f"{result.processing_time:.2f}"
                row["Fehler"] = (
                    (result.error[:30] + "...") if len(result.error) > 30 else result.error
                )
            else:
                for emotion in emotions:
                    row[get_emotion_display_name(emotion, self.language)] = (
                        f"{result.scores.get(emotion, 0):.3f}"
                    )
                row["Zeit (s)"] = f"{result.processing_time:.2f}"
                row["Fehler"] = ""
            table_data.append(row)

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, key=key)

    def _render_emotion_arc_results(
        self, results: List[Dict[str, AnalysisResult]], settings: Dict[str, Any]
    ) -> None:
        """Render interactive summaries for emotion arc analyses."""
        st.markdown("### " + get_text("emotion_arc_results", self.language))

        if not results:
            st.info("Keine Emotion Arc Ergebnisse verfügbar")
            return

        for index, text_results in enumerate(results):
            expanded = index == 0
            with st.expander(f"📄 Text {index + 1}", expanded=expanded):
                base_result = next(iter(text_results.values()))
                text_preview = base_result.text or ""
                if text_preview:
                    snippet = text_preview[:300]
                    suffix = "..." if len(text_preview) > 300 else ""
                    st.text_area(
                        "Text",
                        value=f"{snippet}{suffix}",
                        height=120,
                        disabled=True,
                        key=f"emotion_arc_text_{index}",
                    )

                for model_name, result in text_results.items():
                    st.markdown(f"#### 🤖 {result.model}")
                    if result.error:
                        st.error(f"❌ {result.error}")
                        continue

                    avg_happiness = result.scores.get("happiness")
                    if avg_happiness is not None:
                        st.metric("⌀ Happiness", f"{avg_happiness:.3f}")

                    metadata = result.metadata or {}
                    arc_analysis = metadata.get("arc_analysis", {})
                    happiness_scores = metadata.get("happiness_scores", [])

                    if arc_analysis:
                        archetype = arc_analysis.get("archetype") or "-"
                        confidence = arc_analysis.get("confidence")
                        confidence_display = (
                            f"{confidence:.0%}"
                            if isinstance(confidence, (int, float))
                            else "-"
                        )
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(get_text("arc_pattern", self.language), archetype)
                        with col2:
                            st.metric(
                                get_text("arc_confidence", self.language),
                                confidence_display,
                            )

                    if happiness_scores:
                        df = pd.DataFrame(
                            {
                                "Segment": list(range(1, len(happiness_scores) + 1)),
                                "Happiness": happiness_scores,
                            }
                        ).set_index("Segment")
                        st.line_chart(df, height=220)

                    key_moments = []
                    if arc_analysis:
                        key_moments = arc_analysis.get("key_moments", []) or []
                    if key_moments:
                        st.markdown(f"**{get_text('key_moments', self.language)}:**")
                        for moment in key_moments:
                            position = moment.get("position")
                            moment_type = moment.get("type", "-")
                            position_display = (
                                position + 1 if isinstance(position, int) else "-"
                            )
                            st.write(f"• {moment_type} (Segment {position_display})")

                    arc_df = self.exporter.arc_to_dataframe(metadata)
                    if not arc_df.empty:
                        display_df = arc_df.copy()
                        if "segment_text" in display_df.columns:
                            texts = display_df["segment_text"].fillna("").astype(str)
                            display_df["Segment"] = texts.str.slice(0, 120)
                            mask = texts.str.len() > 120
                            display_df.loc[mask, "Segment"] = (
                                display_df.loc[mask, "Segment"] + "..."
                            )
                            display_df = display_df.drop(columns=["segment_text"])
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            key=f"emotion_arc_table_{index}_{model_name.replace('/', '_')}",
                        )

    def _render_export_section(
        self, results: List[Dict[str, AnalysisResult]], analysis_type: str
    ) -> None:
        """Render download buttons for multiple export formats."""
        st.divider()
        st.subheader("💾 " + get_text("export", self.language))

        df = self.exporter.results_to_dataframe(results, analysis_type)
        if df.empty:
            st.warning("Keine Daten zum Exportieren verfügbar")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = self.exporter.export_to_csv(df)
            csv_name = f"sentiment_analysis_{analysis_type}_{timestamp}.csv"
            st.download_button(
                label=get_text("export_csv", self.language),
                data=csv_data,
                file_name=csv_name,
                mime="text/csv",
            )

        with col2:
            excel_data = self.exporter.export_to_excel(df)
            excel_name = f"sentiment_analysis_{analysis_type}_{timestamp}.xlsx"
            st.download_button(
                label=get_text("export_excel", self.language),
                data=excel_data,
                file_name=excel_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col3:
            json_data = self.exporter.export_to_json(results)
            json_name = f"sentiment_analysis_{analysis_type}_{timestamp}.json"
            st.download_button(
                label="Export als JSON",
                data=json_data,
                file_name=json_name,
                mime="application/json",
            )
