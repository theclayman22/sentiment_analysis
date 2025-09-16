"""Main content components for the Streamlit application."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from config.languages import get_text
from utils.data_loader import DataLoader
from utils.text_processor import TextProcessor


class MainContentUI:
    """Manage the central content area of the application."""

    def __init__(self, language: str = "DE") -> None:
        self.language = language
        self.data_loader = DataLoader()
        self.text_processor = TextProcessor()

    def render_header(self) -> None:
        """Render the main page header."""
        st.title(get_text("title", self.language))
        st.markdown(get_text("subtitle", self.language))
        st.divider()

    def render_input_section(self) -> Dict[str, Any]:
        """Render the input section and return the captured data."""
        st.subheader("📝 " + get_text("input_method", self.language))

        input_method = st.radio(
            get_text("input_method", self.language),
            options=["single_text", "batch_upload"],
            format_func=lambda key: get_text(key, self.language),
            horizontal=True,
            key="input_method",
        )

        if input_method == "single_text":
            return self._render_single_text_input()
        return self._render_batch_upload()

    def _render_single_text_input(self) -> Dict[str, Any]:
        """Render a text area for analysing a single text snippet."""
        st.markdown("#### " + get_text("text_input", self.language))

        text = st.text_area(
            get_text("text_input", self.language),
            placeholder=get_text("text_placeholder", self.language),
            height=200,
            key="single_text_input",
            label_visibility="collapsed",
        )

        if text:
            is_valid, error_msg = self.text_processor.validate_text(text)
            if not is_valid:
                st.error(f"❌ {error_msg}")
                return {"texts": [], "valid": False, "method": "single"}

            cleaned_text = self.text_processor.clean_text(text)
            st.success(f"✅ Text bereit ({len(cleaned_text)} Zeichen)")
            return {"texts": [cleaned_text], "valid": True, "method": "single"}

        return {"texts": [], "valid": False, "method": "single"}

    def _render_batch_upload(self) -> Dict[str, Any]:
        """Render the batch upload controls."""
        st.markdown("#### " + get_text("file_upload", self.language))

        uploaded_file = st.file_uploader(
            get_text("file_upload", self.language),
            type=["csv", "txt", "xlsx"],
            help=get_text("file_types", self.language),
            key="batch_file_upload",
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            return {"texts": [], "valid": False, "method": "batch"}

        with st.spinner("Datei wird verarbeitet..."):
            texts, error = self.data_loader.load_from_file(uploaded_file)

        if error:
            st.error(f"❌ {error}")
            return {"texts": [], "valid": False, "method": "batch"}

        if not texts:
            st.error("❌ Keine Texte in der Datei gefunden")
            return {"texts": [], "valid": False, "method": "batch"}

        valid_texts, errors = self.data_loader.validate_texts(texts)

        if errors:
            with st.expander("⚠️ Warnungen", expanded=False):
                for warning in errors:
                    st.warning(warning)

        if not valid_texts:
            st.error("❌ Keine gültigen Texte gefunden")
            return {"texts": [], "valid": False, "method": "batch"}

        st.success(f"✅ {len(valid_texts)} Texte erfolgreich geladen")

        with st.expander("👁️ Vorschau", expanded=False):
            preview_count = min(3, len(valid_texts))
            for index in range(preview_count):
                snippet = valid_texts[index][:200]
                suffix = "..." if len(valid_texts[index]) > 200 else ""
                st.text(f"Text {index + 1}: {snippet}{suffix}")
            if len(valid_texts) > preview_count:
                st.info(f"... und {len(valid_texts) - preview_count} weitere Texte")

        return {"texts": valid_texts, "valid": True, "method": "batch"}

    def render_analysis_button(self, input_data: Dict[str, Any], settings: Dict[str, Any]) -> bool:
        """Render the analyse button and return ``True`` when clicked."""
        st.divider()

        if input_data.get("valid") and input_data.get("texts"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(get_text("text_count", self.language), len(input_data["texts"]))

            with col2:
                model_count = len(settings.get("selected_models", []))
                st.metric(get_text("model_selection", self.language), model_count)

            with col3:
                analysis_type = settings.get("analysis_type", "valence")
                analysis_label = get_text(f"{analysis_type}_results", self.language)
                if analysis_label == f"{analysis_type}_results":
                    analysis_label = analysis_type.replace("_", " ").title()
                st.metric(get_text("analysis_type", self.language), analysis_label)

            button_disabled = (
                not input_data.get("valid")
                or not input_data.get("texts")
                or not settings.get("selected_models")
            )

            if st.button(
                get_text("analyze_button", self.language),
                type="primary",
                disabled=button_disabled,
                use_container_width=True,
                key="analyze_button",
            ):
                return True
        else:
            st.info("💡 " + get_text("error_no_text", self.language))

        return False

    def render_progress_section(self, total_tasks: int) -> Tuple[DeltaGenerator, DeltaGenerator]:
        """Render a progress bar and return the progress related widgets."""
        st.divider()
        st.subheader("⚡ " + get_text("analyzing", self.language))

        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(get_text("analyzing", self.language) + f" (0/{total_tasks})")
        return progress_bar, status_text

    def update_progress(
        self,
        progress_bar: DeltaGenerator,
        status_text: DeltaGenerator,
        current: int,
        total: int,
        task_description: str | None = None,
    ) -> None:
        """Update the visual progress information."""
        total = max(total, 1)
        progress_value = min(max(current / total, 0.0), 1.0)
        progress_bar.progress(progress_value)

        if task_description:
            status_text.text(f"{task_description} ({current}/{total})")
        else:
            status_text.text(
                get_text("analyzing", self.language) + f" ({current}/{total})"
            )
