"""Display analysis results."""

import streamlit as st


def show_results(results: dict) -> None:
    """Show analysis results."""

    st.json(results)
