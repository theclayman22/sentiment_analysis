"""Main content for the Streamlit app."""

import streamlit as st


def render_main() -> None:
    """Render main content."""

    st.title("Sentiment Analysis Toolkit")
    st.text_area("Enter text")
