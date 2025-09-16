"""Benchmark mode UI components."""

import streamlit as st


def show_benchmark(results: dict) -> None:
    """Display benchmark results."""

    st.subheader("Benchmark")
    st.json(results)
