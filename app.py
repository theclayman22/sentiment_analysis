"""Streamlit entrypoint."""

import streamlit as st

from ui.sidebar import render_sidebar
from ui.main_content import render_main


def main() -> None:
    """Run the Streamlit app."""

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
