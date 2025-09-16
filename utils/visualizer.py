"""Plotting helpers."""

from typing import List
import plotly.express as px


def plot_valence(values: List[float]):
    """Return a simple line chart for values."""

    return px.line(y=values)
