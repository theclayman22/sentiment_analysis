"""Integration tests for utilities and models."""

from pathlib import Path

from models.valence_analyzer import ValenceAnalyzer
from utils.data_loader import load_texts
from utils.text_processor import clean_text


def test_integration(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Hello World!\n")
    texts = load_texts(sample)
    assert texts == ["Hello World!"]

    processed = [clean_text(t) for t in texts]
    assert processed == ["hello world"]

    analyzer = ValenceAnalyzer()
    result = analyzer.analyze(processed[0])
    assert "valence" in result
