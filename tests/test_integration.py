"""
Integration tests for utilities and models.
"""

from pathlib import Path

from models.valence_analyzer import ValenceAnalyzer
from utils.data_loader import load_texts
from utils.text_processor import TextProcessor


def test_integration(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Hello World!\n")
    texts = load_texts(sample)
    assert texts == ["Hello World!"]

    processor = TextProcessor()
    processed = [processor.clean_text(t) for t in texts]
    assert processed == ["Hello World!"]

    sentences = processor.split_into_sentences("Hello World! This is great.")
    assert len(sentences) >= 2

    is_valid, error = processor.validate_text("A sufficiently long sentence.")
    assert is_valid is True
    assert error is None

    chunks = processor.chunk_text("word " * 600, max_tokens=100, overlap=10)
    assert len(chunks) > 1

    segments = processor.extract_segments_for_arc("word " * 400, n_segments=10)
    assert len(segments) <= 10
    assert segments  # not empty

    analyzer = ValenceAnalyzer()
    result = analyzer.analyze(processed[0])
    assert "valence" in result
