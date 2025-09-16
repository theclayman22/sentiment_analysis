"""Tests for analyzer classes."""

from analyzers.openai_analyzer import OpenAIAnalyzer
from analyzers.deepseek_analyzer import DeepSeekAnalyzer
from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from analyzers.vader_analyzer import VaderAnalyzer


def test_analyzers_return_provider() -> None:
    text = "Hello"
    analyzers = [
        OpenAIAnalyzer(),
        DeepSeekAnalyzer(),
        HuggingFaceAnalyzer(),
        VaderAnalyzer(),
    ]
    for analyzer in analyzers:
        result = analyzer.analyze(text)
        assert "provider" in result
        assert result["text"] == text
