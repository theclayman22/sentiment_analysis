"""
Tests for analyzer classes.
"""

from analyzers.openai_analyzer import OpenAIAnalyzer
from analyzers.deepseek_analyzer import DeepSeekAnalyzer
from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from analyzers.vader_analyzer import VaderAnalyzer
from analyzers.base_analyzer import AnalysisResult


def test_analyzers_return_standard_result() -> None:
    text = "Hello"
    analyzers = [
        (OpenAIAnalyzer(), "openai"),
        (DeepSeekAnalyzer(), "deepseek"),
        (HuggingFaceAnalyzer(), "huggingface"),
        (VaderAnalyzer(), "vader"),
    ]
    for analyzer, provider in analyzers:
        result = analyzer.analyze_single(text, analysis_type="valence")
        assert isinstance(result, AnalysisResult)
        assert result.text == text
        assert result.analysis_type == "valence"
        assert result.scores
        assert all(isinstance(value, float) for value in result.scores.values())
        assert result.processing_time >= 0
        assert result.metadata is not None
        assert result.metadata.get("provider") == provider
        assert analyzer.is_available()

    batch_results = analyzers[0][0].analyze_batch([text, text], analysis_type="valence")
    assert len(batch_results) == 2
    assert all(isinstance(item, AnalysisResult) for item in batch_results)
