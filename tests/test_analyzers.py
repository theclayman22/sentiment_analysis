"""
Tests for analyzer classes.
"""

from analyzers.deepseek_analyzer import DeepSeekAnalyzer
from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from analyzers.vader_analyzer import VaderAnalyzer
from analyzers.base_analyzer import AnalysisResult
from analyzers.openai_analyzer import OpenAIAnalyzer
from config.emotion_mappings import EKMAN_EMOTIONS


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


def test_openai_ekman_scores_preserve_intensity(monkeypatch) -> None:
    analyzer = OpenAIAnalyzer()
    labels = list(EKMAN_EMOTIONS.keys())

    def fake_request(text: str, requested_labels):
        assert requested_labels == labels
        return {
            labels[0]: 1.5,  # Should clamp to 1.0
            labels[1]: -0.2,  # Should clamp to 0.0
            labels[2]: 0.6,
        }

    monkeypatch.setattr(analyzer, "_request_model_scores", fake_request)

    scores, available = analyzer._analyze_ekman("example text")

    assert available is True
    assert scores[labels[0]] == 1.0
    assert scores[labels[1]] == 0.0
    assert scores[labels[2]] == 0.6
    assert sum(scores.values()) > 1.0
    for label in labels[3:]:
        assert scores[label] == 0.0
