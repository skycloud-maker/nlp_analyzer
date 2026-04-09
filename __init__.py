"""
nlp_analyzer

YouTube VoC 댓글 자연어 처리 모듈.
외부에서는 아래 두 함수만 사용하면 된다.

    from nlp_analyzer import analyze_comment, analyze_batch

    result = analyze_comment("yt_001", "냉장고 소음이 너무 심해요")
    results = analyze_batch([{"id": "yt_001", "text": "..."}])
"""

from .analyzer import analyze_comment, analyze_batch
from .models import AnalysisResult, make_error_result, validate, is_valid

__all__ = [
    "analyze_comment",
    "analyze_batch",
    "AnalysisResult",
    "make_error_result",
    "validate",
    "is_valid",
]
