from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzer import (
    _build_insight_summary,
    _build_semantic_summary,
    _calibrate_confidence,
    _clean_keywords,
    _clean_summary,
    _extract_point_layer,
    _infer_inquiry_from_text,
    _normalize_topics,
)


def test_normalize_topics_localizes_english_labels() -> None:
    topics = _normalize_topics(["Performance", "A/S & Support", "General", "Performance"])
    assert topics == ["성능", "A/S", "제품 경험"]


def test_clean_keywords_merges_topload_variants() -> None:
    cleaned = _clean_keywords(["통돌", "통돌이는", "통돌이로", "세탁기"], "통돌이는 통돌이로 쓰고 통돌 세탁기")
    assert cleaned[0] == "통돌이"
    assert cleaned.count("통돌이") == 1


def test_clean_summary_replaces_truncated_copy() -> None:
    raw = "구독 가입 과정에서 매니저가 지나치게 권유해서 불쾌했고 서비스 응대도 별로였습니다."
    copied = "구독 가입 과정에서 매니저가 지나치게 권유해서..."
    summary = _clean_summary(copied, raw, "negative", ["A/S"], ["구독", "권유"], False)
    assert summary is not None
    assert "..." not in summary
    assert "불편" in summary or "불만" in summary


def test_inquiry_inference_without_question_mark() -> None:
    assert _infer_inquiry_from_text("설치비가 별도인가요 알려주세요")
    assert not _infer_inquiry_from_text("서비스가 최악이라 다시는 안 씁니다")


def test_semantic_summary_for_inquiry() -> None:
    summary = _build_semantic_summary(
        raw_text="AS 신청은 어떻게 하나요",
        label="neutral",
        topics=["A/S"],
        keywords=["AS", "신청"],
        is_inquiry=True,
    )
    assert "문의성 댓글" in summary


def test_extract_point_layer_builds_context_tags_and_similarity_keys() -> None:
    raw = "구독 가입 과정에서 매니저 권유가 심하고 해지 안내가 불친절해서 불만입니다."
    layer = _extract_point_layer(
        raw_text=raw,
        topics=["A/S"],
        keywords=["구독", "권유", "해지"],
        label="negative",
        is_inquiry=False,
    )
    assert layer["core_points"]
    assert any(tag in layer["context_tags"] for tag in ["구독/계약", "불만 신호"])
    assert any(key.startswith("ctx:") for key in layer["similarity_keys"])
    assert any(key.startswith("pt:") for key in layer["similarity_keys"])


def test_confidence_returns_metadata_and_intensity() -> None:
    score, factors, breakdown, intensity = _calibrate_confidence(
        label="negative",
        confidence=0.52,
        raw_text="고장이 반복되고 소음이 심해서 불편합니다.",
        topics=["성능"],
        keywords=["고장", "소음"],
        summary="성능·품질 과정에서 고장, 소음 문제로 불편·불만을 제기하는 댓글입니다.",
        is_inquiry=False,
        core_points=["고장", "소음"],
        context_tags=["성능·품질", "불만 신호"],
        sentiment_reason="반복 고장과 소음 불만이 명확합니다.",
    )
    assert 0.0 <= score <= 1.0
    assert score >= 0.58
    assert factors
    assert "llm_base" in breakdown
    assert 0.0 <= intensity <= 1.0


def test_insight_summary_positive_supports_strength_hypothesis() -> None:
    insight = _build_insight_summary(
        raw_text="타사 대비 소음이 적고 추천하고 싶어요.",
        label="positive",
        scenario="비교 구매 상황",
        core_points=["소음", "추천"],
        context_tags=["비교 맥락", "추천 의도", "강점 신호"],
        is_inquiry=False,
    )
    assert insight is not None
    assert "강점" in insight or "추천" in insight
