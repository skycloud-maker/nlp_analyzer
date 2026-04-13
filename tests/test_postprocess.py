from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzer import (
    _build_semantic_summary,
    _clean_keywords,
    _clean_summary,
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
    assert "불편·불만" in summary


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
