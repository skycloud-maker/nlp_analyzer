"""
nlp_analyzer/analyzer.py

메인 진입점.
외부에서는 analyze_comment() 와 analyze_batch() 만 사용하면 된다.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

try:
    from .models import (
        AnalysisResult, make_error_result, validate, EMPTY_CONTENT_LABELS,
    )
    from .prompts import build_prompt, detect_language
    from .providers.base import LLMInterface, LLMResponse
    from .providers.claude import ClaudeProvider
except ImportError:
    from models import (
        AnalysisResult, make_error_result, validate, EMPTY_CONTENT_LABELS,
    )
    from prompts import build_prompt, detect_language
    from providers.base import LLMInterface, LLMResponse
    from providers.claude import ClaudeProvider


# ─────────────────────────────────────────
# 기본 Provider 설정
# ─────────────────────────────────────────

def _get_default_provider() -> LLMInterface:
    """기본 Provider 반환 (Claude Primary)."""
    return ClaudeProvider()


# ─────────────────────────────────────────
# 핵심 파싱 로직
# ─────────────────────────────────────────

def _parse_llm_response(
    response: LLMResponse,
    comment_id: str,
    raw_text: str,
) -> tuple[dict | None, str | None]:
    """
    LLMResponse 의 raw_text 를 dict 로 파싱.
    실패 시 (None, error_message) 반환.
    """
    text = response.raw_text.strip()

    # 마크다운 코드블록 제거
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(text)
        return parsed, None
    except json.JSONDecodeError as e:
        return None, f"JSON 파싱 실패: {e} | 원문 앞 200자: {response.raw_text[:200]}"


def _build_result(
    parsed: dict,
    comment_id: str,
    raw_text: str,
    language: str,
    provider: LLMInterface,
) -> AnalysisResult:
    """
    파싱된 dict 로부터 AnalysisResult 를 생성.
    LLM 이 규칙을 어긴 경우 자동 보정 후 반환.
    """
    label = parsed.get("label", "undecidable")

    # label 허용값 보정
    try:
        from .models import VALID_LABELS
    except ImportError:
        from models import VALID_LABELS
    if label not in VALID_LABELS:
        label = "undecidable"

    is_empty = label in EMPTY_CONTENT_LABELS

    # trash / undecidable 강제 보정
    topics = [] if is_empty else parsed.get("topics", [])[:3]
    topic_sentiments = {} if is_empty else _sanitize_topic_sentiments(
        parsed.get("topic_sentiments", {}), topics
    )
    summary = None if is_empty else parsed.get("summary")
    keywords = [] if is_empty else parsed.get("keywords", [])[:5]
    product_mentions = [] if is_empty else parsed.get("product_mentions", [])
    is_inquiry = False if is_empty else bool(parsed.get("is_inquiry", False))
    is_rhetorical = False if is_empty else bool(parsed.get("is_rhetorical", False))

    # is_inquiry + is_rhetorical 상호 배타 보정
    if is_inquiry and is_rhetorical:
        is_rhetorical = False  # inquiry 우선

    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))  # 범위 클램핑

    return AnalysisResult(
        comment_id=comment_id,
        raw_text=raw_text,
        language=language,
        label=label,
        confidence=confidence,
        sentiment_reason=parsed.get("sentiment_reason", ""),
        topics=topics,
        topic_sentiments=topic_sentiments,
        is_inquiry=is_inquiry,
        is_rhetorical=is_rhetorical,
        summary=summary,
        keywords=keywords,
        product_mentions=product_mentions,
        analyzed_at=datetime.utcnow(),
        llm_provider=provider.provider_name,
        model_name=provider.model_name,
        error=None,
    )


def _sanitize_topic_sentiments(
    raw: dict,
    topics: list[str],
) -> dict[str, str]:
    """
    topic_sentiments 값 보정.
    - 허용값(positive/negative/neutral) 외 값은 neutral 로 대체
    - topics 에 없는 키는 제거
    """
    try:
        from .models import VALID_TOPIC_SENTIMENTS
    except ImportError:
        from models import VALID_TOPIC_SENTIMENTS
    result = {}
    for topic in topics:
        sentiment = raw.get(topic, "neutral")
        if sentiment not in VALID_TOPIC_SENTIMENTS:
            sentiment = "neutral"
        result[topic] = sentiment
    return result


# ─────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────

def analyze_comment(
    comment_id: str,
    text: str,
    provider: Optional[LLMInterface] = None,
) -> AnalysisResult:
    """
    댓글 하나를 분석하여 AnalysisResult 를 반환한다.

    Args:
        comment_id : 원본 댓글 식별자
        text       : 분석할 댓글 텍스트
        provider   : LLM Provider (None 이면 ClaudeProvider 기본값 사용)

    Returns:
        AnalysisResult — 성공 시 분석 결과, 실패 시 error 필드에 메시지 포함

    Examples:
        result = analyze_comment("yt_001", "냉장고 소음이 너무 심해요")
        print(result.label)          # "negative"
        print(result.topic_sentiments)  # {"소음": "negative"}
    """
    _provider = provider or _get_default_provider()

    # 빈 텍스트 조기 반환
    if not text or not text.strip():
        return make_error_result(
            comment_id=comment_id,
            raw_text=text,
            error_message="빈 텍스트",
            llm_provider=_provider.provider_name,
            model_name=_provider.model_name,
        )

    # 언어 감지
    language = detect_language(text)

    # 프롬프트 생성
    system_prompt, user_message = build_prompt(text)

    # LLM 호출
    response = _provider.call(system_prompt, user_message)

    if not response.success:
        return make_error_result(
            comment_id=comment_id,
            raw_text=text,
            error_message=response.error or "LLM 호출 실패",
            llm_provider=response.provider,
            model_name=response.model_name,
        )

    # 파싱
    parsed, parse_error = _parse_llm_response(response, comment_id, text)

    if parsed is None:
        return make_error_result(
            comment_id=comment_id,
            raw_text=text,
            error_message=parse_error or "파싱 실패",
            llm_provider=response.provider,
            model_name=response.model_name,
        )

    # AnalysisResult 생성 (자동 보정 포함)
    result = _build_result(parsed, comment_id, text, language, _provider)

    return result


def analyze_batch(
    comments: list[dict],
    provider: Optional[LLMInterface] = None,
) -> list[AnalysisResult]:
    """
    댓글 목록을 순차적으로 분석.

    Args:
        comments : [{"id": "...", "text": "..."}, ...] 형식의 목록
        provider : LLM Provider (None 이면 기본값 사용)

    Returns:
        AnalysisResult 목록 (입력 순서 유지)

    Note:
        현재는 순차 처리. v0.2 에서 병렬 처리 지원 예정.
    """
    _provider = provider or _get_default_provider()
    results = []

    for comment in comments:
        comment_id = comment.get("id", "unknown")
        text = comment.get("text", "")
        result = analyze_comment(comment_id, text, provider=_provider)
        results.append(result)

    return results
