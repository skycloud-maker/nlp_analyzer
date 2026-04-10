"""
nlp_analyzer/analyzer.py

메인 진입점.
외부에서는 analyze_comment() 와 analyze_batch() 만 사용하면 된다.
"""

from __future__ import annotations

# ─────────────────────────────────────────
# .env 자동 로드 (로컬 실행 시)
# ─────────────────────────────────────────
import os as _os
from pathlib import Path as _Path

def _load_env():
    def _read_env(path):
        if path.exists():
            with open(path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _key, _, _value = _line.partition("=")
                        _os.environ.setdefault(_key.strip(), _value.strip())

    # 1) 현재 모듈 위치의 .env
    _read_env(_Path(__file__).parent / ".env")

    # 2) git worktree 에서 실행 중이면 메인 레포 루트의 .env 도 탐색
    git_common = _Path(__file__).parent / ".git"
    if git_common.is_file():
        # .git 이 파일이면 worktree — "gitdir: <path>" 형식
        try:
            content = git_common.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                git_dir = _Path(content.split(":", 1)[1].strip())
                # .git/worktrees/<name> → 메인 레포 루트는 2단계 위
                main_root = git_dir.resolve().parent.parent.parent
                _read_env(main_root / ".env")
        except Exception:
            pass

_load_env()

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
    """
    기본 Provider 반환.
    LLMRouter 를 통해 Claude → OpenAI 순서로 자동 전환.
    Claude 키 없으면 OpenAI 로 자동 사용.
    """
    try:
        from .router import LLMRouter
    except ImportError:
        from router import LLMRouter
    return LLMRouter()


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
    llm_provider: str,
    model_name: str,
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
        llm_provider=llm_provider,
        model_name=model_name,
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
    result = _build_result(parsed, comment_id, text, language, response.provider, response.model_name)

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
