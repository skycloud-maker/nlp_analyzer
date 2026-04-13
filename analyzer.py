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
import re
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

# Post-processing in this module keeps LLM output usable for dashboards.

TOPIC_ALIASES = {
    "performance": "성능",
    "core performance": "성능",
    "convenience/usability": "사용 편의",
    "convenience": "사용 편의",
    "usability": "사용 편의",
    "a/s & support": "A/S",
    "as & support": "A/S",
    "a/s": "A/S",
    "as": "A/S",
    "delivery/installation": "배송/설치",
    "delivery/install": "배송/설치",
    "delivery/installlation": "배송/설치",
    "delivery": "배송/설치",
    "installation": "배송/설치",
    "maintenance/cleaning": "관리/청소",
    "maintenance": "관리/청소",
    "cleaning": "관리/청소",
    "price/value": "가격/가치",
    "price": "가격/가치",
    "value": "가격/가치",
    "product experience": "제품 경험",
    "general": "제품 경험",
    "durability": "내구성",
    "design": "디자인",
}

SUMMARY_FALLBACK_MESSAGES = {
    "요약 가능한 댓글 내용이 없습니다.",
    "한국어 번역을 준비 중입니다.",
    "분석 가능한 텍스트가 없습니다.",
}

SUMMARY_SCENARIO_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("구독", "가입", "권유", "매니저", "할부", "요금"), "구독 가입/권유 과정"),
    (("as", "a/s", "서비스", "수리", "기사", "출장", "보증"), "A/S·수리 대응"),
    (("배송", "설치", "기사 방문"), "배송·설치 과정"),
    (("소음", "진동", "탈수", "세척", "건조", "성능"), "핵심 성능 체감"),
    (("불편", "힘들", "무겁", "허리", "조작"), "사용 편의"),
    (("가격", "비용", "비싸", "전기세"), "가격·비용 부담"),
    (("옷감", "보풀", "엉킴", "손상"), "옷감 관리 경험"),
]

KEYWORD_STOPWORDS = {
    "있다", "있는데", "있어서", "있고", "없다", "없는데", "없어서", "없이",
    "샀다", "샀는데", "샀어요", "있음", "없음", "그냥", "진짜", "너무",
    "광고", "광고했다고", "유튜버", "유튜브", "채널", "리뷰",
    "그리고", "근데", "그래서", "때문", "정도", "그거", "이거",
}

KEYWORD_ALIASES = {
    "통돌": "통돌이",
    "통돌이는": "통돌이",
    "통돌이로": "통돌이",
    "통돌이랑": "통돌이",
    "통돌이를": "통돌이",
    "드럼세탁기": "드럼",
    "드럼으로": "드럼",
    "as": "as",
    "a/s": "as",
}

KOREAN_PARTICLE_SUFFIXES = (
    "으로", "로", "에서", "에게", "한테", "과", "와", "이", "가", "은", "는", "을", "를", "도", "만",
    "부터", "까지", "처럼", "보다", "마저", "조차", "라도", "이라도", "이나", "나",
)

NEGATIVE_MARKERS = ("불만", "불편", "문제", "고장", "최악", "별로", "환불", "짜증", "실망")
POSITIVE_MARKERS = ("만족", "좋", "추천", "최고", "편하", "잘되", "가성비", "훌륭")
INQUIRY_MARKERS = (
    "어떻게", "어디", "왜", "무엇", "뭐", "몇", "얼마", "문의", "질문",
    "가능한가", "가능할까요", "인가요", "될까요", "알려주세요", "확인해",
)
RHETORICAL_MARKERS = ("왜그럴까요", "말이되나요", "아닌가요", "맞나요", "되겠냐")

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+|[0-9A-Za-z가-힣/]+")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_korean_particles(token: str) -> str:
    stripped = token
    changed = True
    while changed:
        changed = False
        for suffix in KOREAN_PARTICLE_SUFFIXES:
            if stripped.endswith(suffix) and len(stripped) > len(suffix) + 1:
                stripped = stripped[: -len(suffix)]
                changed = True
                break
    return stripped


def _normalize_topic(topic: str) -> str:
    raw = _normalize_space(topic)
    if not raw:
        return ""
    lowered = raw.lower()
    mapped = TOPIC_ALIASES.get(lowered, raw)
    return _normalize_space(mapped)


def _normalize_topics(topics: list[str]) -> list[str]:
    normalized: list[str] = []
    for topic in topics:
        value = _normalize_topic(topic)
        if value and value not in normalized:
            normalized.append(value)
    return normalized[:3]


def _normalize_keyword(token: str) -> str:
    value = _normalize_space(token).lower()
    if not value:
        return ""
    value = value.strip(".,!?()[]{}\"'`“”‘’:;/-_")
    if not value:
        return ""
    if value in KEYWORD_STOPWORDS or value.isdigit():
        return ""
    value = KEYWORD_ALIASES.get(value, value)
    value = _strip_korean_particles(value)
    value = KEYWORD_ALIASES.get(value, value)
    if value in KEYWORD_STOPWORDS:
        return ""
    if len(value) < 2:
        return ""
    if re.fullmatch(r"[a-z]+", value) and value not in {"as"}:
        return ""
    if re.search(r"[^a-z0-9가-힣/+]", value):
        return ""
    return value


def _extract_keywords_from_text(raw_text: str, limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    ordered: list[str] = []
    for token in TOKEN_RE.findall(raw_text):
        normalized = _normalize_keyword(token)
        if not normalized:
            continue
        counter[normalized] += 1
        if normalized not in ordered:
            ordered.append(normalized)
    if not counter:
        return []
    ranked = sorted(counter.items(), key=lambda item: (-item[1], ordered.index(item[0])))
    return [keyword for keyword, _ in ranked[:limit]]


def _clean_keywords(raw_keywords: list[str], raw_text: str, limit: int = 5) -> list[str]:
    cleaned: list[str] = []
    for keyword in raw_keywords:
        normalized = _normalize_keyword(str(keyword))
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    if not cleaned:
        cleaned = _extract_keywords_from_text(raw_text, limit=limit)
    return cleaned[:limit]


def _is_summary_low_quality(summary: str, raw_text: str) -> bool:
    normalized_summary = _normalize_space(summary)
    if not normalized_summary:
        return True
    if normalized_summary in SUMMARY_FALLBACK_MESSAGES:
        return True
    if normalized_summary.endswith("..."):
        return True
    if len(normalized_summary) < 12:
        return True

    raw = _normalize_space(raw_text)
    if raw and (raw.startswith(normalized_summary) or normalized_summary in raw):
        similarity = SequenceMatcher(None, normalized_summary, raw[: len(normalized_summary)]).ratio()
        if similarity >= 0.9:
            return True
    return False


def _infer_summary_scenario(raw_text: str, topics: list[str], keywords: list[str]) -> str:
    lowered = _normalize_space(raw_text).lower()
    for markers, scenario in SUMMARY_SCENARIO_HINTS:
        if any(marker in lowered for marker in markers):
            return scenario
    if topics:
        return topics[0]
    if keywords:
        return f"{', '.join(keywords[:2])} 이슈"
    return "제품 사용 경험"


def _build_semantic_summary(
    raw_text: str,
    label: str,
    topics: list[str],
    keywords: list[str],
    is_inquiry: bool,
) -> str:
    scenario = _infer_summary_scenario(raw_text, topics, keywords)
    if is_inquiry:
        return f"{scenario} 관련 조건이나 해결 방법을 확인하려는 문의성 댓글입니다."
    if label == "negative":
        return f"{scenario} 과정에서 불편·불만을 제기한 댓글입니다."
    if label == "positive":
        return f"{scenario} 경험에 대한 만족 또는 추천 의사를 드러낸 댓글입니다."
    return f"{scenario} 관련 경험과 의견을 전달한 중립적 댓글입니다."


def _clean_summary(
    summary: str | None,
    raw_text: str,
    label: str,
    topics: list[str],
    keywords: list[str],
    is_inquiry: bool,
) -> str | None:
    if label in EMPTY_CONTENT_LABELS:
        return None
    candidate = _normalize_space(summary or "")
    if _is_summary_low_quality(candidate, raw_text):
        candidate = _build_semantic_summary(raw_text, label, topics, keywords, is_inquiry)
    return candidate


def _infer_inquiry_from_text(raw_text: str) -> bool:
    lowered = _normalize_space(raw_text).lower()
    if not lowered:
        return False
    if "?" in raw_text:
        return True
    if any(marker in lowered for marker in INQUIRY_MARKERS):
        if any(marker in lowered for marker in RHETORICAL_MARKERS):
            return False
        return True
    return False


def _calibrate_confidence(
    label: str,
    confidence: float,
    raw_text: str,
    topics: list[str],
    keywords: list[str],
    summary: str | None,
    is_inquiry: bool,
) -> float:
    score = max(0.0, min(1.0, float(confidence)))
    if label not in {"positive", "negative", "neutral"}:
        return score

    lowered = _normalize_space(raw_text).lower()
    if topics:
        score += 0.04
    if len(keywords) >= 2:
        score += 0.05
    if summary and len(summary) >= 16:
        score += 0.04
    if label == "negative" and any(marker in lowered for marker in NEGATIVE_MARKERS):
        score += 0.06
    if label == "positive" and any(marker in lowered for marker in POSITIVE_MARKERS):
        score += 0.06
    if label in {"positive", "negative"} and is_inquiry:
        score -= 0.03

    if label in {"positive", "negative"}:
        score = max(0.45, min(0.97, score))
    else:
        score = max(0.35, min(0.9, score))
    return score

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
    topics = [] if is_empty else _normalize_topics(parsed.get("topics", [])[:3])
    topic_sentiments = {} if is_empty else _sanitize_topic_sentiments(
        parsed.get("topic_sentiments", {}), topics
    )
    keywords = [] if is_empty else _clean_keywords(parsed.get("keywords", [])[:8], raw_text, limit=5)
    product_mentions = [] if is_empty else parsed.get("product_mentions", [])
    is_inquiry = False if is_empty else (
        bool(parsed.get("is_inquiry", False)) or _infer_inquiry_from_text(raw_text)
    )
    is_rhetorical = False if is_empty else bool(parsed.get("is_rhetorical", False))
    summary = _clean_summary(
        parsed.get("summary"),
        raw_text=raw_text,
        label=label,
        topics=topics,
        keywords=keywords,
        is_inquiry=is_inquiry,
    )

    # is_inquiry + is_rhetorical 상호 배타 보정
    if is_inquiry and is_rhetorical:
        is_rhetorical = False  # inquiry 우선

    confidence = _calibrate_confidence(
        label=label,
        confidence=float(parsed.get("confidence", 0.5)),
        raw_text=raw_text,
        topics=topics,
        keywords=keywords,
        summary=summary,
        is_inquiry=is_inquiry,
    )

    return AnalysisResult(
        comment_id=comment_id,
        raw_text=raw_text,
        language=language,
        label=label,
        confidence=confidence,
        sentiment_reason=_normalize_space(parsed.get("sentiment_reason", "")),
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
    normalized_raw: dict[str, str] = {}
    for raw_topic, raw_sentiment in (raw or {}).items():
        key = _normalize_topic(str(raw_topic))
        if key and key not in normalized_raw:
            normalized_raw[key] = str(raw_sentiment)

    result = {}
    for topic in topics:
        sentiment = normalized_raw.get(topic, "neutral")
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
