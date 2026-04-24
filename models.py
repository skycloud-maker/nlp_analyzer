"""
nlp_analyzer/models.py

AnalysisResult — 댓글 분석 결과 구조체
모든 NLP 분석의 입력/출력 계약을 정의하는 단일 진실 공급원(Single Source of Truth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────
# 허용 값 상수
# ─────────────────────────────────────────

VALID_LABELS = frozenset({"positive", "negative", "neutral", "trash", "undecidable"})
VALID_TOPIC_SENTIMENTS = frozenset({"positive", "negative", "neutral"})
VALID_PROVIDERS = frozenset({"claude", "openai", "mock", "echo", "fail", "none"})

# trash / undecidable 이면 topics, keywords 등이 비어야 하는 레이블
EMPTY_CONTENT_LABELS = frozenset({"trash", "undecidable"})


# ─────────────────────────────────────────
# 메인 구조체
# ─────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    댓글 하나의 NLP 분석 결과.

    설계 원칙:
    - label 과 is_inquiry / is_rhetorical 은 독립적인 두 개의 축
    - topic_sentiments 는 label 의 근거이며, 동점 시 negative 우선
    - is_inquiry / is_rhetorical 은 상호 배타적 (둘 다 True 불가)
    - trash / undecidable 이면 topics, topic_sentiments, keywords, summary 는 비어야 함
    - error 가 있으면 label 은 "undecidable" 로 고정
    """

    # ── 식별 ────────────────────────────────────
    comment_id: str
    raw_text: str
    language: str                           # "ko" | "en" | "unknown"

    # ── 감성 레이블 ─────────────────────────────────
    label: str                              # positive | negative | neutral | trash | undecidable
    confidence: float                       # 0.0 ~ 1.0

    # ── 감성 근거 ────────────────────────────────────
    sentiment_reason: str                   # 레이블 판단 근거 한 문장

    # ── 토픽 ──────────────────────────────────────────
    topics: list[str] = field(default_factory=list)             # 최대 3개
    topic_sentiments: dict[str, str] = field(default_factory=dict)
    # {"디자인": "positive", "AS": "negative"}
    # 값은 VALID_TOPIC_SENTIMENTS 만 허용
    # trash / undecidable → 빈 dict

    # ── 문의 / 수사적 질문 ───────────────────────────────
    is_inquiry: bool = False
    # 실제 답변/안내/확인/조치를 기대하는 발화
    # 판단 기준: 문장 형식(물음표)이 아닌 발화 의도
    # true  예) "AS 어떻게 신청해요?", "소비자원에 신고해야 하나요?"
    # false 예) "이게 말이 되나요?" (수사적 반문)
    # trash / undecidable → 항상 False

    is_rhetorical: bool = False
    # 수사적 질문 — 불만/비난/억울함 강조 목적의 질문형 표현
    # true  예) "버리라는 건가요?", "사기 아닌가요?"
    # is_inquiry=True 이면 is_rhetorical 은 반드시 False (상호 배타적)
    # trash / undecidable → 항상 False

    # ── 요약 / 키워드 ───────────────────────────────────
    summary: Optional[str] = None           # 한 문장 요약. trash/undecidable → None
    keywords: list[str] = field(default_factory=list)           # 최대 5개
    product_mentions: list[str] = field(default_factory=list)   # 언급된 LG 제품명

    # point extraction / similarity layer (dashboard-facing evidence fields)
    # BREAKING CHANGE (v0.3): core_points 출력 형식 변경
    #   이전: 단어 키워드 배열 ["거름망", "냄새", "세첩"]
    #   이후: 문장형 서술문 배열 ["거름망을 안 씨으면 냄새 남", "근시원서 손설거지로 회귀"]
    #   youtube-comment-analyzer 등 core_points를 직접 파싱/분리하는 하위 소비자는 수정 필요
    core_points: list[str] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)
    similarity_keys: list[str] = field(default_factory=list)
    insight_summary: Optional[str] = None
    user_wants: str = ""                    # 사용자가 진짜 원하는 것 (10~25자 한국어 한 문장, LLM 직접 생성)

    # confidence metadata (formula disclosure 대신 판단 근거 제공)
    confidence_factors: list[str] = field(default_factory=list)
    confidence_breakdown: dict[str, float] = field(default_factory=dict)
    sentiment_intensity: float = 0.0

    # ── 메타 ──────────────────────────────────────────
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    llm_provider: str = "claude"            # "claude" | "openai"
    model_name: str = ""                    # "claude-sonnet-4-20250514" 등
    error: Optional[str] = None            # 분석 실패 시 에러 메시지


# ─────────────────────────────────────────
# 팩토리 — 정상 결과 / 오류 결과
# ─────────────────────────────────────────

def make_error_result(
    comment_id: str,
    raw_text: str,
    error_message: str,
    llm_provider: str = "claude",
    model_name: str = "",
) -> AnalysisResult:
    """
    LLM 호출 실패 또는 파싱 오류 시 반환하는 표준 오류 결과.
    label 은 "undecidable" 로 고정.
    """
    return AnalysisResult(
        comment_id=comment_id,
        raw_text=raw_text,
        language="unknown",
        label="undecidable",
        confidence=0.0,
        sentiment_reason="분석 실패",
        is_inquiry=False,
        is_rhetorical=False,
        summary=None,
        error=error_message,
        llm_provider=llm_provider,
        model_name=model_name,
    )


# ─────────────────────────────────────────
# 검증
# ─────────────────────────────────────────

@dataclass
class ValidationError:
    field: str
    message: str


def validate(result: AnalysisResult) -> list[ValidationError]:
    """
    AnalysisResult 의 비즈니스 규칙 위반을 검사.
    오류 목록을 반환하며, 빈 리스트이면 유효.
    """
    errors: list[ValidationError] = []

    # label
    if result.label not in VALID_LABELS:
        errors.append(ValidationError("label", f"허용되지 않는 값: {result.label!r}"))

    # confidence
    if not (0.0 <= result.confidence <= 1.0):
        errors.append(ValidationError("confidence", f"범위 초과: {result.confidence}"))
    if not (0.0 <= result.sentiment_intensity <= 1.0):
        errors.append(ValidationError("sentiment_intensity", f"범위 초과: {result.sentiment_intensity}"))

    # topic_sentiments 값
    for topic, sentiment in result.topic_sentiments.items():
        if sentiment not in VALID_TOPIC_SENTIMENTS:
            errors.append(ValidationError(
                "topic_sentiments",
                f"토픽 {topic!r}의 감성값 {sentiment!r}이 허용되지 않음"
            ))

    # topics 최대 3개
    if len(result.topics) > 3:
        errors.append(ValidationError("topics", f"최대 3개 허용, 현재 {len(result.topics)}개"))

    # keywords 최대 5개
    if len(result.keywords) > 5:
        errors.append(ValidationError("keywords", f"최대 5개 허용, 현재 {len(result.keywords)}개"))
    if len(result.core_points) > 8:
        errors.append(ValidationError("core_points", f"최대 8개 허용, 현재 {len(result.core_points)}개"))
    if len(result.context_tags) > 6:
        errors.append(ValidationError("context_tags", f"최대 6개 허용, 현재 {len(result.context_tags)}개"))
    if len(result.similarity_keys) > 10:
        errors.append(ValidationError("similarity_keys", f"최대 10개 허용, 현재 {len(result.similarity_keys)}개"))

    # trash / undecidable → 빈 필드 강제
    if result.label in EMPTY_CONTENT_LABELS:
        if result.topics:
            errors.append(ValidationError("topics", f"{result.label} 레이블은 빈 배열이어야 함"))
        if result.topic_sentiments:
            errors.append(ValidationError("topic_sentiments", f"{result.label} 레이블은 빈 dict이어야 함"))
        if result.summary is not None:
            errors.append(ValidationError("summary", f"{result.label} 레이블은 None이어야 함"))
        if result.keywords:
            errors.append(ValidationError("keywords", f"{result.label} 레이블은 빈 배열이어야 함"))
        if result.core_points:
            errors.append(ValidationError("core_points", f"{result.label} 레이블은 빈 배열이어야 함"))
        if result.context_tags:
            errors.append(ValidationError("context_tags", f"{result.label} 레이블은 빈 배열이어야 함"))
        if result.similarity_keys:
            errors.append(ValidationError("similarity_keys", f"{result.label} 레이블은 빈 배열이어야 함"))
        if result.insight_summary is not None:
            errors.append(ValidationError("insight_summary", f"{result.label} 레이블은 None이어야 함"))
        if result.is_inquiry:
            errors.append(ValidationError("is_inquiry", f"{result.label} 레이블은 False이어야 함"))
        if result.is_rhetorical:
            errors.append(ValidationError("is_rhetorical", f"{result.label} 레이블은 False이어야 함"))

    # is_inquiry / is_rhetorical 상호 배타적
    if result.is_inquiry and result.is_rhetorical:
        errors.append(ValidationError(
            "is_inquiry/is_rhetorical",
            "is_inquiry=True 이면 is_rhetorical 은 반드시 False"
        ))

    # llm_provider
    if result.llm_provider not in VALID_PROVIDERS:
        errors.append(ValidationError("llm_provider", f"허용되지 않는 provider: {result.llm_provider!r}"))

    # error 있으면 label 은 undecidable
    if result.error and result.label != "undecidable":
        errors.append(ValidationError(
            "label",
            f"error 가 있으면 label 은 'undecidable' 이어야 함, 현재: {result.label!r}"
        ))

    return errors


def is_valid(result: AnalysisResult) -> bool:
    return len(validate(result)) == 0
