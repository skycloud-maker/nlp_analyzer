"""
nlp_analyzer/providers/mock.py

MockProvider — 테스트 전용 가짜 LLM Provider.

목적:
    - 실제 API 호출 없이 코드 로직/검증 로직만 테스트
    - 과금 없음
    - CI/CD 환경에서 API 키 없이도 테스트 가능

사용 구분:
    - MockProvider    : 미리 정해진 응답을 항상 반환 (결정적)
    - EchoProvider    : 입력 텍스트 기반으로 간단한 규칙 응답 (휴리스틱)
    - FailProvider    : 항상 실패 — 폴백/에러 경로 테스트용
"""

from __future__ import annotations

import json
from typing import Any

from .base import LLMInterface, LLMResponse


# ─────────────────────────────────────────
# MockProvider — 고정 응답 반환
# ─────────────────────────────────────────

class MockProvider(LLMInterface):
    """
    미리 정해진 JSON 응답을 항상 반환하는 테스트용 Provider.
    실제 API 호출 없음. 과금 없음.

    사용 예:
        # 특정 결과를 고정으로 반환
        mock = MockProvider(label="negative", confidence=0.95)
        result = analyze_comment("t001", "소음이 심해요", provider=mock)

        # 커스텀 전체 응답
        mock = MockProvider(response={
            "label": "positive",
            "confidence": 0.9,
            "sentiment_reason": "만족 표현",
            "is_inquiry": False,
            "is_rhetorical": False,
            "topics": ["소음"],
            "topic_sentiments": {"소음": "positive"},
            "summary": "소음이 조용하다고 칭찬",
            "keywords": ["소음"],
            "product_mentions": ["냉장고"],
        })
    """

    _DEFAULT_RESPONSE = {
        "label": "neutral",
        "confidence": 0.8,
        "sentiment_reason": "Mock 응답",
        "is_inquiry": False,
        "is_rhetorical": False,
        "topics": [],
        "topic_sentiments": {},
        "summary": "Mock 요약",
        "keywords": [],
        "product_mentions": [],
    }

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        label: str | None = None,
        confidence: float | None = None,
        should_fail: bool = False,
        fail_message: str = "Mock 강제 실패",
    ):
        """
        Args:
            response     : 반환할 전체 JSON dict. 없으면 기본값 사용.
            label        : response 없을 때 label 만 지정하는 단축키.
            confidence   : response 없을 때 confidence 만 지정하는 단축키.
            should_fail  : True 이면 항상 실패 응답 반환.
            fail_message : 실패 시 에러 메시지.
        """
        self._should_fail = should_fail
        self._fail_message = fail_message

        # 응답 구성: response > 단축키 > 기본값
        base = dict(self._DEFAULT_RESPONSE)
        if response:
            base.update(response)
        if label is not None:
            base["label"] = label
        if confidence is not None:
            base["confidence"] = confidence

        self._response_json = json.dumps(base, ensure_ascii=False)

    def call(self, system_prompt: str, user_message: str) -> LLMResponse:
        if self._should_fail:
            return LLMResponse(
                raw_text="",
                provider=self.provider_name,
                model_name=self.model_name,
                success=False,
                error=self._fail_message,
            )
        return LLMResponse(
            raw_text=self._response_json,
            provider=self.provider_name,
            model_name=self.model_name,
            success=True,
        )

    def is_available(self) -> bool:
        return True  # Mock 은 항상 사용 가능

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-v1"


# ─────────────────────────────────────────
# EchoProvider — 규칙 기반 자동 응답
# ─────────────────────────────────────────

class EchoProvider(LLMInterface):
    """
    입력 텍스트에서 간단한 키워드 규칙으로 label 을 결정하는 테스트용 Provider.
    Golden Set 전체를 빠르게 흘려볼 때 유용.
    정확도보다 흐름 테스트 목적.

    규칙 (우선순위 순):
        trash      : ㅋ, ㅎ, 👍, 구독, 팔로우 포함
        negative   : 실망, 고장, 환불, 소음, 최악, 화나 포함
        positive   : 만족, 추천, 좋아요, 최고, 감사 포함
        neutral    : 그 외 (기본값)
    """

    _RULES = [
        ("trash",    ["ㅋ", "ㅎ", "👍", "구독", "팔로우", "맞구독", "채널"]),
        ("negative", ["실망", "고장", "환불", "소음", "최악", "화납", "불만", "최악", "사기"]),
        ("positive", ["만족", "추천", "좋아요", "최고", "감사", "강추", "칭찬"]),
    ]

    def call(self, system_prompt: str, user_message: str) -> LLMResponse:
        label = "neutral"
        for candidate_label, keywords in self._RULES:
            if any(kw in user_message for kw in keywords):
                label = candidate_label
                break

        response = {
            "label": label,
            "confidence": 0.6,
            "sentiment_reason": f"EchoProvider 규칙 기반 분류: {label}",
            "is_inquiry": "?" in user_message or "요?" in user_message,
            "is_rhetorical": False,
            "topics": [],
            "topic_sentiments": {},
            "summary": None if label in ("trash", "undecidable") else user_message[:30],
            "keywords": [],
            "product_mentions": [],
        }
        return LLMResponse(
            raw_text=json.dumps(response, ensure_ascii=False),
            provider=self.provider_name,
            model_name=self.model_name,
            success=True,
        )

    def is_available(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "echo"

    @property
    def model_name(self) -> str:
        return "echo-v1"


# ─────────────────────────────────────────
# FailProvider — 항상 실패
# ─────────────────────────────────────────

class FailProvider(LLMInterface):
    """
    항상 실패 응답을 반환하는 Provider.
    Router 의 폴백 경로, 에러 처리 로직 테스트용.

    사용 예:
        router = LLMRouter(
            primary=FailProvider(),
            secondary=FailProvider(),
        )
        response = router.call(...)
        assert response.success is False
    """

    def __init__(self, error_message: str = "FailProvider: 의도적 실패"):
        self._error_message = error_message

    def call(self, system_prompt: str, user_message: str) -> LLMResponse:
        return LLMResponse(
            raw_text="",
            provider=self.provider_name,
            model_name=self.model_name,
            success=False,
            error=self._error_message,
        )

    def is_available(self) -> bool:
        return True  # 가용하지만 항상 실패

    @property
    def provider_name(self) -> str:
        return "fail"

    @property
    def model_name(self) -> str:
        return "fail-v1"
