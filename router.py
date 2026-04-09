"""
nlp_analyzer/router.py

LLMRouter — Primary → Secondary → Fallback 순서로 자동 전환.

호출 흐름:
    1. Claude (Primary)   → 성공 시 반환
    2. OpenAI (Secondary) → Claude 실패 시 시도
    3. Fallback           → 둘 다 실패 시 명시적 오류 반환

설계 원칙:
    - Router 는 Provider 를 선택할 뿐, 비즈니스 로직을 모른다
    - 폴백은 "분석 불가" 상태를 명시적으로 반환 (로컬 모델 사용 안 함)
    - 어떤 Provider 가 사용됐는지 LLMResponse 에 기록됨
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from .providers.base import LLMInterface, LLMResponse
    from .providers.claude import ClaudeProvider
    from .providers.openai import OpenAIProvider
except ImportError:
    from providers.base import LLMInterface, LLMResponse
    from providers.claude import ClaudeProvider
    from providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Primary → Secondary → Fallback 순서로 LLM 을 자동 전환.

    사용 예:
        router = LLMRouter()
        response = router.call(system_prompt, user_message)

        # 커스텀 Provider 순서 지정
        router = LLMRouter(
            primary=ClaudeProvider(model="claude-opus-4-20250514"),
            secondary=OpenAIProvider(model="gpt-4o"),
        )
    """

    def __init__(
        self,
        primary: Optional[LLMInterface] = None,
        secondary: Optional[LLMInterface] = None,
    ):
        self._primary = primary or ClaudeProvider()
        self._secondary = secondary or OpenAIProvider()

    def call(self, system_prompt: str, user_message: str) -> LLMResponse:
        """
        Provider 순서대로 호출. 성공하면 즉시 반환.
        모두 실패하면 Fallback LLMResponse 반환.
        """

        # ── 1. Primary (Claude) ───────────────
        if self._primary.is_available():
            response = self._primary.call(system_prompt, user_message)
            if response.success:
                return response
            logger.warning(
                "Primary(%s) 실패: %s → Secondary 시도",
                self._primary.provider_name,
                response.error,
            )
        else:
            logger.info(
                "Primary(%s) 비활성 (API 키 없음) → Secondary 시도",
                self._primary.provider_name,
            )

        # ── 2. Secondary (OpenAI) ─────────────
        if self._secondary.is_available():
            response = self._secondary.call(system_prompt, user_message)
            if response.success:
                logger.info("Secondary(%s) 성공", self._secondary.provider_name)
                return response
            logger.warning(
                "Secondary(%s) 실패: %s → Fallback",
                self._secondary.provider_name,
                response.error,
            )
        else:
            logger.info(
                "Secondary(%s) 비활성 (API 키 없음) → Fallback",
                self._secondary.provider_name,
            )

        # ── 3. Fallback ───────────────────────
        # 두 Provider 모두 실패 — 명시적 오류 반환
        # 로컬 모델 폴백은 품질 손실로 채택하지 않음
        return LLMResponse(
            raw_text="",
            provider="none",
            model_name="none",
            success=False,
            error=(
                "모든 LLM Provider 실패. "
                "ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 를 확인하세요."
            ),
        )

    def status(self) -> dict:
        """
        현재 각 Provider 의 가용 상태를 반환.
        디버깅 및 헬스체크용.

        Returns:
            {
                "primary":   {"provider": "claude",  "available": True},
                "secondary": {"provider": "openai",  "available": False},
            }
        """
        return {
            "primary": {
                "provider":  self._primary.provider_name,
                "model":     self._primary.model_name,
                "available": self._primary.is_available(),
            },
            "secondary": {
                "provider":  self._secondary.provider_name,
                "model":     self._secondary.model_name,
                "available": self._secondary.is_available(),
            },
        }
# 아래는 추가분 — 파일 끝에 붙임

    def health_check(self) -> dict:
        """
        실제 API 호출 없이 시스템 준비 상태만 확인.
        과금 없음.

        Returns:
            {
                "claude":  {"available": True,  "reason": "API 키 확인됨"},
                "openai":  {"available": False, "reason": "OPENAI_API_KEY 없음"},
                "overall": "all_ok" | "degraded" | "unavailable"
            }
        """
        try:
            from .providers.claude import _ANTHROPIC_AVAILABLE
            from .providers.openai import _OPENAI_AVAILABLE
        except ImportError:
            from providers.claude import _ANTHROPIC_AVAILABLE
            from providers.openai import _OPENAI_AVAILABLE
        import os

        def _check(provider, pkg_available, env_key):
            if not pkg_available:
                return {"available": False, "reason": f"패키지 미설치 (pip install {env_key.split('_')[0].lower()})"}
            if not os.environ.get(env_key) and not provider._api_key:
                return {"available": False, "reason": f"{env_key} 환경변수 없음"}
            return {"available": True, "reason": "API 키 확인됨"}

        claude_status = _check(self._primary,   _ANTHROPIC_AVAILABLE, "ANTHROPIC_API_KEY")
        openai_status = _check(self._secondary,  _OPENAI_AVAILABLE,    "OPENAI_API_KEY")

        both_ok   = claude_status["available"] and openai_status["available"]
        both_fail = not claude_status["available"] and not openai_status["available"]

        overall = "all_ok" if both_ok else ("unavailable" if both_fail else "degraded")

        return {
            "claude":  claude_status,
            "openai":  openai_status,
            "overall": overall,
        }

