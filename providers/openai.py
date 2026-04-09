"""
nlp_analyzer/providers/openai.py

OpenAIProvider — OpenAI API 호출 구현.
Secondary LLM Provider (Claude 실패 시 폴백).
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from .base import LLMInterface, LLMResponse

# openai 는 런타임에 지연 임포트 — 패키지 미설치 환경에서도 모듈 로드 가능
try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _openai = None  # type: ignore
    _OPENAI_AVAILABLE = False


# ─────────────────────────────────────────
# 기본 설정값
# ─────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0   # 초


class OpenAIProvider(LLMInterface):
    """
    OpenAI API Provider.
    Claude 실패 시 자동으로 전환되는 Secondary Provider.

    환경변수:
        OPENAI_API_KEY — 필수. 없으면 is_available() = False

    사용 예:
        provider = OpenAIProvider()
        response = provider.call(system_prompt, comment_text)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        api_key: Optional[str] = None,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # API 키: 인자 우선, 없으면 환경변수
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

        if self._api_key and _OPENAI_AVAILABLE:
            self._client = _openai.OpenAI(api_key=self._api_key)

    # ── 공개 인터페이스 ───────────────────

    def call(self, system_prompt: str, user_message: str) -> LLMResponse:
        """
        OpenAI API 를 호출하고 원시 응답을 반환.
        Claude Provider 와 동일한 에러 처리 패턴.
        """
        if not self.is_available():
            return LLMResponse(
                raw_text="",
                provider=self.provider_name,
                model_name=self._model,
                success=False,
                error="OPENAI_API_KEY 가 설정되지 않았습니다.",
            )

        last_error: str = ""

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    # JSON 모드 활성화 — OpenAI 전용 기능
                    # 프롬프트에 "JSON으로만 응답" 지침이 있어야 동작
                    response_format={"type": "json_object"},
                )

                raw_text = response.choices[0].message.content or ""

                return LLMResponse(
                    raw_text=raw_text,
                    provider=self.provider_name,
                    model_name=self._model,
                    success=True,
                )

            except Exception as e:
                # openai 패키지 예외 클래스를 동적으로 처리
                err_type = type(e).__name__

                if "RateLimitError" in err_type:
                    last_error = f"Rate limit: {e}"
                    if attempt < self._max_retries:
                        time.sleep(self._retry_delay * (attempt + 2))

                elif "APITimeoutError" in err_type or "Timeout" in err_type:
                    last_error = f"Timeout: {e}"
                    if attempt < self._max_retries:
                        time.sleep(self._retry_delay)

                elif "AuthenticationError" in err_type:
                    return LLMResponse(
                        raw_text="",
                        provider=self.provider_name,
                        model_name=self._model,
                        success=False,
                        error=f"인증 오류 (API 키 확인 필요): {e}",
                    )

                else:
                    last_error = f"{err_type}: {e}"
                    if attempt < self._max_retries:
                        time.sleep(self._retry_delay)
                    else:
                        break

        return LLMResponse(
            raw_text="",
            provider=self.provider_name,
            model_name=self._model,
            success=False,
            error=last_error,
        )

    def is_available(self) -> bool:
        return bool(self._api_key and self._client and _OPENAI_AVAILABLE)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model
