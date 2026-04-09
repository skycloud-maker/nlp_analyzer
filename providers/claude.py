"""
nlp_analyzer/providers/claude.py

ClaudeProvider — Anthropic Claude API 호출 구현.
Primary LLM Provider.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from .base import LLMInterface, LLMResponse

# anthropic 은 런타임에 지연 임포트 — 패키지 미설치 환경에서도 모듈 로드 가능
try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic = None  # type: ignore
    _ANTHROPIC_AVAILABLE = False


# ─────────────────────────────────────────
# 기본 설정값
# ─────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0   # 초


class ClaudeProvider(LLMInterface):
    """
    Anthropic Claude API Provider.

    환경변수:
        ANTHROPIC_API_KEY  — 필수. 없으면 is_available() = False

    사용 예:
        provider = ClaudeProvider()
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
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Optional[anthropic.Anthropic] = None

        if self._api_key and _ANTHROPIC_AVAILABLE:
            self._client = _anthropic.Anthropic(api_key=self._api_key)

    # ── 공개 인터페이스 ───────────────────

    def call(self, system_prompt: str, user_message: str) -> LLMResponse:
        """
        Claude API 를 호출하고 원시 응답을 반환.
        네트워크 오류 등 모든 예외를 잡아 LLMResponse(success=False) 로 반환.
        재시도 로직 내장 (기본 2회).
        """
        if not self.is_available():
            return LLMResponse(
                raw_text="",
                provider=self.provider_name,
                model_name=self._model,
                success=False,
                error="ANTHROPIC_API_KEY 가 설정되지 않았습니다.",
            )

        last_error: str = ""

        for attempt in range(self._max_retries + 1):
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

                raw_text = message.content[0].text if message.content else ""

                return LLMResponse(
                    raw_text=raw_text,
                    provider=self.provider_name,
                    model_name=self._model,
                    success=True,
                )

            except (_anthropic.RateLimitError if _ANTHROPIC_AVAILABLE else Exception) as e:
                last_error = f"Rate limit: {e}"
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * (attempt + 2))  # 백오프

            except (_anthropic.APITimeoutError if _ANTHROPIC_AVAILABLE else Exception) as e:
                last_error = f"Timeout: {e}"
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

            except (_anthropic.AuthenticationError if _ANTHROPIC_AVAILABLE else Exception) as e:
                # 재시도해도 소용없음 — 즉시 반환
                return LLMResponse(
                    raw_text="",
                    provider=self.provider_name,
                    model_name=self._model,
                    success=False,
                    error=f"인증 오류 (API 키 확인 필요): {e}",
                )

            except (_anthropic.APIError if _ANTHROPIC_AVAILABLE else Exception) as e:
                last_error = f"API 오류: {e}"
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

            except Exception as e:
                last_error = f"예상치 못한 오류: {type(e).__name__}: {e}"
                break  # 알 수 없는 오류는 재시도 없이 즉시 종료

        return LLMResponse(
            raw_text="",
            provider=self.provider_name,
            model_name=self._model,
            success=False,
            error=last_error,
        )

    def is_available(self) -> bool:
        return bool(self._api_key and self._client and _ANTHROPIC_AVAILABLE)

    @property
    def provider_name(self) -> str:
        return "claude"

    @property
    def model_name(self) -> str:
        return self._model

    # ── 내부 유틸 ─────────────────────────

    def _parse_json_safe(self, raw_text: str) -> tuple[dict | None, str | None]:
        """
        LLM 응답에서 JSON 을 안전하게 파싱.
        마크다운 코드블록(```json ... ```) 제거 후 시도.

        Returns:
            (parsed_dict, error_message) — 성공 시 error=None
        """
        text = raw_text.strip()

        # 마크다운 펜스 제거
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            return json.loads(text), None
        except json.JSONDecodeError as e:
            return None, f"JSON 파싱 실패: {e} / 원문: {raw_text[:200]}"
