"""
nlp_analyzer/providers/base.py

LLMInterface — 모든 LLM Provider 가 구현해야 하는 추상 계약.
Claude, OpenAI, 미래의 온프레미스 모델 모두 이 인터페이스를 따른다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """
    LLM 호출의 원시 결과.
    Provider 구현체는 항상 이 구조로 반환하며,
    파싱/변환은 analyzer.py 에서 담당한다.
    """
    raw_text: str           # LLM 이 반환한 원문 텍스트 (JSON 문자열 기대)
    provider: str           # "claude" | "openai"
    model_name: str         # 실제 사용된 모델명
    success: bool           # 호출 자체의 성공 여부
    error: str | None = None  # 실패 시 에러 메시지


class LLMInterface(ABC):
    """
    LLM Provider 추상 인터페이스.

    구현 규칙:
    - call() 은 네트워크 오류 등 예외를 잡아 LLMResponse(success=False) 로 반환
    - 예외를 caller 로 전파하지 않는다
    - 재시도 로직은 구현체 내부에서 처리
    """

    @abstractmethod
    def call(self, system_prompt: str, user_message: str) -> LLMResponse:
        """
        LLM 을 호출하고 원시 응답을 반환한다.

        Args:
            system_prompt: 분석 지침이 담긴 시스템 프롬프트
            user_message:  분석할 댓글 텍스트

        Returns:
            LLMResponse — 성공/실패 여부와 원시 텍스트
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Provider 가 현재 사용 가능한 상태인지 확인.
        API 키 존재 여부 등 기본 조건만 체크 (실제 네트워크 호출 없음).
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider 식별자 — "claude" | "openai" """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """현재 설정된 모델명"""
        ...
