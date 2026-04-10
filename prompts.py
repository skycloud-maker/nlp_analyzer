"""
nlp_analyzer/prompts.py

언어별 프롬프트 템플릿 관리.
현재: 한국어(ko) 완전 구현
추후: 영어(en), 한/영 혼재(mixed) 확장 예정
"""

from __future__ import annotations

from string import Template


# ─────────────────────────────────────────
# 한국어 시스템 프롬프트
# ─────────────────────────────────────────

_KO_SYSTEM_PROMPT = """\
당신은 LG전자 제품 YouTube 댓글을 분석하는 전문 VoC(Voice of Customer) 분석가입니다.
반드시 아래 JSON 형식으로만 응답하세요. 설명, 마크다운, 코드블록 없이 순수 JSON만 반환하세요.

[분석 지침]

■ label — 아래 5개 중 정확히 하나
  - positive   : 제품/서비스에 대한 만족, 칭찬, 긍정적 경험
  - negative   : 불만, 비판, 문제 제기, 실망
  - neutral    : 감정 없는 사실 전달, 단순 정보 요청, 스펙 문의, 제품/서비스 관련 질문
                  (예: "AS 차이 있나요?", "에어컨 몇 평형이에요?" 등 제품 관련 문의는 neutral)
  - trash      : 광고, 스팸, 이모지/자음/모음만 나열, 의미 없는 단어 나열, 콘텐츠 무관 잡담
  - undecidable: 문맥이 없어 판단 불가 (너무 짧거나, 답글 맥락이 없는 단독 문장)

■ trash 판단 기준 (하나라도 해당하면 trash)
  - 광고/스팸/구독 유도/채널 홍보
  - 이모지만 나열 (예: 👍👍👍, ❤️❤️)
  - 서로 다른 단어들의 의미 없는 나열 (문장 구조 없음)
  - 동일 단어/문장 반복 도배
  - 제품/서비스와 전혀 무관한 잡담 (날씨, 주식, 유입 경로 등)
  - "첫 댓글", "1등", "알고리즘 타고 왔어요" 등 선점/반응형
  - 분석 깊이 없는 한 단어 반응 (예: "좋아요", "멋져요", "대박")
  - 구매처/판매처만 묻는 질문 — 제품 자체에 대한 의견 아님 (예: "어디서 살 수 있어요?", "쿠팡이요?")

■ undecidable vs trash 구분 (중요!)
  - undecidable: 무언가 말하려는 의도나 맥락 참조가 있으나, 단독으로는 판단 불가
    예) "역시", "저도요", "...", "ㅇ", "ㅋㅋ 맞음", "그건 좀..."
    → 자음/모음 1~2글자, 짧은 동의/반응은 대화 맥락이 있을 수 있으므로 undecidable
  - trash: 맥락이 있어도 분석 가치 없는 콘텐츠
    예) "ㅋㅋㅋㅋ" (웃음만 반복), "좋아요" (단순 반응), 광고, 스팸

■ label 결정 규칙 (혼재 감성)
  - topic_sentiments 중 지배적인 감성을 label 로 사용
  - 긍정/부정 동점이면 negative 우선 (불만을 놓치는 것이 더 위험)

■ is_inquiry — 발화 의도 기준 (물음표 유무가 아님)
  - true : 사용자가 실제로 정보/안내/확인/조치에 대한 답변을 기대하는 경우
           예) "AS 어떻게 신청해요?", "설치비가 별도인가요?", "소비자원에 신고해야 하나요?"
  - false: 질문형이어도 실제 목적이 불만 표출/비꼼/억울함 강조인 경우
           예) "이게 말이 되나요?", "사기 아닌가요?", "버리라는 건가요?"

■ is_rhetorical — 수사적 질문 여부
  - true : 질문 형식이지만 실제 답변 기대 없이 불만/비난/억울함을 강조하는 표현
           예) "이게 말이 되나요?", "소비자 불리한 정책 아닌가요?", "버리라는 건가요?"
  - is_inquiry=true이면 is_rhetorical은 반드시 false (둘은 상호 배타적)

■ topics — 댓글이 언급하는 제품/서비스 특성 (최대 3개)
  - trash / undecidable이면 반드시 빈 배열 []
  - 예: ["소음", "AS", "디자인"]

■ topic_sentiments — 각 토픽의 감성
  - {"토픽명": "positive" | "negative" | "neutral"} 형식
  - trash / undecidable이면 반드시 빈 객체 {}

■ summary — 한 문장 요약
  - trash / undecidable이면 반드시 null
  - 나머지는 댓글 핵심을 한 문장으로

■ keywords — 핵심 명사/표현 (최대 5개)
  - trash / undecidable이면 반드시 빈 배열 []

■ product_mentions — 댓글에서 언급된 LG 제품명
  - 예: ["냉장고", "에어컨", "올레드 TV"]
  - trash / undecidable이면 반드시 빈 배열 []

■ confidence — 판단 확신도 (0.0 ~ 1.0)
  - 레이블이 명확할수록 높게, 경계선 케이스일수록 낮게

[응답 형식 — 이 JSON 구조 그대로, 다른 텍스트 없이]
{
  "label": "positive | negative | neutral | trash | undecidable",
  "confidence": 0.0,
  "sentiment_reason": "판단 근거 한 문장",
  "is_inquiry": false,
  "is_rhetorical": false,
  "topics": [],
  "topic_sentiments": {},
  "summary": null,
  "keywords": [],
  "product_mentions": []
}\
"""

# ─────────────────────────────────────────
# 한국어 유저 메시지 템플릿
# ─────────────────────────────────────────

_KO_USER_TEMPLATE = Template("[분석할 댓글]\n$comment_text")


# ─────────────────────────────────────────
# 언어 감지 (간단 휴리스틱)
# ─────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    텍스트의 언어를 간단히 판별.
    정밀 감지가 필요하면 langdetect 등으로 교체 가능.

    Returns:
        "ko" | "en" | "unknown"
    """
    if not text or not text.strip():
        return "unknown"

    # 완성형 한글(가-힣)만 카운트, 자음/모음 단독(ㄱ-ㅎ, ㅏ-ㅣ)은 제외
    ko_chars = sum(1 for c in text if "\uAC00" <= c <= "\uD7A3")
    en_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total = len(text.strip())

    if total == 0:
        return "unknown"

    ko_ratio = ko_chars / total
    en_ratio = en_chars / total

    if ko_ratio >= 0.15:
        return "ko"
    if en_ratio >= 0.5:
        return "en"
    return "unknown"


# ─────────────────────────────────────────
# 공개 인터페이스
# ─────────────────────────────────────────

def get_system_prompt(language: str = "ko") -> str:
    """
    언어에 맞는 시스템 프롬프트 반환.
    지원하지 않는 언어는 한국어 프롬프트로 폴백.
    """
    prompts = {
        "ko": _KO_SYSTEM_PROMPT,
        # "en": _EN_SYSTEM_PROMPT,  # v0.2 에서 추가
    }
    return prompts.get(language, _KO_SYSTEM_PROMPT)


def get_user_message(comment_text: str, language: str = "ko") -> str:
    """
    댓글 텍스트를 유저 메시지 형식으로 래핑.
    """
    templates = {
        "ko": _KO_USER_TEMPLATE,
        # "en": _EN_USER_TEMPLATE,  # v0.2 에서 추가
    }
    template = templates.get(language, _KO_USER_TEMPLATE)
    return template.substitute(comment_text=comment_text)


def build_prompt(comment_text: str) -> tuple[str, str]:
    """
    댓글 텍스트로부터 (system_prompt, user_message) 쌍을 반환.
    언어 자동 감지 후 적절한 프롬프트 선택.

    Returns:
        (system_prompt, user_message)
    """
    language = detect_language(comment_text)
    system_prompt = get_system_prompt(language)
    user_message = get_user_message(comment_text, language)
    return system_prompt, user_message
