# nlp_analyzer

LG전자 YouTube VoC(Voice of Customer) 댓글 자연어 처리 모듈

> **설계 원칙**: 구조보다 분석 정확도 우선. 댓글 하나를 입력받아 분석 결과 구조체 하나를 반환한다.

---

## 목표 정확도

| 항목 | 기준 |
|------|------|
| 레이블 정확도 (한국어) | **98% 이상** |
| Trash 감지율 | 98% 이상 |
| JSON 파싱 실패율 | 0% |

---

## 폴더 구조

```
nlp_analyzer/
├── __init__.py              # 패키지 공개 인터페이스
├── models.py                # AnalysisResult 구조체 + 검증
├── prompts.py               # LLM 프롬프트 템플릿 + 언어 감지
├── analyzer.py              # 메인 진입점 (analyze_comment, analyze_batch)
├── router.py                # Primary → Secondary → Fallback 자동 전환
│
├── providers/
│   ├── __init__.py
│   ├── base.py              # LLMInterface 추상 클래스
│   ├── claude.py            # Claude API (Primary)
│   ├── openai.py            # OpenAI API (Secondary / 폴백)
│   └── mock.py              # MockProvider / EchoProvider / FailProvider (테스트용)
│
└── tests/
    ├── __init__.py
    ├── test_analyzer.py     # 자동 검증 테스트
    └── golden_set_ko.json   # 한국어 정답 셋 105개
```

---

## 분석 레이블

| 레이블 | 설명 | 예시 |
|--------|------|------|
| `positive` | 만족, 칭찬, 긍정적 경험 | "냉장고 소음이 정말 조용해요" |
| `negative` | 불만, 비판, 문제 제기 | "AS 연락이 너무 안 돼요" |
| `neutral` | 감정 없는 사실 전달, 정보 요청 | "이 모델 용량이 몇 리터예요?" |
| `trash` | 광고, 스팸, 의미 없는 나열 | "ㅋㅋㅋ", "구독하고 갑니다~" |
| `undecidable` | 문맥 부족, 판단 불가 | "역시", "저도요" |

---

## 설치

```bash
pip install anthropic          # Claude API (Primary)
pip install openai             # OpenAI API (Secondary, 선택)
```

---

## 환경변수 설정

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # 필수 (Primary)
export OPENAI_API_KEY="sk-..."          # 선택 (폴백용)
```

---

## 사용법

```python
from nlp_analyzer import analyze_comment, analyze_batch

# 단일 댓글 분석
result = analyze_comment(
    comment_id="yt_001",
    text="냉장고 소음이 너무 심해요. 환불하고 싶어요.",
)

print(result.label)              # "negative"
print(result.confidence)         # 0.95
print(result.topic_sentiments)   # {"소음": "negative", "환불": "negative"}
print(result.is_inquiry)         # False
print(result.is_rhetorical)      # False
print(result.summary)            # "냉장고 소음 심각, 환불 요청"

# 배치 분석
results = analyze_batch([
    {"id": "yt_001", "text": "정말 만족해요!"},
    {"id": "yt_002", "text": "AS가 너무 느려요"},
])
```

---

## LLM 호출 흐름

```
1. Claude API (Primary)    → 성공 시 반환
       ↓ 실패
2. OpenAI API (Secondary)  → 성공 시 반환
       ↓ 실패
3. Fallback                → AnalysisResult(label="undecidable", error="...")
                             재시도 큐에 적재
```

---

## 시스템 상태 확인 (과금 없음)

```python
from nlp_analyzer.router import LLMRouter

router = LLMRouter()
print(router.health_check())
# {
#   "claude":  {"available": True,  "reason": "API 키 확인됨"},
#   "openai":  {"available": False, "reason": "OPENAI_API_KEY 없음"},
#   "overall": "degraded"
# }
```

---

## 테스트 실행

```bash
# 로직 테스트 (과금 없음 — 항상 먼저 실행)
python tests/test_analyzer.py

# Golden Set 정확도 테스트 (실제 API 호출 — 과금 발생)
python tests/test_analyzer.py --golden

# 샘플링 테스트 (10개만, 빠른 확인용)
python tests/test_analyzer.py --golden --max 10
```

---

## AnalysisResult 구조

```python
@dataclass
class AnalysisResult:
    comment_id: str               # 원본 댓글 ID
    raw_text: str                 # 원본 텍스트
    language: str                 # "ko" | "en" | "unknown"

    label: str                    # positive | negative | neutral | trash | undecidable
    confidence: float             # 0.0 ~ 1.0

    sentiment_reason: str         # 판단 근거 한 문장
    topics: list[str]             # 언급 토픽 (최대 3개)
    topic_sentiments: dict        # {"디자인": "positive", "AS": "negative"}

    is_inquiry: bool              # 실제 답변을 기대하는 문의 여부
    is_rhetorical: bool           # 수사적 질문 여부 (불만 강조형)

    summary: str | None           # 한 문장 요약 (trash/undecidable → None)
    keywords: list[str]           # 핵심 키워드 (최대 5개)
    product_mentions: list[str]   # 언급된 LG 제품명

    analyzed_at: datetime
    llm_provider: str             # "claude" | "openai"
    model_name: str
    error: str | None             # 분석 실패 시 에러 메시지
```

---

## 버전 히스토리

| 버전 | 내용 |
|------|------|
| v0.1 | 한국어 기준 완전 구현. 98% 정확도 달성 목표 |
| v0.2 | 영어 지원 추가 예정 |
| v0.3 | 한/영 혼재 댓글 처리 예정 |
| v1.0 | 온프레미스 LLM 지원 예정 |

---

## 다음 단계 (이 모듈 완성 후)

- `video_context` 연동 — 영상 메타데이터와 결합한 인사이트 추출
- `insight_engine` — 댓글 빈도/패턴 기반 VoC 인사이트 생성
- 대시보드 연결 — 기존 YouTube VoC Dashboard에 이 모듈 탑재
