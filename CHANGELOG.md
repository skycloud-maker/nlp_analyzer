# Changelog

## [Unreleased]

### BREAKING CHANGE — `core_points` 형식 변경 (단어 → 서술형 클레임)

**영향 범위**: `AnalysisResult.core_points` 필드를 직접 읽는 모든 소비자

**변경 전**: 단어/구문 키워드 배열
```json
["거름망", "냄새", "세척"]
```

**변경 후**: 사용 경험·행동을 담은 서술형 클레임 (2~4개, 각 30자 이하)
```json
["거름망을 안 씻으면 냄새 남", "귀찮아서 손설거지로 회귀"]
```

**마이그레이션**: `core_points`를 대시보드 키워드 섹션이나 태그 목록에 그대로
렌더링하던 코드는 문자열 길이가 늘어난 점을 감안해 레이아웃을 확인하세요.

---

### 신규 필드 — `AnalysisResult.user_wants: str`

사용자가 댓글을 통해 진짜 원하는 것을 LLM이 10~25자 한국어 한 문장으로 직접 생성.

| label    | 예시                            |
|----------|---------------------------------|
| negative | `"냄새 없는 관리 편의 환경"`      |
| positive | `"세척·위생 기능의 편리한 사용성 확인"` |
| neutral  | `""` (빈 문자열)                |
| trash / undecidable | `""` (빈 문자열)   |

**연동 방법** (youtube-comment-analyzer):
```python
# sentiment.py  _nlp_result_to_dict() 에 한 줄 추가
"nlp_user_wants": getattr(result, "user_wants", ""),
```

---

### 개선 — `is_rhetorical` 감지율 향상

수사의문문 few-shot 예시를 프롬프트에 추가하여 감지율 개선:
- true 예: `"이게 말이 됩니까?"`, `"왜 이렇게 만든 거죠?"`, `"Can't they just fix this?"`
- false 예 (실제 질문): `"이거 A/S 어떻게 받나요?"`, `"언제 입고되나요?"`
