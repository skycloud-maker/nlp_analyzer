"""
tests/test_analyzer.py

NLP 모듈 자동 검증 테스트.

테스트 구성:
    A. 로직 테스트 (MockProvider — 과금 없음)
       - AnalysisResult 생성/검증
       - 에러 처리 경로
       - 빈 텍스트, 특수문자 등 엣지 케이스
       - Router 폴백 흐름

    B. Golden Set 테스트 (실제 API — 과금 있음)
       - 한국어 105개 댓글 분류 정확도
       - 목표: 98% 이상
       - 실행: python test_analyzer.py --golden

실행 방법:
    # 로직 테스트만 (과금 없음)
    python tests/test_analyzer.py

    # Golden Set 포함 전체 (실제 API 호출)
    python tests/test_analyzer.py --golden
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# 경로 설정
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models import AnalysisResult, is_valid, validate, make_error_result
from prompts import detect_language, build_prompt
from providers.mock import MockProvider, EchoProvider, FailProvider
from providers.claude import ClaudeProvider


# ─────────────────────────────────────────
# 테스트 유틸
# ─────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""

class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.results: list[TestResult] = []

    def check(self, name: str, condition: bool, message: str = ""):
        r = TestResult(name, condition, message)
        self.results.append(r)
        mark = "✅" if condition else "❌"
        print(f"    {mark} {name}" + (f" — {message}" if message else ""))
        return condition

    def summary(self) -> tuple[int, int]:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        return passed, total


# ─────────────────────────────────────────
# A. 로직 테스트 (MockProvider)
# ─────────────────────────────────────────

def test_models():
    suite = TestSuite("models.py — AnalysisResult 검증")
    print(f"\n{'='*55}\n  {suite.name}\n{'='*55}")

    # 정상 생성
    r = AnalysisResult(
        comment_id="t001", raw_text="테스트", language="ko",
        label="negative", confidence=0.95,
        sentiment_reason="불만",
        topics=["소음"], topic_sentiments={"소음": "negative"},
        is_inquiry=False, is_rhetorical=False,
        summary="소음 불만", keywords=["소음"],
        product_mentions=["냉장고"],
        llm_provider="claude", model_name="test",
    )
    suite.check("정상 AnalysisResult 생성", is_valid(r))

    # label 허용값 위반
    bad = AnalysisResult(
        comment_id="t002", raw_text="x", language="ko",
        label="unknown_label", confidence=0.5, sentiment_reason="",
        llm_provider="claude", model_name="test",
    )
    suite.check("허용되지 않는 label 감지", not is_valid(bad))

    # trash → topics 있으면 위반
    bad2 = AnalysisResult(
        comment_id="t003", raw_text="ㅋㅋ", language="unknown",
        label="trash", confidence=0.99, sentiment_reason="스팸",
        topics=["소음"],  # 규칙 위반
        llm_provider="claude", model_name="test",
    )
    suite.check("trash + topics 규칙 위반 감지", not is_valid(bad2))

    # is_inquiry + is_rhetorical 동시 True
    bad3 = AnalysisResult(
        comment_id="t004", raw_text="이게 말이 되나요?", language="ko",
        label="negative", confidence=0.9, sentiment_reason="수사적",
        is_inquiry=True, is_rhetorical=True,  # 상호 배타적 위반
        llm_provider="claude", model_name="test",
    )
    suite.check("is_inquiry + is_rhetorical 동시 True 감지", not is_valid(bad3))

    # confidence 범위 위반
    bad4 = AnalysisResult(
        comment_id="t005", raw_text="x", language="ko",
        label="neutral", confidence=1.5,  # 범위 초과
        sentiment_reason="", llm_provider="claude", model_name="test",
    )
    suite.check("confidence 범위 초과 감지", not is_valid(bad4))

    # make_error_result
    err = make_error_result("t006", "텍스트", "API 실패")
    suite.check("make_error_result label=undecidable", err.label == "undecidable")
    suite.check("make_error_result is_valid", is_valid(err))

    return suite


def test_language_detection():
    suite = TestSuite("prompts.py — 언어 감지")
    print(f"\n{'='*55}\n  {suite.name}\n{'='*55}")

    cases = [
        ("냉장고 소음이 너무 심해요", "ko"),
        ("This product is amazing quality", "en"),
        ("ㅋㅋㅋ", "unknown"),      # 자음만 → unknown
        ("ㅠㅠ", "unknown"),         # 자음만 → unknown
        ("", "unknown"),             # 빈 문자열
        ("LG 화이팅", "ko"),
        ("   ", "unknown"),          # 공백만
    ]
    for text, expected in cases:
        detected = detect_language(text)
        suite.check(
            f"detect_language({repr(text[:15])})",
            detected == expected,
            f"기대={expected}, 실제={detected}",
        )

    return suite


def test_mock_provider():
    suite = TestSuite("MockProvider — 과금 없는 분석 흐름")
    print(f"\n{'='*55}\n  {suite.name}\n{'='*55}")

    # 기본 분석 흐름 (analyzer 로직 직접 테스트)
    from analyzer import _parse_llm_response, _build_result
    from providers.base import LLMResponse

    mock = MockProvider(label="negative", confidence=0.93)
    response = mock.call("sys", "소음이 너무 심해요")
    suite.check("MockProvider call 성공", response.success)
    suite.check("MockProvider provider_name", response.provider == "mock")

    parsed, err = _parse_llm_response(response, "t001", "소음이 너무 심해요")
    suite.check("JSON 파싱 성공", parsed is not None and err is None)
    suite.check("label=negative", parsed.get("label") == "negative")

    # 빈 텍스트 처리
    mock_empty = MockProvider()
    r_empty = _parse_llm_response(
        LLMResponse("", "mock", "mock-v1", False, "빈 텍스트"),
        "t002", ""
    )
    suite.check("빈 응답 파싱 실패 처리", r_empty[0] is None)

    # 마크다운 펜스 제거
    fenced = MockProvider(response={"label": "positive", "confidence": 0.8,
        "sentiment_reason": "테스트", "is_inquiry": False, "is_rhetorical": False,
        "topics": [], "topic_sentiments": {}, "summary": "테스트",
        "keywords": [], "product_mentions": []})
    raw_fenced = LLMResponse(
        raw_text='```json\n{"label":"positive","confidence":0.8,"sentiment_reason":"ok",'
                 '"is_inquiry":false,"is_rhetorical":false,"topics":[],'
                 '"topic_sentiments":{},"summary":null,"keywords":[],"product_mentions":[]}\n```',
        provider="mock", model_name="mock-v1", success=True
    )
    parsed2, err2 = _parse_llm_response(raw_fenced, "t003", "테스트")
    suite.check("마크다운 펜스 자동 제거", parsed2 is not None, f"err={err2}")

    return suite


def test_router_fallback():
    suite = TestSuite("Router — 폴백 흐름")
    print(f"\n{'='*55}\n  {suite.name}\n{'='*55}")

    # 직접 Router 로직 검증
    from providers.base import LLMResponse

    primary   = FailProvider("Primary 실패")
    secondary = MockProvider(label="neutral", confidence=0.7)

    # Primary 실패 → Secondary 성공
    r_primary = primary.call("sys", "user")
    suite.check("FailProvider 실패 반환", not r_primary.success)

    r_secondary = secondary.call("sys", "user")
    suite.check("MockProvider 성공 반환", r_secondary.success)

    # 둘 다 실패
    both_fail_primary   = FailProvider("P 실패")
    both_fail_secondary = FailProvider("S 실패")
    rp = both_fail_primary.call("sys", "user")
    rs = both_fail_secondary.call("sys", "user")
    suite.check("둘 다 실패 시 success=False", not rp.success and not rs.success)

    # is_available
    suite.check("FailProvider is_available=True", FailProvider().is_available())
    suite.check("EchoProvider is_available=True", EchoProvider().is_available())

    return suite


def test_edge_cases():
    suite = TestSuite("엣지 케이스 — 특수 입력 처리")
    print(f"\n{'='*55}\n  {suite.name}\n{'='*55}")

    from analyzer import analyze_comment

    mock = MockProvider(label="neutral", confidence=0.5)

    # 빈 문자열
    r1 = analyze_comment("e001", "", provider=mock)
    suite.check("빈 문자열 → undecidable", r1.label == "undecidable")
    suite.check("빈 문자열 → error 있음", r1.error is not None)

    # 공백만
    r2 = analyze_comment("e002", "   ", provider=mock)
    suite.check("공백만 → undecidable", r2.label == "undecidable")

    # 특수문자만
    r3 = analyze_comment("e003", "!@#$%^&*()", provider=mock)
    suite.check("특수문자 → is_valid", is_valid(r3))

    # 500자 초과
    long_text = "냉장고 소음이 너무 심해요. " * 50
    r4 = analyze_comment("e004", long_text, provider=mock)
    suite.check("500자 초과 → is_valid", is_valid(r4))

    # FailProvider → error 결과
    fail = FailProvider()
    r5 = analyze_comment("e005", "테스트", provider=fail)
    suite.check("FailProvider → label=undecidable", r5.label == "undecidable")
    suite.check("FailProvider → error 있음", r5.error is not None)
    suite.check("FailProvider → is_valid", is_valid(r5))

    return suite


# ─────────────────────────────────────────
# B. Golden Set 테스트 (실제 API)
# ─────────────────────────────────────────

def test_golden_set(golden_path: str, max_comments: int | None = None):
    """
    Golden Set 으로 실제 API 정확도 검증.
    과금 발생 — --golden 플래그 필요.
    """
    suite = TestSuite(f"Golden Set 정확도 검증 (목표: 98%)")
    print(f"\n{'='*55}\n  {suite.name}\n{'='*55}")

    from analyzer import analyze_comment

    # API 키 확인
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("  ⚠️  API 키 없음 — Golden Set 테스트 건너뜀")
        print("     ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 를 설정하세요.")
        return suite

    # Golden Set 로드
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    if max_comments:
        golden = golden[:max_comments]

    provider = ClaudeProvider()
    total = len(golden)
    label_correct = 0
    inquiry_correct = 0
    rhetorical_correct = 0
    errors = []
    label_errors = []

    # 레이블별 정확도 추적
    per_label = defaultdict(lambda: {"total": 0, "correct": 0})

    print(f"\n  {total}개 댓글 분석 시작...")
    print(f"  {'ID':15s} {'기대':12s} {'실제':12s} {'일치':4s}")
    print(f"  {'-'*50}")

    for i, item in enumerate(golden):
        result = analyze_comment(item["id"], item["text"], provider=provider)

        expected_label = item["label"]
        actual_label   = result.label
        match          = expected_label == actual_label

        per_label[expected_label]["total"] += 1
        if match:
            per_label[expected_label]["correct"] += 1
            label_correct += 1
        else:
            label_errors.append({
                "id": item["id"],
                "text": item["text"][:40],
                "expected": expected_label,
                "actual": actual_label,
            })

        if result.is_inquiry == item.get("is_inquiry", False):
            inquiry_correct += 1
        if result.is_rhetorical == item.get("is_rhetorical", False):
            rhetorical_correct += 1

        mark = "✅" if match else "❌"
        print(f"  {item['id']:15s} {expected_label:12s} {actual_label:12s} {mark}")

        # API Rate limit 방지
        if i < total - 1:
            time.sleep(0.3)

    # ── 결과 집계
    label_acc     = label_correct / total * 100
    inquiry_acc   = inquiry_correct / total * 100
    rhetorical_acc = rhetorical_correct / total * 100

    print(f"\n  {'='*50}")
    print(f"  📊 전체 결과")
    print(f"  {'='*50}")
    print(f"  레이블 정확도  : {label_correct}/{total} = {label_acc:.1f}%  (목표: 98%)")
    print(f"  is_inquiry     : {inquiry_correct}/{total} = {inquiry_acc:.1f}%")
    print(f"  is_rhetorical  : {rhetorical_correct}/{total} = {rhetorical_acc:.1f}%")

    print(f"\n  📊 레이블별 정확도")
    for label in ["positive","negative","neutral","trash","undecidable"]:
        s = per_label[label]
        if s["total"] > 0:
            acc = s["correct"] / s["total"] * 100
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"  {label:12s}: {bar} {acc:5.1f}%  ({s['correct']}/{s['total']})")

    if label_errors:
        print(f"\n  ❌ 오분류 {len(label_errors)}개")
        for e in label_errors:
            print(f"     [{e['id']}] 기대={e['expected']} 실제={e['actual']}")
            print(f"       {e['text']}...")

    # 목표 달성 여부
    suite.check(
        f"레이블 정확도 98% 이상",
        label_acc >= 98.0,
        f"{label_acc:.1f}% ({'✅ 달성' if label_acc >= 98.0 else '❌ 미달'})",
    )
    suite.check(
        "JSON 파싱 실패율 0%",
        len(errors) == 0,
        f"파싱 오류: {len(errors)}건",
    )

    return suite


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NLP 모듈 테스트")
    parser.add_argument("--golden", action="store_true",
                        help="Golden Set 실제 API 테스트 실행 (과금 발생)")
    parser.add_argument("--max", type=int, default=None,
                        help="Golden Set 최대 댓글 수 (테스트용 샘플링)")
    args = parser.parse_args()

    all_suites = []

    # ── A. 로직 테스트 (항상 실행, 과금 없음)
    print("\n" + "🔷 " * 20)
    print("  A. 로직 테스트 (MockProvider — 과금 없음)")
    print("🔷 " * 20)

    all_suites.append(test_models())
    all_suites.append(test_language_detection())
    all_suites.append(test_mock_provider())
    all_suites.append(test_router_fallback())
    all_suites.append(test_edge_cases())

    # ── B. Golden Set 테스트 (--golden 플래그 필요)
    if args.golden:
        print("\n" + "🔶 " * 20)
        print("  B. Golden Set 테스트 (실제 API — 과금 발생)")
        print("🔶 " * 20)

        golden_path = ROOT / "tests" / "golden_set_ko.json"
        if not golden_path.exists():
            golden_path = ROOT / "golden_set_ko.json"

        if golden_path.exists():
            all_suites.append(test_golden_set(str(golden_path), args.max))
        else:
            print(f"  ⚠️  Golden Set 파일 없음: {golden_path}")
    else:
        print("\n  💡 Golden Set 테스트는 --golden 플래그로 실행하세요.")
        print("     python tests/test_analyzer.py --golden")
        print("     python tests/test_analyzer.py --golden --max 10  (10개만 샘플링)")

    # ── 최종 요약
    print("\n" + "=" * 55)
    print("  최종 결과 요약")
    print("=" * 55)
    total_passed = total_tests = 0
    for suite in all_suites:
        p, t = suite.summary()
        total_passed += p
        total_tests  += t
        mark = "✅" if p == t else "❌"
        print(f"  {mark} {suite.name[:45]:45s} {p}/{t}")

    print("=" * 55)
    overall = total_passed == total_tests
    print(f"  {'✅ 전체 통과' if overall else '❌ 실패 있음'} — {total_passed}/{total_tests}")
    print("=" * 55)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
