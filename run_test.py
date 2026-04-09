"""
run_test.py — 로직 테스트 전용 (과금 없음)

MockProvider 를 사용하므로 API 키 없어도 실행 가능합니다.
코드 구조, 검증 로직, 에러 처리 등을 확인합니다.

실행:
    python run_test.py
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent


def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def main():
    load_env()

    print("\n" + "=" * 55)
    print("  NLP 모듈 로직 테스트")
    print("=" * 55)
    print("  ✅ 과금 없음  |  API 키 불필요")
    print("  📋 테스트 항목:")
    print("     - AnalysisResult 구조 및 검증 (7개)")
    print("     - 언어 감지 (7개)")
    print("     - MockProvider 분석 흐름 (6개)")
    print("     - Router 폴백 흐름 (5개)")
    print("     - 엣지 케이스 처리 (8개)")
    print("=" * 55 + "\n")

    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "test_analyzer.py")],
        cwd=str(ROOT),
    )

    print("\n" + "=" * 55)
    if result.returncode == 0:
        print("  ✅ 전체 통과 — 코드 구조 정상")
        print("  💡 정확도 테스트: python run_golden.py")
    else:
        print("  ❌ 실패 — 위 오류 내용을 확인하세요")
    print("=" * 55)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
