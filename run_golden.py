"""
run_golden.py — Golden Set 정확도 테스트 (실제 API 호출)

ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 가 필요합니다.
실행 시 댓글 수를 선택할 수 있습니다.

실행:
    python run_golden.py
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent


def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def check_api_key():
    claude_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not claude_key or claude_key == "your-anthropic-api-key-here":
        if not openai_key or openai_key == "your-openai-api-key-here":
            return False, "❌ API 키가 설정되지 않았습니다.\n   .env 파일에 ANTHROPIC_API_KEY 를 입력하세요."

    if claude_key and claude_key != "your-anthropic-api-key-here":
        return True, "✅ Claude API 키 확인됨 (Primary)"
    return True, "✅ OpenAI API 키 확인됨 (Secondary)"


def select_count():
    print("\n  분석할 댓글 수를 선택하세요:")
    print("  [1] 10개  — 빠른 확인 (약 30초, 소액 과금)")
    print("  [2] 30개  — 중간 확인 (약 1분)")
    print("  [3] 105개 — 전체 Golden Set (약 3~5분, 98% 목표)")
    print("  [4] 직접 입력")

    while True:
        choice = input("\n  선택 (1~4): ").strip()
        if choice == "1": return 10
        if choice == "2": return 30
        if choice == "3": return None  # 전체
        if choice == "4":
            try:
                n = int(input("  댓글 수 입력: ").strip())
                if 1 <= n <= 105:
                    return n
                print("  1~105 사이로 입력하세요.")
            except ValueError:
                print("  숫자를 입력하세요.")
        else:
            print("  1~4 중 선택하세요.")


def main():
    load_env()

    print("\n" + "=" * 55)
    print("  NLP 모듈 Golden Set 정확도 테스트")
    print("=" * 55)
    print("  ⚠️  실제 API 호출 — 소액 과금 발생")
    print("  🎯 목표: 98% 이상")
    print("=" * 55)

    # API 키 확인
    key_ok, key_msg = check_api_key()
    print(f"\n  {key_msg}")
    if not key_ok:
        print("\n  .env 파일을 열어 API 키를 입력하세요:")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # 댓글 수 선택
    max_count = select_count()

    print("\n" + "=" * 55)
    if max_count:
        print(f"  {max_count}개 댓글 분석 시작...")
    else:
        print("  전체 105개 댓글 분석 시작...")
    print("=" * 55 + "\n")

    cmd = [sys.executable, str(ROOT / "tests" / "test_analyzer.py"), "--golden"]
    if max_count:
        cmd += ["--max", str(max_count)]

    result = subprocess.run(cmd, cwd=str(ROOT))

    print("\n" + "=" * 55)
    if result.returncode == 0:
        print("  ✅ 98% 달성 — NLP 모듈 v0.1 완성!")
        print("  💡 다음 단계: GitHub 커밋 후 app.py 실행")
    else:
        print("  ❌ 98% 미달 — 프롬프트 수정이 필요합니다")
        print("  💡 오분류 케이스를 확인하고 prompts.py 를 수정하세요")
    print("=" * 55)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
