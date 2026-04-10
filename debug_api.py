"""
debug_api.py - API 연결 진단 스크립트
실행: python debug_api.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# .env 로드
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    print("[OK] .env file loaded")
else:
    print("[ERROR] .env file not found")
    sys.exit(1)

# API 키 상태 출력
claude_key = os.environ.get("ANTHROPIC_API_KEY", "")
openai_key = os.environ.get("OPENAI_API_KEY", "")

claude_ok = bool(claude_key and claude_key != "your-anthropic-api-key-here")
openai_ok = bool(openai_key and openai_key != "your-openai-api-key-here")

print(f"Claude API key : {'[OK] ' + claude_key[:12] + '...' if claude_ok else '[MISSING]'}")
print(f"OpenAI API key : {'[OK] ' + openai_key[:12] + '...' if openai_ok else '[MISSING]'}")

if not claude_ok and not openai_ok:
    print("\n[ERROR] No API key found. Please set at least one key in .env file.")
    sys.exit(1)

# OpenAI 테스트
if openai_ok:
    print("\nTesting OpenAI API connection...")
    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hello in one word"}]
        )
        print(f"[OK] OpenAI API works! Response: {resp.choices[0].message.content}")
    except openai.AuthenticationError:
        print("[ERROR] OpenAI authentication failed. Check your API key.")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")

# Claude 테스트
if claude_ok:
    print("\nTesting Claude API connection...")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=claude_key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hello in one word"}]
        )
        print(f"[OK] Claude API works! Response: {msg.content[0].text}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
