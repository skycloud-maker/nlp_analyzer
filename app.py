"""
app.py — NLP 분석 Streamlit 앱

기능:
    탭 1. 단건 분석  — 댓글 직접 입력 → 분석 결과 표시
    탭 2. 일괄 분석  — 엑셀 업로드 → 전체 분석 → 결과 엑셀 다운로드

실행:
    pip install streamlit pandas openpyxl anthropic
    streamlit run app.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────
# 환경변수 로드 (.env)
# ─────────────────────────────────────────
def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

load_env()


# ─────────────────────────────────────────
# NLP 모듈 임포트
# ─────────────────────────────────────────
try:
    from analyzer import analyze_comment, analyze_batch
    from router import LLMRouter
    NLP_AVAILABLE = True
except Exception as e:
    NLP_AVAILABLE = False
    NLP_ERROR = str(e)


# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="LG VoC NLP 분석",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans KR', sans-serif;
    }
    .mono { font-family: 'IBM Plex Mono', monospace; }

    .stApp { background-color: #080c14; color: #e2e8f0; }

    .label-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 15px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 12px;
        margin: 2px;
        background: #1e293b;
        color: #94a3b8;
    }
    .section-title {
        font-size: 11px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 6px;
    }
    .result-box {
        background: #0f172a;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1e293b;
        margin-top: 12px;
    }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

LABEL_COLOR = {
    "positive":    ("#22c55e", "#052e16"),
    "negative":    ("#ef4444", "#2d0a0a"),
    "neutral":     ("#94a3b8", "#0f172a"),
    "trash":       ("#f59e0b", "#1c1407"),
    "undecidable": ("#a78bfa", "#1a0a2e"),
}
LABEL_KO = {
    "positive": "긍정", "negative": "부정", "neutral": "중립",
    "trash": "쓰레기", "undecidable": "판단불가",
}
SENTIMENT_COLOR = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#94a3b8"}


# ─────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 LG VoC NLP")
    st.markdown("---")

    # 시스템 상태
    st.markdown("**시스템 상태**")
    if NLP_AVAILABLE:
        router = LLMRouter()
        status = router.health_check()
        for provider, info in [("Claude", status["claude"]), ("OpenAI", status["openai"])]:
            icon = "🟢" if info["available"] else "🔴"
            st.markdown(f"{icon} **{provider}**: {info['reason']}")
        overall = status["overall"]
        color = {"all_ok": "🟢", "degraded": "🟡", "unavailable": "🔴"}.get(overall, "⚪")
        st.markdown(f"{color} 전체: `{overall}`")
    else:
        st.error("모듈 로드 실패")

    st.markdown("---")
    st.markdown("**버전**: v0.1")
    st.markdown("**언어**: 한국어")
    st.markdown("**목표 정확도**: 98%")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
st.markdown("# LG VoC 댓글 분석")
st.markdown("YouTube 댓글의 감성, 토픽, 문의 여부를 자동으로 분석합니다.")
st.markdown("---")

if not NLP_AVAILABLE:
    st.error(f"NLP 모듈 로드 실패: {NLP_ERROR}")
    st.stop()

tab1, tab2 = st.tabs(["💬 단건 분석", "📊 일괄 분석 (엑셀)"])


# ═══════════════════════════════════════
# 탭 1 — 단건 분석
# ═══════════════════════════════════════
with tab1:
    st.markdown("### 댓글 입력")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        comment_text = st.text_area(
            "분석할 댓글을 입력하세요",
            height=150,
            placeholder="예) 냉장고 소음이 너무 심해요. 환불하고 싶어요.",
            key="single_input",
        )

        # 샘플 댓글
        st.markdown('<p class="section-title">샘플 댓글</p>', unsafe_allow_html=True)
        samples = [
            ("positive",    "이 냉장고 산 지 3년 됐는데 아직도 소음 없이 잘 돌아가요."),
            ("negative",    "AS 신청하고 2주째 연락이 없어요. 진짜 화납니다."),
            ("negative",    "디자인은 예쁜데 실용성이 너무 떨어져요."),
            ("neutral",     "이 모델 용량이 몇 리터예요?"),
            ("trash",       "구독하고 갑니다~ 맞구독 환영해요!"),
            ("undecidable", "역시"),
        ]
        cols = st.columns(2)
        for i, (label, text) in enumerate(samples):
            color, bg = LABEL_COLOR.get(label, ("#94a3b8", "#0f172a"))
            with cols[i % 2]:
                if st.button(
                    f"[{LABEL_KO[label]}] {text[:18]}...",
                    key=f"sample_{i}",
                    use_container_width=True,
                ):
                    st.session_state["single_input"] = text
                    st.rerun()

        analyze_btn = st.button("🔍 분석 실행", type="primary", use_container_width=True)

    with col_result:
        if analyze_btn and comment_text.strip():
            with st.spinner("분석 중..."):
                result = analyze_comment("app_single", comment_text)

            label = result.label
            color, bg = LABEL_COLOR.get(label, ("#94a3b8", "#0f172a"))
            ko = LABEL_KO.get(label, label)

            # 레이블 + confidence
            conf_pct = int((result.confidence or 0) * 100)
            st.markdown(f"""
            <div class="result-box">
                <span class="label-badge" style="background:{bg};color:{color};border:1px solid {color}44">
                    {ko} ({label})
                </span>
                <span style="font-family:'IBM Plex Mono',monospace;color:{color};margin-left:12px;font-size:18px;font-weight:700">
                    {conf_pct}%
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.progress(result.confidence or 0)

            # 플래그
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                flag_color = "#38bdf8" if result.is_inquiry else "#334155"
                st.markdown(f'<span class="tag" style="color:{flag_color};border:1px solid {flag_color}">문의: {"✓ true" if result.is_inquiry else "false"}</span>', unsafe_allow_html=True)
            with col_f2:
                flag_color2 = "#fb923c" if result.is_rhetorical else "#334155"
                st.markdown(f'<span class="tag" style="color:{flag_color2};border:1px solid {flag_color2}">수사적질문: {"✓ true" if result.is_rhetorical else "false"}</span>', unsafe_allow_html=True)

            # 판단 근거
            if result.sentiment_reason:
                st.markdown('<p class="section-title" style="margin-top:16px">판단 근거</p>', unsafe_allow_html=True)
                st.markdown(f'<div style="background:#0f172a55;padding:10px 14px;border-radius:8px;font-size:14px">{result.sentiment_reason}</div>', unsafe_allow_html=True)

            # 요약
            if result.summary:
                st.markdown('<p class="section-title" style="margin-top:12px">요약</p>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-style:italic;color:#94a3b8;font-size:13px">{result.summary}</div>', unsafe_allow_html=True)

            # 토픽 감성
            if result.topics:
                st.markdown('<p class="section-title" style="margin-top:12px">토픽 감성</p>', unsafe_allow_html=True)
                topic_cols = st.columns(len(result.topics))
                for i, topic in enumerate(result.topics):
                    s = result.topic_sentiments.get(topic, "neutral")
                    sc = SENTIMENT_COLOR.get(s, "#94a3b8")
                    with topic_cols[i]:
                        st.markdown(f'''<div style="background:#0f172a;border:1px solid {sc}44;border-radius:8px;padding:8px 12px;text-align:center">
                            <div style="font-size:13px;color:#e2e8f0">{topic}</div>
                            <div style="font-size:11px;color:{sc};margin-top:4px">{s}</div>
                        </div>''', unsafe_allow_html=True)

            # 키워드 + 제품
            col_k, col_p = st.columns(2)
            with col_k:
                if result.keywords:
                    st.markdown('<p class="section-title" style="margin-top:12px">키워드</p>', unsafe_allow_html=True)
                    st.markdown(" ".join([f'<span class="tag">{k}</span>' for k in result.keywords]), unsafe_allow_html=True)
            with col_p:
                if result.product_mentions:
                    st.markdown('<p class="section-title" style="margin-top:12px">제품</p>', unsafe_allow_html=True)
                    st.markdown(" ".join([f'<span class="tag" style="color:#34d399;border:1px solid #064e3b">{p}</span>' for p in result.product_mentions]), unsafe_allow_html=True)

            if result.error:
                st.error(f"분석 오류: {result.error}")


# ═══════════════════════════════════════
# 탭 2 — 일괄 분석 (엑셀)
# ═══════════════════════════════════════
with tab2:
    st.markdown("### 엑셀 업로드 → 일괄 분석 → 결과 다운로드")

    st.info("""
    **엑셀 파일 형식**
    - 필수 열: `text` (댓글 내용)
    - 선택 열: `id` (댓글 ID — 없으면 자동 부여)
    - 첫 번째 시트를 읽습니다
    """)

    # 샘플 엑셀 다운로드
    sample_df = pd.DataFrame({
        "id": ["yt_001", "yt_002", "yt_003", "yt_004", "yt_005"],
        "text": [
            "냉장고 소음이 너무 심해요. 환불하고 싶어요.",
            "에어컨 설치 기사님이 너무 친절하셨어요!",
            "이 모델 용량이 몇 리터예요?",
            "구독하고 갑니다~ 맞구독 환영해요!",
            "AS 신청하고 2주째 연락이 없어요.",
        ]
    })
    sample_buf = BytesIO()
    with pd.ExcelWriter(sample_buf, engine="openpyxl") as writer:
        sample_df.to_excel(writer, index=False, sheet_name="댓글목록")
    st.download_button(
        "📥 샘플 엑셀 다운로드",
        data=sample_buf.getvalue(),
        file_name="sample_comments.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")

    uploaded = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])

    if uploaded:
        try:
            df = pd.read_excel(uploaded)
            st.success(f"✅ {len(df)}개 댓글 로드 완료")
            st.dataframe(df.head(5), use_container_width=True)

            if "text" not in df.columns:
                st.error("'text' 열이 없습니다. 열 이름을 확인하세요.")
            else:
                # ID 열 없으면 자동 부여
                if "id" not in df.columns:
                    df["id"] = [f"row_{i+1:04d}" for i in range(len(df))]

                total = len(df)
                st.warning(f"⚠️ {total}개 댓글을 분석합니다. 실제 API 과금이 발생합니다.")

                if st.button(f"🚀 {total}개 분석 시작", type="primary"):
                    comments = df[["id", "text"]].rename(
                        columns={"id": "id", "text": "text"}
                    ).to_dict("records")

                    progress = st.progress(0)
                    status_text = st.empty()
                    results_data = []

                    for i, comment in enumerate(comments):
                        status_text.text(f"분석 중... {i+1}/{total}")
                        result = analyze_comment(
                            str(comment["id"]),
                            str(comment["text"])
                        )
                        results_data.append({
                            "id":               result.comment_id,
                            "text":             result.raw_text,
                            "label":            result.label,
                            "label_ko":         LABEL_KO.get(result.label, result.label),
                            "confidence":       round(result.confidence or 0, 3),
                            "sentiment_reason": result.sentiment_reason,
                            "is_inquiry":       result.is_inquiry,
                            "is_rhetorical":    result.is_rhetorical,
                            "topics":           ", ".join(result.topics),
                            "topic_sentiments": json.dumps(result.topic_sentiments, ensure_ascii=False),
                            "summary":          result.summary or "",
                            "keywords":         ", ".join(result.keywords),
                            "product_mentions": ", ".join(result.product_mentions),
                            "error":            result.error or "",
                        })
                        progress.progress((i + 1) / total)
                        if i < total - 1:
                            time.sleep(0.2)  # Rate limit 방지

                    status_text.text("✅ 분석 완료!")
                    result_df = pd.DataFrame(results_data)

                    # 정확도 요약
                    st.markdown("---")
                    st.markdown("### 📊 분석 결과 요약")
                    label_counts = result_df["label"].value_counts()
                    cols = st.columns(5)
                    for i, (label, count) in enumerate(label_counts.items()):
                        color, bg = LABEL_COLOR.get(label, ("#94a3b8", "#0f172a"))
                        with cols[i % 5]:
                            st.markdown(f'''<div style="background:{bg};border:1px solid {color}44;border-radius:8px;padding:12px;text-align:center">
                                <div style="color:{color};font-size:20px;font-weight:700">{count}</div>
                                <div style="color:{color};font-size:11px">{LABEL_KO.get(label,label)}</div>
                            </div>''', unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("### 결과 미리보기")
                    st.dataframe(result_df, use_container_width=True)

                    # 엑셀 다운로드
                    out_buf = BytesIO()
                    with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                        result_df.to_excel(writer, index=False, sheet_name="분석결과")

                        # 레이블별 시트 추가
                        for label in ["positive", "negative", "neutral", "trash", "undecidable"]:
                            subset = result_df[result_df["label"] == label]
                            if not subset.empty:
                                subset.to_excel(writer, index=False, sheet_name=LABEL_KO[label])

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        "📥 결과 엑셀 다운로드",
                        data=out_buf.getvalue(),
                        file_name=f"nlp_result_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

        except Exception as e:
            st.error(f"파일 처리 오류: {e}")
