# streamlit_rag_app.py

import os
import re
import textwrap
import html
from urllib.parse import unquote
import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# 🔧 기본 설정
# -----------------------------
# === 경로를 상대경로로 설정 (chatbot_repo/data) ===
try:
    BASE_DIR = Path(__file__).resolve().parent   # 이 파일이 있는 폴더 = chatbot_repo
except NameError:                                 # Jupyter 등 __file__이 없을 때
    BASE_DIR = Path.cwd()


# Path 객체로 경로 정의
DEFAULT_CHROMA_DIR_PATH = BASE_DIR / "data"

# ✨✨ 이 부분을 추가하세요! ✨✨
# ./data 디렉터리가 존재하지 않으면, 자동으로 생성합니다.
# 부모 디렉터리가 없어도 만들어주고(parents=True), 이미 있어도 오류를 내지 않습니다(exist_ok=True).
DEFAULT_CHROMA_DIR_PATH.mkdir(parents=True, exist_ok=True)

# 실제 사용할 때는 문자열로 변환
DEFAULT_CHROMA_DIR = str(DEFAULT_CHROMA_DIR_PATH)

#DEFAULT_CHROMA_DIR = r"C:\Users\PC1\OneDrive\프로젝트\250801_아모레\chroma_db"
DEFAULT_MODEL       = "google/gemma-2-9b-it"   # 03_RAG와 동일 계열
EMBED_MODEL_NAME    = "nlpai-lab/KURE-v1"      # 03_RAG와 동일
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """You are a precise RAG assistant. Use ONLY the provided context.
If the answer is not in the context, say you don't know briefly.
Answer in Korean.
"""

USER_PROMPT_TEMPLATE = """[Context]
{context}

[Question]
{question}

[Instructions]
- Keep the answer concise but complete (≤ 5 sentences).
"""


# -----------------------------
# 🎨 페이지 스타일
# -----------------------------
st.set_page_config(page_title="RAG Q&A", page_icon="🔎", layout="wide")

CUSTOM_CSS = """
<style>
header {visibility: hidden;}
.answer-card {
  border-radius: 16px; padding: 18px 20px;
  background: #ffffff; border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}
.chip {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:#eef2ff; color:#3949ab; font-weight:600; font-size:12px; margin-bottom:6px;
}
.doc-card {
  border-radius: 14px; padding: 12px 14px; background:#fafafa; border:1px solid #eee; margin-bottom:12px;
}
.doc-head {font-weight:700; font-size:14px;}
.doc-body {font-size:13px; color:#333;}
.small {color:#666; font-size:12px;}
.footer {color:#777; font-size:12px; margin-top:8px;}
.stat {color:#444; font-size:13px; margin-bottom:6px;}
.history-card {
  border-radius: 12px; padding:10px 12px; background:#f6f7fb; border:1px solid #e6e8f5; margin-bottom:8px;
}
.history-q {font-weight:700; font-size:13px;}
.history-a {font-size:12px; color:#555; margin-top:4px;}
.suggest-box {margin-top:8px; padding:10px 12px; background:#f9fafb; border:1px dashed #d7dbe2; border-radius:12px;}

.doc-tags {margin:6px 0 8px 0}
.badge {display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#3949ab; font-size:11px; margin-right:6px;}
.filepill {display:inline-block; padding:2px 8px; background:#f1f5ff; border:1px solid #dfe6ff; color:#2146b7; border-radius:8px; font-weight:600;}
.sum-list {margin:6px 0 0 0; padding-left:18px;}
.kv-table {width:100%; border-collapse:collapse; margin-top:8px; font-size:12px;}
.kv-table th, .kv-table td {border:1px solid #e9e9ef; padding:6px 8px; text-align:left;}
.kv-table th {background:#f6f7fb; font-weight:700;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🔎 RAG Q&A")
st.caption("03_RAG 파이프라인을 Streamlit UI로 — LangChain + Chroma + OpenRouter")

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []   # [{q, a, k, sources}]
if "retrieved_docs" not in st.session_state:
    st.session_state["retrieved_docs"] = []
if "answer" not in st.session_state:
    st.session_state["answer"] = ""
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None


# -----------------------------
# 🧰 사이드바 (설정 + 최근 기록)
# -----------------------------
with st.sidebar:
    st.subheader("Settings")
    chroma_dir  = st.text_input("Chroma persist directory", value=DEFAULT_CHROMA_DIR)
    top_k       = st.slider("Top-K (retrieval)", 1, 10, 3)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2)
    model_name  = st.text_input("LLM (OpenRouter)", value=DEFAULT_MODEL)
    embed_name  = st.text_input("Embedding model", value=EMBED_MODEL_NAME)
    device_opt  = st.selectbox("Embedding device", ["auto (cuda if available)", "cpu", "cuda"], index=0)
    st.caption("※ OPENROUTER_API_KEY 환경변수 설정 시 LLM 호출 활성화")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    st.divider()
    st.subheader("🕘 Recent (last 5)")
    if not st.session_state["history"]:
        st.caption("최근 Q&A 기록이 없습니다.")
    else:
        for idx, item in enumerate(st.session_state["history"][:5]):
            q_short = textwrap.shorten(item["q"], width=60, placeholder=" …")
            a_short = textwrap.shorten(item.get("a",""), width=70, placeholder=" …") if item.get("a") else "(생성 없음)"
            st.markdown(f'<div class="history-card"><div class="history-q">Q: {html.escape(q_short)}</div><div class="history-a">A: {html.escape(a_short)}</div></div>', unsafe_allow_html=True)
            if st.button("↺ 이 질문 불러오기", key=f"load_hist_{idx}"):
                st.session_state["pending_query"] = item["q"]
                st.rerun()


# -----------------------------
# ♻️ 리소스 로드 (cache)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_embeddings(model_name: str, device_choice: str):
    # device 결정
    dev = "cpu"
    if device_choice == "cpu":
        dev = "cpu"
    elif device_choice == "cuda":
        dev = "cuda"
    else:
        # auto
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": dev},
        encode_kwargs={"normalize_embeddings": True},
    )

# 🔧 UnhashableParamError 대응: 해시 불가 객체는 인자명 앞에 '_'로 전달
@st.cache_resource(show_spinner=True)
def load_vectorstore(persist_dir: str, _embeddings):
    return Chroma(persist_directory=persist_dir, embedding_function=_embeddings, collection_name="amore_v1")

@st.cache_resource(show_spinner=True)
def build_chain(model_name: str, temperature: float, base_url: str, api_key: str):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT_TEMPLATE)]
    )
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key if api_key else "DUMMY",
    )
    parser = StrOutputParser()
    return prompt | llm | parser


# -----------------------------
# 🔧 유틸: 문서 카드 렌더링
# -----------------------------
def _clean_filename(name: str) -> str:
    if not name:
        return "unknown"
    return unquote(name).replace("+", " ")

def parse_summary_and_figures(text: str):
    """문서 텍스트에서 '핵심요약:', '핵심수치' 패턴을 찾아 정돈.
       줄 맨 앞 번호(예: '1) ')만 제거하고, 필드 값은 건드리지 않는다."""
    if not text:
        return [], []

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    summary_items, figure_rows = [], []

    # '핵심요약' 수집
    in_summary = False
    for ln in lines:
        if ln.startswith("핵심요약"):
            in_summary = True
            after = ln.split(":", 1)[1].strip() if ":" in ln else ""
            if after:
                for piece in after.split(" - "):
                    p = piece.strip(" -")
                    if p:
                        summary_items.append(p)
            continue
        if in_summary:
            if ln.startswith("핵심수치") or ln.startswith("메모"):
                in_summary = False
                continue
            if ln.startswith("-"):
                summary_items.append(ln.lstrip("- ").strip())

    # '핵심수치' 파싱 (번호 접두어만 제거)
    for ln in lines:
        if "값:" in ln and "|" in ln:
            ln2 = re.sub(r'^\s*\d+\)\s*', '', ln)  # ← 여기만 번호 제거
            parts = [p.strip() for p in ln2.split("|")]

            row = {"라벨": "", "값": "", "단위": "", "기간/기준": "", "근거영역": ""}

            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    k = k.strip()
                    v = v.strip()  # 값 앞자리 숫자 보존
                    if k in row:
                        row[k] = v

            if row.get("라벨") or row.get("값"):
                figure_rows.append(row)

    return summary_items[:5], figure_rows[:5]

def format_doc(doc, idx=None):
    meta = doc.metadata or {}
    src  = meta.get("source") or meta.get("file") or meta.get("path") or "unknown"
    page = meta.get("page_number") or meta.get("page") or "N/A"
    txt  = (doc.page_content or "").strip()
    head = f"[{idx}] {os.path.basename(src)} (p.{page})" if idx is not None else f"{os.path.basename(src)} (p.{page})"
    body = textwrap.shorten(txt, width=500, placeholder=" …")
    return head, body, meta

def render_doc_card(idx, doc):
    _, _, meta = format_doc(doc, idx)
    fname = _clean_filename(meta.get("source") or meta.get("file") or meta.get("path") or "unknown")
    page  = meta.get("page_number") or meta.get("page") or "N/A"
    lvl1  = meta.get("level1")
    lvl2  = meta.get("level2")

    summary_items, figure_rows = parse_summary_and_figures(doc.page_content or "")

    tags_html = '<div class="doc-tags">'
    tags_html += f'<span class="filepill">{html.escape(fname)}</span>'
    if lvl1: tags_html += f' <span class="badge">{html.escape(str(lvl1))}</span>'
    if lvl2: tags_html += f' <span class="badge">{html.escape(str(lvl2))}</span>'
    tags_html += f' <span class="badge">p.{html.escape(str(page))}</span></div>'

    # 요약 bullets
    if summary_items:
        items_html = "".join([f"<li>{html.escape(it)}</li>" for it in summary_items])
        sum_html = f'<ul class="sum-list">{items_html}</ul>'
    else:
        sum_html = '<div class="small">요약 정보 없음</div>'

    # 핵심수치 표 (라벨 컬럼 제거)
    if figure_rows:
        header = "<tr><th>값</th><th>단위</th><th>기간/기준</th><th>근거영역</th></tr>"
        rows = ""
        for r in figure_rows:
            rows += (
                "<tr>"
                f"<td>{html.escape(r.get('값',''))}</td>"
                f"<td>{html.escape(r.get('단위',''))}</td>"
                f"<td>{html.escape(r.get('기간/기준',''))}</td>"
                f"<td>{html.escape(r.get('근거영역',''))}</td>"
                "</tr>"
            )
        fig_html = f'<table class="kv-table">{header}{rows}</table>'
    else:
        fig_html = '<div class="small">핵심 수치 없음</div>'

    html_block = (
        f'<div class="doc-card">'
        f'<div class="doc-head">[{idx}] {html.escape(fname)} <span class="small">(p.{html.escape(str(page))})</span></div>'
        f'{tags_html}'
        f'<div class="small" style="margin-top:4px;"><b>핵심요약</b></div>'
        f'{sum_html}'
        f'<div class="small" style="margin-top:8px;"><b>핵심수치</b></div>'
        f'{fig_html}'
        f'</div>'
    )
    st.markdown(html_block, unsafe_allow_html=True)


# -----------------------------
# 🔄 위젯 생성 전 프리필 처리
# -----------------------------
if st.session_state.get("pending_query"):
    st.session_state["user_query"] = st.session_state.pop("pending_query")

# -----------------------------
# 🔎 검색 + 생성
# -----------------------------
query = st.text_input("질문을 입력하세요", key="user_query")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    run_btn = st.button("Ask", type="primary")
with col2:
    clear_btn = st.button("Clear (이 화면만)")
with col3:
    clear_hist = st.button("Recent 비우기")

if clear_btn:
    st.session_state["answer"] = ""
    st.session_state["retrieved_docs"] = []
    st.session_state["user_query"] = ""
    st.rerun()

if clear_hist:
    st.session_state["history"] = []
    st.rerun()

# 제안 질문 3개
def make_suggestions(q: str, docs):
    suggestions = []
    if q:
        suggestions.append(f"‘{q}’에 대한 핵심 근거만 항목별로 요약해줘")
        suggestions.append(f"‘{q}’와 관련된 반대 관점이나 잠재 리스크는 뭐가 있어?")
    src_name = None
    for d in (docs or []):
        meta = d.metadata or {}
        src  = meta.get("source") or meta.get("file") or meta.get("path")
        if src:
            src_name = os.path.basename(src)
            break
    if src_name:
        suggestions.append(f"{src_name} 문서 기준으로 구체 사례/수치를 뽑아줘")
    else:
        suggestions.append("관련 지표/수치를 표로 정리해줘")
    uniq = []
    for s in suggestions:
        if s not in uniq:
            uniq.append(s)
        if len(uniq) == 3:
            break
    return uniq

suggestions = make_suggestions(st.session_state.get("user_query",""), st.session_state.get("retrieved_docs", []))
with st.container():
    st.markdown('<div class="suggest-box"><b>혹시 이런게 궁금하신가요?</b></div>', unsafe_allow_html=True)
    scols = st.columns(3)
    for i, s in enumerate(suggestions):
        if s is None:
            continue
        if scols[i].button(f"🧩 {s}", key=f"sugg_{i}"):
            st.session_state["pending_query"] = s
            st.rerun()

if run_btn:
    if not os.path.isdir(chroma_dir):
        st.error(f"Chroma 디렉터리를 찾을 수 없습니다: {chroma_dir}")
        st.stop()

    # Embeddings / Vectorstore
    with st.spinner("임베딩/벡터스토어 로딩 중…"):
        embeddings = load_embeddings(embed_name, device_opt)
        try:
            db = load_vectorstore(chroma_dir, embeddings)
        except Exception as e:
            st.exception(e)
            st.stop()

    # 간단 통계(컬렉션 문서 수)
    try:
        count = db._collection.count()
        st.markdown(f'<div class="stat">📦 현재 컬렉션 문서 수: <b>{count}</b></div>', unsafe_allow_html=True)
    except Exception:
        pass

    retriever = db.as_retriever(search_kwargs={"k": top_k})

    with st.spinner("검색 중…"):
        docs = retriever.invoke(query) if query.strip() else []
        st.session_state["retrieved_docs"] = docs

    # context 만들기
    ctx_chunks, src_list = [], []
    for i, d in enumerate(docs, start=1):
        h, b, m = format_doc(d, i)
        ctx_chunks.append(f"{h}\n{b}")
        src = m.get("source") or m.get("file") or m.get("path") or "unknown"
        if src:
            src_list.append(os.path.basename(src))
    context_text = "\n\n---\n\n".join(ctx_chunks) if ctx_chunks else "N/A"

    # LLM 호출
    answer_text = ""
    if not api_key:
        st.info("OPENROUTER_API_KEY가 없어 LLM 호출을 생략합니다. 검색 결과만 표시합니다.")
    else:
        chain = build_chain(model_name, temperature, OPENROUTER_BASE_URL, api_key)
        with st.spinner("답변 생성 중…"):
            answer_text = chain.invoke({"context": context_text, "question": query}) or ""

    st.session_state["answer"] = answer_text

    # 히스토리 업데이트(맨 앞에 추가, 5개 유지)
    hist = st.session_state["history"]
    hist.insert(0, {
        "q": query,
        "a": answer_text,
        "k": top_k,
        "sources": list(dict.fromkeys(src_list))[:5],
    })
    st.session_state["history"] = hist[:5]
    st.rerun()


# -----------------------------
# 🧠 답변 표시
# -----------------------------
if st.session_state.get("answer"):
    st.markdown('<div class="chip">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-card">{st.session_state["answer"]}</div>', unsafe_allow_html=True)

# -----------------------------
# 📚 참고 문서 표시
# -----------------------------
if "retrieved_docs" in st.session_state:
    st.markdown("### 📚 참고 문서")
    docs = st.session_state["retrieved_docs"]
    if not docs:
        st.info("검색 결과가 없습니다. (DB가 비어있거나, 쿼리와 문서가 덜 유사할 수 있어요)")
    else:
        for i, d in enumerate(docs, start=1):
            render_doc_card(i, d)
