# streamlit_rag_app.py

import os
import re
import sys
import shutil
import subprocess
import textwrap
import html
from time import sleep
from datetime import datetime
from urllib.parse import unquote
import warnings
import gc
from contextlib import contextmanager

import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import json
import chromadb  # Chroma 1.x client

# ===== 전역 안정화: 경고/스레드 억제 =====
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


@contextmanager
def guard(title: str = "예기치 못한 오류가 발생했습니다."):
    """에러를 패널로 보여주고 안전 종료(st.stop)하는 컨텍스트"""
    try:
        yield
    except SystemExit:
        raise
    except Exception as e:
        # ✅ Streamlit 재실행용 예외는 그냥 통과시켜야 함
        if e.__class__.__name__ == "RerunException":
            raise
        st.error(title)
        st.exception(e)
        st.stop()


# -----------------------------
# 🔧 기본 설정
# -----------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# temp 폴더(단일 사용자 전제) + 포인터 방식
DEFAULT_CHROMA_DIR = str(Path(tempfile.gettempdir()) / "chroma_store")  # /tmp/chroma_store
DEFAULT_MODEL       = "google/gemma-2-9b-it"
EMBED_MODEL_NAME    = "nlpai-lab/KURE-v1"
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
# 🔧 포인터 유틸
# -----------------------------
def _pointer_file(root: str) -> Path:
    return Path(root) / "CURRENT.txt"

def _resolve_store_path(root: str) -> Path | None:
    p = _pointer_file(root)
    if not p.exists():
        return None
    try:
        sp = p.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return Path(sp) if sp and Path(sp).exists() else None

# -----------------------------
# 📅 날짜 추출 유틸(파일명/본문/요약)
# -----------------------------
_DATE_PATTERNS = [
    re.compile(r"\((\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\)"),     # (YYYY-MM-DD)
    re.compile(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})"),         # YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD
    re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"), # YYYY년 MM월 DD일
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),                       # YYYYMMDD
    re.compile(r"(\d{4})[.\-/](\d{1,2})"),                        # YYYY-MM
    re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월"),                 # YYYY년 MM월
    re.compile(r"(20\d{2})\s*년"),                                # YYYY년
    re.compile(r"(?<!\d)(20\d{2})(?!\d)"),                        # YYYY
]

_HALF_RE    = re.compile(r"(20\d{2})\s*(상반기|하반기|H[12])", re.IGNORECASE)
_QUARTER_RE = re.compile(r"(20\d{2})\s*(?:Q([1-4])|([1-4])\s*분기)", re.IGNORECASE)

def _mk_date(y, m=1, d=1) -> datetime | None:
    try:
        y, m, d = int(y), int(m), int(d)
        if not (1 <= m <= 12): m = 1
        if not (1 <= d <= 31): d = 1
        return datetime(y, m, d)
    except Exception:
        try:
            return datetime(int(y), int(m), 1)
        except Exception:
            return None

def _extract_dates_from_text(s: str) -> list[datetime]:
    if not s:
        return []
    out: list[datetime] = []
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(s):
            g = m.groups()
            if len(g) >= 3:
                dt = _mk_date(g[0], g[1], g[2])
            elif len(g) == 2:
                dt = _mk_date(g[0], g[1], 1)
            else:
                dt = _mk_date(g[0], 1, 1)
            if dt:
                out.append(dt)
    for m in _HALF_RE.finditer(s):
        y, tag = m.group(1), m.group(2).upper()
        month = 6 if tag in ("상반기", "H1") else 12
        dt = _mk_date(y, month, 1)
        if dt:
            out.append(dt)
    for m in _QUARTER_RE.finditer(s):
        y = m.group(1)
        q = m.group(2) or m.group(3)
        try:
            qi = int(q)
        except Exception:
            continue
        month = {1: 3, 2: 6, 3: 9, 4: 12}.get(qi, 12)
        dt = _mk_date(y, month, 1)
        if dt:
            out.append(dt)
    return out

# -----------------------------
# 🔧 유틸: 렌더 함수/파서
# -----------------------------
def _clean_filename(name: str) -> str:
    if not name:
        return "unknown"
    return unquote(name).replace("+", " ")

def parse_summary_and_figures(text: str):
    if not text:
        return [], []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    summary_items, figure_rows = [], []
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

    for ln in lines:
        if "값:" in ln and "|" in ln:
            ln2 = re.sub(r'^\s*\d+\)\s*', '', ln)
            parts = [p.strip() for p in ln2.split("|")]
            row = {"라벨": "", "값": "", "단위": "", "기간/기준": "", "근거영역": ""}
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    k = k.strip(); v = v.strip()
                    if k in row:
                        row[k] = v
            if row.get("라벨") or row.get("값"):
                figure_rows.append(row)
    return summary_items[:5], figure_rows[:5]

# 요약/본문/파일명 순으로 날짜 우선순위 평가
def get_doc_date_info(doc) -> tuple[int, datetime | None]:
    """
    return (rank, dt)
      rank: 0=summary, 1=text, 2=source(path), 3=none
    """
    meta = getattr(doc, "metadata", {}) or {}
    src  = meta.get("source") or meta.get("file") or meta.get("path") or ""
    txt  = (getattr(doc, "page_content", "") or "")

    # 1) 핵심요약에서 날짜 추출 (최우선)
    summary_items, _ = parse_summary_and_figures(txt)
    summary_text = " ".join(summary_items) if summary_items else ""
    cands_sum = _extract_dates_from_text(summary_text)
    if cands_sum:
        return (0, max(cands_sum))

    # 2) 본문 전체에서 추출
    cands_txt = _extract_dates_from_text(txt)
    if cands_txt:
        return (1, max(cands_txt))

    # 3) 파일명/경로에서 추출
    cands_src = _extract_dates_from_text(src)
    if cands_src:
        return (2, max(cands_src))

    # 4) 없음
    return (3, None)

def get_doc_datetime(doc) -> datetime | None:
    return get_doc_date_info(doc)[1]

def _fmt_dt(dt: datetime | None) -> str:
    if not dt:
        return ""
    if dt.day != 1:
        return dt.strftime("%Y-%m-%d")
    if dt.month != 1:
        return dt.strftime("%Y-%m")
    return dt.strftime("%Y")

def format_doc(doc, idx=None):
    meta = getattr(doc, "metadata", {}) or {}
    src  = meta.get("source") or meta.get("file") or meta.get("path") or "unknown"
    page = meta.get("page_number") or meta.get("page") or "N/A"
    txt  = (getattr(doc, "page_content", "") or "").strip()
    head = f"[{idx}] {os.path.basename(src)} (p.{page})" if idx is not None else f"{os.path.basename(src)} (p.{page})"
    body = textwrap.shorten(txt, width=500, placeholder=" …")
    return head, body, meta

def render_doc_card(idx, doc):
    head, body, meta = format_doc(doc, idx)
    fname = _clean_filename(meta.get("source") or meta.get("file") or meta.get("path") or "unknown")
    page  = meta.get("page_number") or meta.get("page") or "N/A"
    lvl1  = meta.get("level1")
    lvl2  = meta.get("level2")
    summary_items, figure_rows = parse_summary_and_figures(getattr(doc, "page_content", "") or "")
    dt    = get_doc_datetime(doc)
    dt_badge = _fmt_dt(dt)

    if summary_items:
        items_html = "".join([f"<li>{html.escape(it)}</li>" for it in summary_items])
        sum_html = f'<ul class="sum-list">{items_html}</ul>'
    else:
        sum_html = '<div class="small">요약 정보 없음</div>'

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
        fig_table = f'<table class="kv-table">{header}{rows}</table>'
        fig_html = (
            "<details>"
            "<summary class='small' style='cursor:pointer;'>핵심수치 보기</summary>"
            f"{fig_table}"
            "</details>"
        )
    else:
        fig_html = '<div class="small">핵심 수치 없음</div>'

    tags_html = '<div class="doc-tags">'
    tags_html += f'<span class="filepill">{html.escape(fname)}</span>'
    if lvl1: tags_html += f' <span class="badge">{html.escape(str(lvl1))}</span>'
    if lvl2: tags_html += f' <span class="badge">{html.escape(str(lvl2))}</span>'
    tags_html += f' <span class="badge">p.{html.escape(str(page))}</span>'
    if dt_badge:
        tags_html += f' <span class="badge">{html.escape(dt_badge)}</span>'
    tags_html += '</div>'

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
# 🎨 페이지 스타일
# -----------------------------
st.set_page_config(page_title="RAG Q&A", page_icon="🔎", layout="wide")

CUSTOM_CSS = """
<style>
header {visibility: hidden;}
.answer-card { border-radius: 16px; padding: 18px 20px; background: #ffffff;
  border: 1px solid rgba(0,0,0,0.08); box-shadow: 0 6px 20px rgba(0,0,0,0.06); }
.chip { display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff;
  color:#3949ab; font-weight:600; font-size:12px; margin-bottom:6px; }
.doc-card { border-radius: 14px; padding: 12px 14px; background:#fafafa; border:1px solid #eee; margin-bottom:12px; }
.doc-head {font-weight:700; font-size:14px;}
.doc-body {font-size:13px; color:#333;}
.small {color:#666; font-size:12px;}
.footer {color:#777; font-size:12px; margin-top:8px;}
.stat {color:#444; font-size:13px; margin-bottom:6px;}
.history-card { border-radius: 12px; padding:10px 12px; background:#f6f7fb; border:1px solid #e6e8f5; margin-bottom:8px; }
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

# -----------------------------
# 🗂️ 세션 상태 초기화
# -----------------------------
for key, default in {
    "history": [],
    "retrieved_docs": [],
    "answer": "",
    "user_query": "",
    "pending_query": None,
    "db_built": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------
# 🧰 사이드바
# -----------------------------
with st.sidebar:
    st.subheader("Settings")

    chroma_dir  = st.text_input("Chroma root directory (pointer)", value=DEFAULT_CHROMA_DIR)
    pkl_path    = st.text_input("PKL 경로(샘플)", value=str(BASE_DIR / "data" / "df_sample_pages.pkl"))

    top_k       = st.slider("Top-K (retrieval)", 1, 10, 3)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2)
    model_name  = st.text_input("LLM (OpenRouter)", value=DEFAULT_MODEL)
    embed_name  = st.text_input("Embedding model", value=EMBED_MODEL_NAME)
    device_opt  = st.selectbox("Embedding device", ["cpu", "auto (cuda if available)", "cuda"], index=0)

    st.caption("※ OPENROUTER_API_KEY 환경변수 설정 시 LLM 호출 활성화")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    st.divider()
    st.subheader("Index")
    rebuild_on_start = st.checkbox("앱 시작 시 강제 재생성", value=False)
    rebuild_now = st.button("🔨 인덱스 재생성 (수동)")

    st.divider()
    st.subheader("🕘 Recent (last 5)")
    if not st.session_state["history"]:
        st.caption("최근 Q&A 기록이 없습니다.")
    else:
        for idx, item in enumerate(st.session_state["history"][:5]):
            q_short = textwrap.shorten(item["q"], width=60, placeholder=" …")
            a_short = textwrap.shorten(item.get("a",""), width=70, placeholder=" …") if item.get("a") else "(생성 없음)"
            st.markdown(
                f'<div class="history-card"><div class="history-q">Q: {html.escape(q_short)}</div>'
                f'<div class="history-a">A: {html.escape(a_short)}</div></div>',
                unsafe_allow_html=True
            )
            if st.button("↺ 이 질문 불러오기", key=f"load_hist_{idx}"):
                st.session_state["pending_query"] = item["q"]
                st.rerun()

# -----------------------------
# 🧱 DB 재생성 유틸 (실시간 로그/포인터 갱신)
# -----------------------------
def _device_arg(choice: str) -> str:
    if choice.startswith("auto"):
        return "auto"
    return choice

def rebuild_db_always(pkl_path: str, chroma_dir: str, collection: str, embed_model: str, device_choice: str):
    try:
        st.cache_resource.clear()
    except Exception:
        pass

    vs_script = Path(__file__).resolve().parent / "02_Vectorstore.py"
    if not vs_script.is_file():
        raise FileNotFoundError(f"02_Vectorstore.py를 찾을 수 없습니다: {vs_script}")

    env = os.environ.copy()
    env.pop("CHROMA_DB_IMPL", None)
    env["TOKENIZERS_PARALLELISM"] = "false"

    cmd = [
        sys.executable, str(vs_script),
        "--root", chroma_dir,
        "--collection", collection,
        "--embed_model", embed_model,
        "--device", _device_arg(device_choice),
        "--pkl", pkl_path,
    ]
    cmd_str = " ".join(cmd)

    use_status = hasattr(st, "status")
    log_box = st.empty()

    def _append_log(lines, new_line):
        lines.append(new_line.rstrip())
        pre = "<pre style='max-height:280px;overflow:auto;'>" + html.escape("\n".join(lines[-300:])) + "</pre>"
        log_box.markdown(pre, unsafe_allow_html=True)

    if use_status:
        with st.status("초기 벡터 DB 재생성 중…(포인터 갱신)", expanded=True) as status:
            status.write("1) 캐시 초기화 완료")
            status.write(f"2) 빌드 스크립트 실행: `{cmd_str}`")

            lines = []
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env
            )
            for line in proc.stdout:
                _append_log(lines, line)
            rc = proc.wait()

            if rc != 0:
                out_str = "\n".join(lines)
                status.update(state="error", label="빌드 실패")
                raise RuntimeError("02_Vectorstore 실행 실패\nCMD: " + cmd_str + "\nSTDOUT/ERR:\n" + out_str)

            status.write("3) 포인터(CURRENT.txt) 갱신 확인 중…")
            pointer = Path(chroma_dir) / "CURRENT.txt"
            if not pointer.exists():
                status.update(state="error", label="포인터 파일 누락")
                raise FileNotFoundError(f"포인터 파일이 없습니다: {pointer}")

            store_path = pointer.read_text(encoding="utf-8").strip()
            if not store_path or not Path(store_path).exists():
                status.update(state="error", label="포인터 경로 불량")
                raise FileNotFoundError(f"포인터가 가리키는 경로가 없습니다: {store_path}")

            status.update(state="complete", label="초기 벡터 DB 재생성 완료")
            return True
    else:
        with st.spinner("초기 벡터 DB 재생성 중…(포인터 갱신)"):
            lines = []
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env
            )
            for line in proc.stdout:
                _append_log(lines, line)
            rc = proc.wait()
            if rc != 0:
                out_str = "\n".join(lines)
                raise RuntimeError("02_Vectorstore 실행 실패\nCMD: " + cmd_str + "\nSTDOUT/ERR:\n" + out_str)

            pointer = Path(chroma_dir) / "CURRENT.txt"
            if not pointer.exists():
                raise FileNotFoundError(f"포인터 파일이 없습니다: {pointer}")

            store_path = pointer.read_text(encoding="utf-8").strip()
            if not store_path or not Path(store_path).exists():
                raise FileNotFoundError(f"포인터가 가리키는 경로가 없습니다: {store_path}")

            st.success("초기 벡터 DB 재생성 완료")
            return True

# -----------------------------
# 🏁 앱 시작 시점: 포인터 확인 & 선택적 재빌드
# -----------------------------
with guard("인덱스 초기화 중 오류가 발생했습니다."):
    pointer_ok = _resolve_store_path(
        DEFAULT_CHROMA_DIR if 'chroma_dir' not in st.session_state else st.session_state.get('chroma_dir', DEFAULT_CHROMA_DIR)
    ) is not None
    need_build = rebuild_on_start or (_resolve_store_path(DEFAULT_CHROMA_DIR) is None)

    if not st.session_state["db_built"]:
        if need_build:
            try:
                rebuild_db_always(
                    pkl_path=pkl_path,
                    chroma_dir=chroma_dir,
                    collection="amore_v1",
                    embed_model=EMBED_MODEL_NAME,
                    device_choice=device_opt
                )
                st.session_state["db_built"] = True
            except Exception as e:
                st.error("인덱스 재생성 실패 — 아래 로그 확인 후 필요 시 다시 시도하세요.")
                st.exception(e)
                if _resolve_store_path(chroma_dir) is None:
                    st.stop()
        else:
            st.session_state["db_built"] = True

    if rebuild_now:
        try:
            rebuild_db_always(
                pkl_path=pkl_path,
                chroma_dir=chroma_dir,
                collection="amore_v1",
                embed_model=EMBED_MODEL_NAME,
                device_choice=device_opt
            )
            st.success("인덱스를 재생성했습니다.")
        except Exception as e:
            st.error("인덱스 재생성 실패")
            st.exception(e)

# -----------------------------
# ♻️ 리소스 로드 (cache)
# -----------------------------
def _normalize_device(dev_choice: str) -> str:
    if dev_choice.startswith("auto"):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return dev_choice

@st.cache_resource(show_spinner=True)
def load_embeddings(model_name: str, device_choice: str):
    dev = _normalize_device(device_choice)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": dev},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner=True)
def load_vectorstore(persist_root: str, embed_model: str, device_choice: str, collection: str = "amore_v1"):
    emb = load_embeddings(embed_model, device_choice)

    pointer = Path(persist_root) / "CURRENT.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            f"포인터 파일이 없습니다: {pointer}\n"
            f"사이드바의 '인덱스 재생성'으로 먼저 만들세요."
        )
    store_path = pointer.read_text(encoding="utf-8").strip()
    if not store_path or not Path(store_path).exists():
        raise FileNotFoundError(
            f"포인터가 가리키는 경로가 없습니다: {store_path}\n"
            f"사이드바에서 인덱스를 다시 생성하세요."
        )

    client = chromadb.PersistentClient(path=store_path)  # 1.x 방식
    db = Chroma(client=client, embedding_function=emb, collection_name=collection)
    return db, emb

@st.cache_resource(show_spinner=True)
def build_chain(model_name: str, temperature: float, base_url: str, api_key: str):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT_TEMPLATE)]
    )
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    parser = StrOutputParser()
    return prompt | llm | parser

# -----------------------------
# 🔄 위젯 생성 전 프리필 처리
# -----------------------------
if st.session_state.get("pending_query"):
    st.session_state["user_query"] = st.session_state.pop("pending_query")

# -----------------------------
# 🔎 검색 + 생성
# -----------------------------
query = st.text_input("질문을 입력하세요", key="user_query")

st.markdown(
    """
    <div class="suggest-box">
      <b>질문 예</b><br/>
      1. 코로나19 이후 소비자 트렌드에 대해 알려줘<br/>
      2. 기능성 화장품의 시장 규모와 전망을 알려줘<br/>
      3. 2024년 뷰티 산업 주요트렌드가 뭐야?
    </div>
    """,
    unsafe_allow_html=True
)

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
    st.session_state["_do_rerun"] = True

if clear_hist:
    st.session_state["history"] = []
    st.session_state["_do_rerun"] = True

def make_suggestions(q: str, docs):
    suggestions = []
    if q:
        suggestions.append(f"‘{q}’에 대한 핵심 근거만 항목별로 요약해줘")
        suggestions.append(f"‘{q}’와 관련된 반대 관점이나 잠재 리스크는 뭐가 있어?")
    src_name = None
    for d in (docs or []):
        meta = getattr(d, "metadata", {}) or {}
        src  = meta.get("source") or meta.get("file") or meta.get("path")
        if src:
            src_name = os.path.basename(src)
            break
    if src_name:
        suggestions.append(f"{os.path.basename(src_name)} 문서 기준으로 구체 사례/수치를 뽑아줘")
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
        # 추천 질문 버튼
        if scols[i].button(f"🧩 {s}", key=f"sugg_{i}"):
            st.session_state["pending_query"] = s
            st.session_state["_do_rerun"] = True

with guard("검색/답변 생성 중 오류가 발생했습니다."):
    if run_btn:
        with st.spinner("임베딩/벡터스토어 로딩 중…"):
            try:
                db, embeddings = load_vectorstore(chroma_dir, embed_name, device_opt, collection="amore_v1")
            except Exception as e:
                st.exception(e)
                st.stop()

        # 임베딩 차원 검증
        try:
            qdim = len(embeddings.embed_query("ping"))
            pointer = Path(chroma_dir) / "CURRENT.txt"
            store_path = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None
            meta_f = os.path.join(store_path, "meta.json") if store_path else None
            if meta_f and os.path.exists(meta_f):
                with open(meta_f, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "dim" in meta and meta["dim"] != qdim:
                    st.error(f"임베딩 차원 불일치: DB={meta['dim']} vs 현재 모델={qdim}. "
                             f"DB 생성 시 사용한 임베딩과 동일한 모델을 선택하세요.")
                    st.stop()
        except Exception as e:
            st.exception(e)
            st.stop()

        # 컬렉션 문서 수
        try:
            count = db._collection.count()
            st.markdown(f'<div class="stat">📦 현재 컬렉션 문서 수: <b>{count}</b></div>', unsafe_allow_html=True)
            if count == 0:
                st.warning("Chroma 컬렉션이 비어 있습니다. 인덱싱 설정을 확인하세요.")
        except Exception:
            pass

        retriever = db.as_retriever(search_kwargs={"k": top_k})

        with st.spinner("검색 중…"):
            docs = retriever.invoke(query) if query.strip() else []
            st.session_state["retrieved_docs"] = docs

        # context 만들기
        ctx_chunks, src_list = [], []
        for i, d in enumerate(docs, start=1):
            head, body, meta = format_doc(d, i)
            ctx_chunks.append(f"{head}\n{body}")
            src = meta.get("source") or meta.get("file") or meta.get("path") or "unknown"
            if src:
                src_list.append(os.path.basename(src))
        context_text = "\n\n---\n\n".join(ctx_chunks) if ctx_chunks else "N/A"

        # LLM 호출
        answer_text = ""
        if not api_key:
            st.info("OPENROUTER_API_KEY가 없어 LLM 호출을 생략합니다. 검색 결과만 표시합니다.")
        else:
            try:
                chain = build_chain(model_name, temperature, OPENROUTER_BASE_URL, api_key)
                with st.spinner("답변 생성 중…"):
                    answer_text = chain.invoke({"context": context_text, "question": query}) or ""
            except Exception as e:
                st.exception(e)
                answer_text = ""

        st.session_state["answer"] = answer_text

        # 히스토리 업데이트
        hist = st.session_state["history"]
        hist.insert(0, {"q": query, "a": answer_text, "k": top_k, "sources": list(dict.fromkeys(src_list))[:5]})
        st.session_state["history"] = hist[:5]
        st.session_state["_do_rerun"] = True    # ✅ 플래그만
        

# -----------------------------
# 🧠 답변 표시
# -----------------------------
if st.session_state.get("answer"):
    st.markdown('<div class="chip">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-card">{st.session_state["answer"]}</div>', unsafe_allow_html=True)

# -----------------------------
# 📚 참고 문서 표시 (최신순)
# -----------------------------
with guard("참고 문서 렌더링 중 오류가 발생했습니다."):
    if "retrieved_docs" in st.session_state:
        st.markdown("### 📚 참고 문서")
        docs = st.session_state["retrieved_docs"]
        if not docs:
            st.info("검색 결과가 없습니다. (DB가 비어있거나, 쿼리와 문서가 덜 유사할 수 있어요)")
        else:
            # (date desc). 오직 날짜 최신순으로 정렬
            def _sort_key(d):
                _, dt = get_doc_date_info(d)  # rank 정보는 사용하지 않음
                ts = -dt.timestamp() if dt else 0  # 날짜가 최신일수록 큰 timestamp 값을 가지므로, 음수로 만들어 오름차순 정렬 (결과적으로 내림차순)
                return ts

            docs_sorted = sorted(docs, key=_sort_key)

            for i, d in enumerate(docs_sorted, start=1):
                render_doc_card(i, d)

# ---- 마지막 정리(메모리 소량 회수) ----
try:
    gc.collect()
except Exception:
    pass

# ✅ 여기서만 실제 재실행 트리거
if st.session_state.pop("_do_rerun", False):
    st.rerun()
