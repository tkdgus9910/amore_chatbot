# -*- coding: utf-8 -*-
"""
02_Vectorstore.py — ChromaDB 1.x 전용 빌더 (Windows 안전, 내부 설정 기본)
- 상단 CONFIG_DEFAULTS에서 PKL/루트경로/모델/옵션을 직접 설정
- 빌드할 때마다 ROOT 아래에 고유 버전 폴더 생성
- ROOT/CURRENT.txt 포인터를 최신 버전 폴더로 원자 갱신
- LangChain Chroma는 chromadb.PersistentClient(path=...)를 client=로 주입
- 수동 persist() 호출 금지 (1.x + client 사용 시 자동 저장)
"""

import os
import sys
import json
import pickle
import argparse
import tempfile
import uuid
import traceback
from datetime import datetime
from pathlib import Path

# =========================
# CONFIG: 기본값 (여기 바꿔서 사용)
# =========================
CONFIG_DEFAULTS = {
    # PKL 경로 (필요에 따라 아래 경로 중 하나로 조정)
    #"PKL_PATH": r"C:\Users\PC1\OneDrive\250801_아모레\df_sample_pages_old.pkl",
    # "PKL_PATH": r"C:/Users/PC1/amore_chatbot/chatbot_repo/data/df_sample_pages.pkl",
    "PKL_PATH": str(Path(__file__).resolve().parent / "data" / "df_sample_pages.pkl"),

    # Chroma 버전 폴더들을 모아둘 루트 (포인터 CURRENT.txt가 여기에 생김)
    "CHROMA_ROOT": str(Path(tempfile.gettempdir()) / "chroma_store"),

    # 임베딩/컬렉션/디바이스
    "EMBED_MODEL": "nlpai-lab/KURE-v1",
    "COLLECTION": "amore_v1",
    "DEVICE": "auto",  # auto | cpu | cuda

    # 청크/배치
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 100,
    "BATCH_SIZE": 16,  # 임베딩 배치
}

# =========================
# 유틸
# =========================
def eprint(*a): print(*a, file=sys.stderr)

def print_versions():
    try:
        import chromadb, langchain, langchain_community
        print(
            "[versions] python={py} chromadb={cver} langchain={lver} lc_community={lcver}".format(
                py=sys.version.split()[0],
                cver=getattr(chromadb, "__version__", "unknown"),
                lver=getattr(langchain, "__version__", "unknown"),
                lcver=getattr(langchain_community, "__version__", "unknown"),
            )
        )
    except Exception:
        pass
    print("[env] TOKENIZERS_PARALLELISM={}".format(os.environ.get("TOKENIZERS_PARALLELISM")))

def print_disk_info(path: str):
    try:
        import shutil
        usage = shutil.disk_usage(path)
        print("[disk] total={}GB, free={}GB".format(usage.total//(1024**3), usage.free//(1024**3)))
    except Exception:
        pass

# =========================
# (선택) 인자 파서 — 주면 덮어쓰기, 안 주면 위 기본값 사용
# =========================
def parse_args_optional():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pkl", type=str)
    ap.add_argument("--root", type=str)
    ap.add_argument("--collection", type=str)
    ap.add_argument("--embed_model", type=str)
    ap.add_argument("--device", type=str, choices=["auto","cpu","cuda"])
    ap.add_argument("--chunk_size", type=int)
    ap.add_argument("--chunk_overlap", type=int)
    ap.add_argument("--batch_size", type=int)
    try:
        args, _ = ap.parse_known_args()
    except SystemExit:
        class _Empty: pass
        args = _Empty()
    return args

def resolve_config():
    args = parse_args_optional()
    cfg = dict(CONFIG_DEFAULTS)
    if getattr(args, "pkl", None):           cfg["PKL_PATH"]       = args.pkl
    if getattr(args, "root", None):          cfg["CHROMA_ROOT"]    = args.root
    if getattr(args, "collection", None):    cfg["COLLECTION"]     = args.collection
    if getattr(args, "embed_model", None):   cfg["EMBED_MODEL"]    = args.embed_model
    if getattr(args, "device", None):        cfg["DEVICE"]         = args.device
    if getattr(args, "chunk_size", None):    cfg["CHUNK_SIZE"]     = args.chunk_size
    if getattr(args, "chunk_overlap", None): cfg["CHUNK_OVERLAP"]  = args.chunk_overlap
    if getattr(args, "batch_size", None):    cfg["BATCH_SIZE"]     = args.batch_size
    return cfg

# =========================
# 메인
# =========================
def main():
    # 안정성: 토크나이저/BLAS 스레드 수 제한 (Oops 완화)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cfg = resolve_config()

    print_versions()
    print(
        "[config]\n"
        "  pkl={pkl}\n"
        "  root={root}\n"
        "  collection={col}\n"
        "  embed_model={em}\n"
        "  device={dev}\n"
        "  chunk_size={cs} chunk_overlap={co}\n"
        "  batch_size={bs}".format(
            pkl=cfg["PKL_PATH"],
            root=cfg["CHROMA_ROOT"],
            col=cfg["COLLECTION"],
            em=cfg["EMBED_MODEL"],
            dev=cfg["DEVICE"],
            cs=cfg["CHUNK_SIZE"], co=cfg["CHUNK_OVERLAP"],
            bs=cfg["BATCH_SIZE"],
        )
    )

    # 0) 입력 로드 (DataFrame)
    try:
        with open(cfg["PKL_PATH"], "rb") as f:
            df = pickle.load(f)
    except Exception:
        eprint("❌ PKL 로드 실패: {}".format(cfg["PKL_PATH"]))
        traceback.print_exc()
        sys.exit(2)

    # 0-1) 컬럼명 정리: VQA 컬럼 ↔ text 스왑 or 리네임
    VQA_COL  = "VQA_result_google/gemma-3-27b-it"
    TEXT_COL = "text"
    try:
        cols = set(getattr(df, "columns", []))
        if not cols:
            raise ValueError("DataFrame 컬럼 정보를 확인할 수 없습니다.")
        if VQA_COL in cols and TEXT_COL in cols:
            # 두 컬럼 모두 있을 경우 '이름 스왑' → 항상 'text' 컬럼을 본문으로 사용
            _tmp = "__tmp_swap_text__"
            while _tmp in cols:
                _tmp += "_"
            df.rename(columns={TEXT_COL: _tmp}, inplace=True)
            df.rename(columns={VQA_COL: TEXT_COL}, inplace=True)
            df.rename(columns={_tmp: VQA_COL}, inplace=True)
            print("[rename] 스왑 완료: '{}' ↔ '{}'".format(VQA_COL, TEXT_COL))
        elif VQA_COL in cols and TEXT_COL not in cols:
            # VQA → text
            df.rename(columns={VQA_COL: TEXT_COL}, inplace=True)
            print("[rename] '{}' → '{}' 로 변경".format(VQA_COL, TEXT_COL))
        elif TEXT_COL in cols and VQA_COL not in cols:
            # 이미 text만 있으면 유지
            print("[rename] '{}' 컬럼만 존재 — 변경 없음".format(TEXT_COL))
        else:
            raise KeyError("'{}' 또는 '{}' 컬럼이 필요합니다. 현재 컬럼 일부: {}".format(
                VQA_COL, TEXT_COL, sorted(cols)[:20]
            ))
    except Exception as e:
        print("⚠️ 컬럼명 스왑/정리 중 경고: {}".format(e))

    # 1) LangChain Documents
    print("\n추출된 내용을 LangChain Document로 래핑합니다...")
    from langchain.docstore.document import Document
    docs = []
    for _, row in df.iterrows():
        text = row.get(TEXT_COL)  # 항상 'text'를 기준으로 사용
        if text is None:
            continue
        if isinstance(text, float):  # NaN 등
            continue
        text = str(text).strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "level1": row.get("level1"),
                "level2": row.get("level2"),
                "source": row.get("source"),
                "page_number": row.get("page_number"),
            }
        ))
    print("총 {}개의 LangChain Document를 생성했습니다.".format(len(docs)))
    if docs:
        print("첫 번째 Document 예시:", docs[0])

    # 2) Split
    print("\nDocument를 청크로 분할합니다...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["CHUNK_SIZE"],
        chunk_overlap=cfg["CHUNK_OVERLAP"],
    )
    chunks = splitter.split_documents(docs)
    print("총 {}개의 청크로 분할되었습니다.".format(len(chunks)))

    # 3) Embeddings
    print("임베딩 모델을 로드합니다 (auto: cuda if available)...")
    from langchain_huggingface import HuggingFaceEmbeddings
    if cfg["DEVICE"] == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = cfg["DEVICE"]

    embeddings = HuggingFaceEmbeddings(
        model_name=cfg["EMBED_MODEL"],
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": cfg["BATCH_SIZE"]},
    )
    try:
        probe = embeddings.embed_query("ping")
        embed_dim = len(probe)
    except Exception:
        eprint("❌ 임베딩 차원 계산 실패")
        traceback.print_exc()
        sys.exit(3)

    print("임베딩 모델: {} | device={} | dim={}".format(cfg["EMBED_MODEL"], device, embed_dim))

    # 4) Chroma 1.x 빌드 (버전 폴더 + 포인터)
    print("\n=== ChromaDB 빌드 시작 (1.x, 버전 폴더 + CURRENT 포인터) ===")
    import chromadb
    from langchain_community.vectorstores import Chroma

    ROOT = Path(cfg["CHROMA_ROOT"]).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    POINTER = ROOT / "CURRENT.txt"
    print("[root ] {}".format(ROOT))
    print_disk_info(str(ROOT))

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    build_dir = ROOT / "{}__{}".format(stamp, uuid.uuid4().hex)
    build_dir.mkdir(parents=True, exist_ok=True)
    print("[build] {}".format(build_dir))

    try:
        client_build = chromadb.PersistentClient(path=str(build_dir))
        print("벡터 저장(빌드 폴더) 시작…")
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=cfg["COLLECTION"],
            client=client_build,  # ✅ 1.x 방식
        )
        _ = vs  # silence linters
    except Exception:
        eprint("❌ Chroma.from_documents(빌드) 실패")
        traceback.print_exc()
        sys.exit(5)

    # meta.json 저장
    meta = {
        "embed_model": cfg["EMBED_MODEL"],
        "dim": embed_dim,
        "collection": cfg["COLLECTION"],
        "chunk_size": cfg["CHUNK_SIZE"],
        "chunk_overlap": cfg["CHUNK_OVERLAP"],
        "device": device,
        "api": "chroma-1.x",
        "store_path": str(build_dir),
    }
    try:
        with open(build_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("[meta] 저장 완료 → {}".format(build_dir / "meta.json"))
    except Exception:
        eprint("⚠️ meta.json 저장 실패(계속)")
        traceback.print_exc()

    # CURRENT 포인터 원자적 갱신
    try:
        tmp_ptr = POINTER.with_suffix(".tmp")
        with open(tmp_ptr, "w", encoding="utf-8") as f:
            f.write(str(build_dir))
        os.replace(tmp_ptr, POINTER)
        print("[pointer] CURRENT -> {}".format(build_dir))
    except Exception:
        eprint("❌ CURRENT 포인터 갱신 실패")
        traceback.print_exc()
        sys.exit(6)

    # 5) 재로딩/검색 테스트
    print("\n--- 저장된 DB로 검색 테스트 ---")
    try:
        current_path = Path(POINTER.read_text(encoding="utf-8").strip())
        client2 = chromadb.PersistentClient(path=str(current_path))
        db = Chroma(client=client2, embedding_function=embeddings, collection_name=cfg["COLLECTION"])
        q = "코로나19 이후 소비자 트렌드에 대해 알려줘"
        print("질문: {}\n".format(q))
        res = db.similarity_search_with_score(q, k=3)
        print("--- 검색 결과 ---")
        if not res:
            print("검색 결과가 없습니다.")
        else:
            for i, (doc, score) in enumerate(res, 1):
                print("[결과 {}] (유사도 점수: {:.4f})".format(i, score))
                print("  - 내용: {}...".format(doc.page_content[:300]))
                print("  - 출처: {}".format(doc.metadata.get("source", "N/A")))
                print("  - 페이지: {}\n".format(doc.metadata.get("page_number", "N/A")))
    except Exception:
        eprint("⚠️ 재로딩/검색 테스트 실패(계속 진행 가능)")
        traceback.print_exc()

    print("\n✅ 빌드 완료")

# ===== 실행 진입부(안전 래퍼) =====
def _run_as_script():
    import warnings, time, gc
    start = time.time()

    # 전역 경고 억제(특히 numpy DeprecationWarning 소음 제거)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # 필요 시 특정 경고만 추가로 억제하고 싶으면 아래 라인처럼 패턴 지정 가능
        # warnings.filterwarnings("ignore", category=UserWarning, message=r".*SentenceTransformer.*")

        try:
            main()
        except KeyboardInterrupt:
            eprint("\n⛔️ 작업이 사용자에 의해 취소되었습니다 (KeyboardInterrupt).")
            sys.exit(130)  # SIGINT 관례
        except SystemExit:
            # main 내부에서 sys.exit()를 사용한 경우 그대로 전달
            raise
        except Exception:
            eprint("❌ 최상위 예외:")
            traceback.print_exc()
            sys.exit(1)

    # 실행 후 정리/요약 출력
    gc.collect()
    elapsed = time.time() - start
    print("\nGC 완료")
    print(" Elapsed: {:.1f}s".format(elapsed))
    # main()이 정상 종료했다면 여기까지 왔다는 뜻
    # (main 안에서도 '✅ 빌드 완료'를 출력하지만, 종합 요약을 한 번 더 남김)
    sys.exit(0)


if __name__ == "__main__":
    _run_as_script()