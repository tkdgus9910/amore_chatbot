# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:25:02 2025

@author: tmlab
"""

#%% 01. 임베딩 및 DB 로드
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import requests
from functools import partial

# LangChain 관련 라이브러리
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================================================================
# 1. OpenRouter API 설정 및 호출 함수
# ================================================================

# ❗ 중요: OpenRouter API 키를 설정하세요.
# os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-..."
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

def ask_openrouter(question: str, model: str, temperature: float = 0.1) -> str:
    """
    OpenAI 호환 chat/completions 형식으로 텍스트 언어 모델에 질의합니다.
    (RAG 파이프라인에 통합하기 위해 파라미터 순서를 조정했습니다.)
    """
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in Korean."
            },
            {
                "role": "user",
                "content": question # RAG 프롬프트가 포함된 전체 질문이 이곳으로 전달됩니다.
            }
        ]
    }
    
    if not OPENROUTER_API_KEY:
        return "오류: OpenRouter API 키가 설정되지 않았습니다."

    try:
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"API 요청 중 오류 발생: {e}"
    except (KeyError, IndexError):
        return f"API 응답 처리 중 오류 발생: {resp.text}"

#%% 02. retriever 준비 

# ================================================================
# 2. Retriever 준비 (기존 코드)
# ================================================================

# 임베딩 모델 로드
print("임베딩 모델을 로드합니다...")
model_name = "dragonkue/BGE-m3-ko" # 확정 
# model_name = "nlpai-lab/KoE5"
# model_name = "nlpai-lab/KURE-v1"


model_kwargs = {'device': 'cuda'} # GPU가 없다면 'cpu'로 변경
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from pathlib import Path
import tempfile
from langchain_community.vectorstores import Chroma


chroma_persist_dir = "chroma_db_V1"
db = Chroma(
    persist_directory=chroma_persist_dir,
    embedding_function=embeddings,
    # collection_name="amore_v1"
)
retriever = db.as_retriever(search_kwargs={"k": 3})
print("DB를 Retriever로 설정했습니다.\n")

#%% 03. 파이프라인 구성

# ================================================================
# 3. RAG 파이프라인 구성 (Context 포함하도록 수정)
# ================================================================

# 3-1. Prompt Template 및 LLM 준비 (이전과 동일)
template = """
주어진 맥락(Context) 정보를 사용하여 다음 질문에 답변해 주세요.
맥락에서 답을 찾을 수 없다면, "제공된 정보만으로는 답변하기 어렵습니다."라고 답하세요. 답변은 한국어로 간결하게 작성해주세요.

[맥락]
{context}

[질문]
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
selected_model = "google/gemma-2-9b-it"

llm = RunnableLambda(lambda p: ask_openrouter(question=p.to_string(), model=selected_model))

# 3-2. ⭐️⭐️⭐️ 최종 체인 구성 (가장 큰 변경점) ⭐️⭐️⭐️

# 검색된 문서(context)를 후속 체인에 전달하는 함수
def format_docs(docs):
    return "\n\n".join(f"--- 문서 {i+1} ---\n{doc.page_content}" for i, doc in enumerate(docs))

# 1. 질문을 받아 문서를 검색하고, context와 question을 딕셔너리로 만듦
setup_and_retrieval = RunnablePassthrough.assign(
    context=lambda x: format_docs(retriever.invoke(x["question"]))
)

# 2. context와 question을 받아 답변을 생성하는 체인
rag_chain_from_docs = (
    prompt
    | llm
    | StrOutputParser()
)

# 3. 최종적으로 context와 answer를 함께 반환하는 체인
final_chain = setup_and_retrieval | RunnablePassthrough.assign(
    answer=rag_chain_from_docs
)

# 04.RAG 파이프라인 실행 (출력 방식 변경)

# ================================================================
# 4. RAG 파이프라인 실행 (출력 방식 변경)
# ================================================================
if __name__ == "__main__":
    query = '코로나19 이후 소비자 트렌드에 대해 알려줘'
    print(f"질문: {query}\n")
    print("--- RAG 파이프라인 답변 생성 중 ---")

    # 체인을 실행하면 'context'와 'answer'가 포함된 딕셔너리를 반환
    result = final_chain.invoke({"question": query})

    # 최종 답변 출력
    print("\n✅ [최종 답변]")
    print(result["answer"])

    # 참고한 원문(Context) 출력
    print("\n\n📚 [참고 원문]")
    print(result["context"])



#%% 05. Retrieval 평가 모듈 (확장판)

import numpy as np
import pandas as pd
from datetime import datetime

# ================================================================
# 1. 확장된 평가 데이터셋 (3개 문서 대상)
# ================================================================

eval_dataset = [
    # ============================================================
    # 문서1: 랑콤 UV 엑스퍼트 톤업 밀크 로지블룸
    # ============================================================
    # Easy
    {"query": "랑콤 UV 엑스퍼트 톤업 밀크 성분", 
     "keywords": ["랑콤", "UV", "톤업"], "difficulty": "easy", "doc_id": "lancome_uv_expert"},
    {"query": "랑콤 로지블룸 가격 용량", 
     "keywords": ["98,000", "50 mL"], "difficulty": "easy", "doc_id": "lancome_uv_expert"},
    # Medium
    {"query": "EHMC 제거한 저자극 선크림 처방 사례", 
     "keywords": ["EHMC", "저자극"], "difficulty": "medium", "doc_id": "lancome_uv_expert"},
    {"query": "5중 차단 클레임 선케어 제품", 
     "keywords": ["UVA", "UVB", "미세먼지", "담배연기"], "difficulty": "medium", "doc_id": "lancome_uv_expert"},
    # Hard
    {"query": "EHT BEMT PBSA 조합 UV 필터 처방", 
     "keywords": ["EHT", "BEMT", "PBSA"], "difficulty": "hard", "doc_id": "lancome_uv_expert"},
    # Realistic
    {"query": "경쟁사 프리미엄 톤업 선블록 벤치마킹 자료", 
     "keywords": ["톤업", "안티에이징", "메이크업 베이스"], "difficulty": "realistic", "doc_id": "lancome_uv_expert"},
    {"query": "민감성 피부용 안티에이징 선케어 레퍼런스", 
     "keywords": ["민감", "안티에이징", "저자극"], "difficulty": "realistic", "doc_id": "lancome_uv_expert"},
    
    # ============================================================
    # 문서2: 글로벌 선케어 시장 동향
    # ============================================================
    # Easy
    {"query": "글로벌 선케어 시장 규모 2028", 
     "keywords": ["134", "억 달러", "17.5조"], "difficulty": "easy", "doc_id": "suncare_market"},
    {"query": "선케어 시장 연평균 성장률 CAGR", 
     "keywords": ["8.5%", "3.48%"], "difficulty": "easy", "doc_id": "suncare_market"},
    # Medium
    {"query": "선케어 시장 주요 경쟁사 현황", 
     "keywords": ["L'Oréal", "Beiersdorf", "Shiseido", "Johnson"], "difficulty": "medium", "doc_id": "suncare_market"},
    {"query": "액체 타입 선스크린 시장 트렌드", 
     "keywords": ["액체", "fluid", "흡수"], "difficulty": "medium", "doc_id": "suncare_market"},
    {"query": "미네랄 선스크린 무기자차 성장률", 
     "keywords": ["미네랄", "167%", "무기자차"], "difficulty": "medium", "doc_id": "suncare_market"},
    # Hard
    {"query": "글로벌 선케어 제형 트렌드 분석", 
     "keywords": ["액체", "fluid", "미네랄"], "difficulty": "hard", "doc_id": "suncare_market"},
    # Realistic
    {"query": "선케어 시장 진입 전략 수립용 경쟁사 데이터", 
     "keywords": ["L'Oréal", "Shiseido", "CAGR", "성장률"], "difficulty": "realistic", "doc_id": "suncare_market"},
    {"query": "2028년 선케어 시장 전망 리포트", 
     "keywords": ["2028", "134", "8.5%"], "difficulty": "realistic", "doc_id": "suncare_market"},
    
    # ============================================================
    # 문서3: UVMune 400 (MCE) 기술 문서
    # ============================================================
    # Easy
    {"query": "UVMune 400 MCE 필터 특성", 
     "keywords": ["UVMune", "MCE", "400"], "difficulty": "easy", "doc_id": "uvmune_tech"},
    {"query": "AAHCP 제형 기술 SPF 효율", 
     "keywords": ["AAHCP", "SPF", "40%"], "difficulty": "easy", "doc_id": "uvmune_tech"},
    # Medium
    {"query": "UVA1 스펙트럼 차단 기술 동향", 
     "keywords": ["UVA1", "400 nm", "스펙트럼"], "difficulty": "medium", "doc_id": "uvmune_tech"},
    {"query": "MCE 필터 흡수 파장 특성", 
     "keywords": ["MCE", "390 nm", "흡수"], "difficulty": "medium", "doc_id": "uvmune_tech"},
    # Hard
    {"query": "차세대 UV 필터 UVA1 보호 기술", 
     "keywords": ["UVA1", "MCE", "400 nm"], "difficulty": "hard", "doc_id": "uvmune_tech"},
    {"query": "선스크린 SPF 효율 향상 제형 기술", 
     "keywords": ["SPF", "AAHCP", "40%"], "difficulty": "hard", "doc_id": "uvmune_tech"},
    # Realistic
    {"query": "경쟁사 신규 UV 필터 기술 벤치마킹", 
     "keywords": ["UVMune", "MCE", "UVA1"], "difficulty": "realistic", "doc_id": "uvmune_tech"},
    {"query": "로레알 선케어 신기술 특허 분석", 
     "keywords": ["UVMune", "400 nm", "MCE"], "difficulty": "realistic", "doc_id": "uvmune_tech"},
    {"query": "피부 섬유아세포 보호 UV 차단 기술", 
     "keywords": ["섬유아세포", "UVA1", "피부"], "difficulty": "realistic", "doc_id": "uvmune_tech"},
]


# ================================================================
# 2. 평가 함수 (DataFrame 출력)
# ================================================================

def evaluate_retriever_detailed(retriever, eval_data, k=3):
    """
    상세 평가 실행 → DataFrame 반환
    """
    results = []
    
    for sample in eval_data:
        query = sample["query"]
        keywords = sample["keywords"]
        
        # 검색 실행
        docs = retriever.invoke(query)[:k]
        retrieved_text = " ".join([doc.page_content for doc in docs])
        
        # 키워드 매칭
        matched = [kw for kw in keywords if kw in retrieved_text]
        hit = len(matched) > 0
        
        # Reciprocal Rank 계산
        rr = 0.0
        for rank, doc in enumerate(docs, 1):
            if any(kw in doc.page_content for kw in keywords):
                rr = 1.0 / rank
                break
        
        # 결과 저장
        results.append({
            "query": query,
            "doc_id": sample["doc_id"],
            "difficulty": sample["difficulty"],
            "hit": hit,
            "matched_keywords": matched,
            "match_count": len(matched),
            "total_keywords": len(keywords),
            "reciprocal_rank": rr,
            "retrieved_count": len(docs),
        })
    
    return pd.DataFrame(results)


def generate_summary_report(df):
    """
    평가 결과 요약 리포트 생성
    """
    print("\n" + "=" * 70)
    print(f"📊 Retrieval 평가 리포트 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 70)
    
    # 1. 전체 요약
    overall_hr = df["hit"].mean()
    overall_mrr = df["reciprocal_rank"].mean()
    
    print(f"\n🎯 전체 성능")
    print(f"   Hit Rate:  {overall_hr:.1%} ({df['hit'].sum()}/{len(df)})")
    print(f"   MRR:       {overall_mrr:.3f}")
    
    # 2. 난이도별 요약
    print(f"\n📈 난이도별 성능")
    difficulty_summary = df.groupby("difficulty").agg({
        "hit": ["mean", "sum", "count"],
        "reciprocal_rank": "mean"
    }).round(3)
    difficulty_summary.columns = ["hit_rate", "hits", "total", "mrr"]
    
    for diff in ["easy", "medium", "hard", "realistic"]:
        if diff in difficulty_summary.index:
            row = difficulty_summary.loc[diff]
            print(f"   {diff:10s}: HR={row['hit_rate']:.1%} ({int(row['hits'])}/{int(row['total'])}), MRR={row['mrr']:.3f}")
    
    # 3. 문서별 요약
    print(f"\n📚 문서별 성능")
    doc_summary = df.groupby("doc_id").agg({
        "hit": ["mean", "sum", "count"],
        "reciprocal_rank": "mean"
    }).round(3)
    doc_summary.columns = ["hit_rate", "hits", "total", "mrr"]
    
    for doc_id in doc_summary.index:
        row = doc_summary.loc[doc_id]
        print(f"   {doc_id:20s}: HR={row['hit_rate']:.1%} ({int(row['hits'])}/{int(row['total'])}), MRR={row['mrr']:.3f}")
    
    # 4. 실패 케이스
    failed = df[df["hit"] == False]
    print(f"\n❌ 실패 케이스 ({len(failed)}건)")
    for _, row in failed.iterrows():
        print(f"   [{row['difficulty']:10s}] {row['query'][:50]}")
    
    # 5. 요약 DataFrame 생성
    summary_df = pd.DataFrame({
        "metric": ["Overall Hit Rate", "Overall MRR", "Easy HR", "Medium HR", "Hard HR", "Realistic HR"],
        "value": [
            overall_hr,
            overall_mrr,
            difficulty_summary.loc["easy", "hit_rate"] if "easy" in difficulty_summary.index else None,
            difficulty_summary.loc["medium", "hit_rate"] if "medium" in difficulty_summary.index else None,
            difficulty_summary.loc["hard", "hit_rate"] if "hard" in difficulty_summary.index else None,
            difficulty_summary.loc["realistic", "hit_rate"] if "realistic" in difficulty_summary.index else None,
        ]
    })
    
    return summary_df, difficulty_summary, doc_summary


# ================================================================
# 3. 실행
# ================================================================

if __name__ == "__main__":
    
    print("🔍 평가 시작...\n")
    
    # 상세 결과 DataFrame
    results_df = evaluate_retriever_detailed(retriever, eval_dataset, k=3)
    
    # 요약 리포트 생성
    summary_df, diff_summary, doc_summary = generate_summary_report(results_df)
    
    # DataFrame 저장 (선택)
    # results_df.to_csv("retrieval_eval_detailed.csv", index=False, encoding="utf-8-sig")
    # summary_df.to_csv("retrieval_eval_summary.csv", index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 70)
    print("📋 상세 결과 DataFrame (results_df)")
    print("=" * 70)
    print(results_df[["query", "difficulty", "hit", "match_count", "reciprocal_rank"]].to_string())


#%% 06. 핵심 요약 DataFrame (단일)

def generate_summary_df(results_df):
    """
    핵심 지표를 하나의 DataFrame으로 정리
    """
    rows = []
    
    # 1. 전체
    rows.append({
        "category": "total",
        "group": "overall",
        "queries": len(results_df),
        "hits": results_df["hit"].sum(),
        "hit_rate": round(results_df["hit"].mean(), 3),
        "mrr": round(results_df["reciprocal_rank"].mean(), 3),
    })
    
    # 2. 난이도별
    for diff in ["easy", "medium", "hard", "realistic"]:
        subset = results_df[results_df["difficulty"] == diff]
        if len(subset) > 0:
            rows.append({
                "category": "difficulty",
                "group": diff,
                "queries": len(subset),
                "hits": subset["hit"].sum(),
                "hit_rate": round(subset["hit"].mean(), 3),
                "mrr": round(subset["reciprocal_rank"].mean(), 3),
            })
    
    # 3. 문서별
    for doc_id in results_df["doc_id"].unique():
        subset = results_df[results_df["doc_id"] == doc_id]
        rows.append({
            "category": "document",
            "group": doc_id,
            "queries": len(subset),
            "hits": subset["hit"].sum(),
            "hit_rate": round(subset["hit"].mean(), 3),
            "mrr": round(subset["reciprocal_rank"].mean(), 3),
        })
    
    return pd.DataFrame(rows)


# 실행
if __name__ == "__main__":
    results_df = evaluate_retriever_detailed(retriever, eval_dataset, k=3)
    summary_df = generate_summary_df(results_df)
    print(summary_df.to_string(index=False))


#%% 07. 검색 전략 비교 평가 (확장 버전)

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List
import numpy as np
import pandas as pd
from itertools import product

# ================================================================
# 1. Sparse Retriever (BM25)
# ================================================================

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)
    
    def invoke(self, query: str, k: int = 3) -> List:
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_idx]


# ================================================================
# 2. Hybrid Retriever (RRF 기반)
# ================================================================

class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha=0.5):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha
    
    def invoke(self, query: str, k: int = 3) -> List:
        dense_docs = self.dense.invoke(query)[:k*2]
        sparse_docs = self.sparse.invoke(query, k*2)
        
        # RRF (Reciprocal Rank Fusion)
        scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(dense_docs):
            doc_id = hash(doc.page_content[:100])
            scores[doc_id] = scores.get(doc_id, 0) + self.alpha / (rank + 1)
            doc_map[doc_id] = doc
        
        for rank, doc in enumerate(sparse_docs):
            doc_id = hash(doc.page_content[:100])
            scores[doc_id] = scores.get(doc_id, 0) + (1 - self.alpha) / (rank + 1)
            doc_map[doc_id] = doc
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids[:k]]


# ================================================================
# 3. Reranker Retriever (다양한 모델 지원)
# ================================================================

class RerankerRetriever:
    def __init__(self, base_retriever, rerank_model: str):
        self.base = base_retriever
        self.model_name = rerank_model
        self.reranker = CrossEncoder(rerank_model)
    
    def invoke(self, query: str, k: int = 3) -> List:
        # 초기 검색 (3배수)
        try:
            candidates = self.base.invoke(query)[:k*3]
        except:
            candidates = self.base.invoke(query, k*3)
        
        if not candidates:
            return []
        
        # Cross-Encoder 재순위화
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]


# ================================================================
# 4. Reranker 모델 목록
# ================================================================

RERANK_MODELS = {
    "MiniLM-L6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "MiniLM-L12": "cross-encoder/ms-marco-MiniLM-L-12-v2", 
    "BGE-reranker": "BAAI/bge-reranker-base",
    "mmarco-mMiniLM": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # 다국어
}


# ================================================================
# 5. 전략 생성기
# ================================================================

def create_all_retrievers(dense_retriever, documents):
    """모든 검색 전략 조합 생성"""
    
    retrievers = {}
    
    # 1) Baseline: Dense, Sparse
    sparse_retriever = BM25Retriever(documents)
    retrievers["Dense"] = dense_retriever
    retrievers["Sparse (BM25)"] = sparse_retriever
    
    # 2) Hybrid: alpha 변화
    alphas = [0.3, 0.5, 0.7]
    hybrid_retrievers = {}
    
    for alpha in alphas:
        name = f"Hybrid (α={alpha})"
        hybrid = HybridRetriever(dense_retriever, sparse_retriever, alpha=alpha)
        retrievers[name] = hybrid
        hybrid_retrievers[alpha] = hybrid
    
    # 3) Reranker: 다양한 모델 × Hybrid alpha 조합
    for model_name, model_path in RERANK_MODELS.items():
        try:
            print(f"🔄 Loading reranker: {model_name}...")
            
            # Dense + Rerank
            retrievers[f"Dense + {model_name}"] = RerankerRetriever(
                dense_retriever, model_path
            )
            
            # Hybrid(0.5) + Rerank
            retrievers[f"Hybrid(0.5) + {model_name}"] = RerankerRetriever(
                hybrid_retrievers[0.5], model_path
            )
            
        except Exception as e:
            print(f"⚠️ {model_name} 로딩 실패: {e}")
    
    return retrievers


# ================================================================
# 6. 평가 함수
# ================================================================

def evaluate_single(retriever, eval_data, k=3):
    """단일 retriever 평가"""
    hits, rrs = [], []
    
    for sample in eval_data:
        query = sample["query"]
        keywords = sample["keywords"]
        
        try:
            docs = retriever.invoke(query)[:k]
        except:
            docs = retriever.invoke(query, k)
        
        text = " ".join([d.page_content for d in docs])
        
        # Hit
        hit = any(kw in text for kw in keywords)
        hits.append(hit)
        
        # RR
        rr = 0.0
        for rank, doc in enumerate(docs, 1):
            if any(kw in doc.page_content for kw in keywords):
                rr = 1.0 / rank
                break
        rrs.append(rr)
    
    return {
        "hit_rate": round(np.mean(hits), 3),
        "mrr": round(np.mean(rrs), 3),
        "hits": sum(hits),
        "total": len(hits)
    }


def compare_all_retrievers(retrievers_dict, eval_data, k=3):
    """전체 retriever 비교"""
    results = []
    
    for name, retriever in retrievers_dict.items():
        print(f"📊 평가 중: {name}")
        metrics = evaluate_single(retriever, eval_data, k)
        metrics["retriever"] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[["retriever", "hit_rate", "mrr", "hits", "total"]]
    df = df.sort_values("mrr", ascending=False).reset_index(drop=True)
    
    return df


# ================================================================
# 7. 결과 분석 함수
# ================================================================

def analyze_results(df):
    """결과 분석 및 요약"""
    
    print("\n" + "=" * 70)
    print("📊 검색 전략 비교 결과 (MRR 기준 정렬)")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Best 전략
    best = df.iloc[0]
    print(f"\n🏆 Best: {best['retriever']}")
    print(f"   Hit Rate: {best['hit_rate']:.1%}, MRR: {best['mrr']:.3f}")
    
    # 카테고리별 분석
    print("\n📈 카테고리별 최고 성능")
    print("-" * 50)
    
    # Baseline
    baseline = df[df["retriever"].isin(["Dense", "Sparse (BM25)"])]
    if len(baseline) > 0:
        best_base = baseline.sort_values("mrr", ascending=False).iloc[0]
        print(f"Baseline: {best_base['retriever']} (MRR: {best_base['mrr']:.3f})")
    
    # Hybrid
    hybrid = df[df["retriever"].str.contains("Hybrid") & ~df["retriever"].str.contains("\+")]
    if len(hybrid) > 0:
        best_hybrid = hybrid.sort_values("mrr", ascending=False).iloc[0]
        print(f"Hybrid: {best_hybrid['retriever']} (MRR: {best_hybrid['mrr']:.3f})")
    
    # Rerank
    rerank = df[df["retriever"].str.contains("\+")]
    if len(rerank) > 0:
        best_rerank = rerank.sort_values("mrr", ascending=False).iloc[0]
        print(f"Rerank: {best_rerank['retriever']} (MRR: {best_rerank['mrr']:.3f})")
    
    return df


# ================================================================
# 8. 실행
# ================================================================

if __name__ == "__main__":
    
    print("🚀 검색 전략 비교 평가 시작\n")
    
    # 전체 문서 가져오기
    all_docs = db.get()
    from langchain.schema import Document
    documents = [Document(page_content=text) for text in all_docs["documents"]]
    print(f"📚 총 문서 수: {len(documents)}\n")
    
    # 모든 retriever 생성
    print("=" * 50)
    print("🔧 Retriever 구성 중...")
    print("=" * 50)
    retrievers = create_all_retrievers(retriever, documents)
    print(f"\n✅ 총 {len(retrievers)}개 전략 구성 완료\n")
    
    # 비교 평가
    print("=" * 50)
    print("🔍 평가 시작...")
    print("=" * 50)
    comparison_df = compare_all_retrievers(retrievers, eval_dataset, k=3)
    
    # 결과 분석
    final_df = analyze_results(comparison_df)
    
    # 저장
    # comparison_df.to_csv("retrieval_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    
    
    #%% 08. 최종 RAG 파이프라인 (최적 전략 적용)

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain.schema import Document
import numpy as np

# ================================================================
# 1. 최적 Retriever 구성 (Hybrid + mmarco-mMiniLM)
# ================================================================

class OptimizedRetriever:
    """Hybrid(α=0.5) + mmarco-mMiniLM Reranker"""
    
    def __init__(self, dense_retriever, documents, alpha=0.5):
        self.dense = dense_retriever
        self.alpha = alpha
        
        # Sparse (BM25)
        self.documents = documents
        self.tokenized = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)
        
        # Reranker (다국어)
        print("🔄 Loading reranker: mmarco-mMiniLM...")
        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        print("✅ Reranker 로딩 완료")
    
    def _sparse_search(self, query: str, k: int) -> list:
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_idx]
    
    def _hybrid_search(self, query: str, k: int) -> list:
        """RRF 기반 Hybrid Search"""
        dense_docs = self.dense.invoke(query)[:k*2]
        sparse_docs = self._sparse_search(query, k*2)
        
        scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(dense_docs):
            doc_id = hash(doc.page_content[:100])
            scores[doc_id] = scores.get(doc_id, 0) + self.alpha / (rank + 1)
            doc_map[doc_id] = doc
        
        for rank, doc in enumerate(sparse_docs):
            doc_id = hash(doc.page_content[:100])
            scores[doc_id] = scores.get(doc_id, 0) + (1 - self.alpha) / (rank + 1)
            doc_map[doc_id] = doc
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids[:k*3]]
    
    def invoke(self, query: str, k: int = 3) -> list:
        """Hybrid + Rerank"""
        # 1) Hybrid 검색
        candidates = self._hybrid_search(query, k)
        
        if not candidates:
            return []
        
        # 2) Reranking
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]


# ================================================================
# 2. 최종 RAG Chain 구성
# ================================================================

def build_final_rag_chain(optimized_retriever, llm_func):
    """최종 RAG 파이프라인"""
    
    def format_docs(docs):
        return "\n\n".join(f"--- 문서 {i+1} ---\n{doc.page_content}" for i, doc in enumerate(docs))
    
    def rag_invoke(query: str) -> dict:
        # 1) Retrieval
        docs = optimized_retriever.invoke(query, k=3)
        context = format_docs(docs)
        
        # 2) Prompt 구성
        prompt = f"""주어진 맥락(Context) 정보를 사용하여 다음 질문에 답변해 주세요.
맥락에서 답을 찾을 수 없다면, "제공된 정보만으로는 답변하기 어렵습니다."라고 답하세요.
답변은 한국어로 간결하게 작성해주세요.

[맥락]
{context}

[질문]
{query}
"""
        # 3) Generation
        answer = llm_func(prompt)
        
        return {
            "query": query,
            "context": context,
            "answer": answer,
            "source_docs": docs
        }
    
    return rag_invoke


# ================================================================
# 3. 초기화 및 실행
# ================================================================
if __name__ == "__main__":
    
    # 문서 로드
    all_docs = db.get()
    documents = [Document(page_content=text) for text in all_docs["documents"]]
    
    # 최적 Retriever 생성
    optimized_retriever = OptimizedRetriever(retriever, documents, alpha=0.5)
    
    # RAG Chain 구성
    rag_chain = build_final_rag_chain(
        optimized_retriever,
        lambda q: ask_openrouter(q, model=selected_model)
    )
    
    # 테스트
    result = rag_chain("AAHCP 기술의 SPF 효율 향상 효과는?")
    result = rag_chain("경쟁사 선케어 UV 필터 트렌드 알려줘")
    result = rag_chain("최근 선케어 시장의 주요 기술 트렌드는?")
    result = rag_chain("선케어 시장 주요 경쟁사는?")
    result = rag_chain("UVMune 400 MCE 필터의 핵심 특징은?")
    
    print("✅ 답변:", result["answer"])


eval_queries = [
    "선케어 시장 주요 경쟁사는?",
    "최근 선케어 시장의 주요 기술 트렌드는?",
    "경쟁사 선케어 UV 필터 트렌드 알려줘", 
    "UVMune 400 MCE 필터의 핵심 특징은?",
    "AAHCP 기술의 SPF 효율 향상 효과는?"
   ]




#%% 11. 다중 Generation 모델 × 다중 Judge 모델 비교 평가

import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict
from itertools import product
import time

# ================================================================
# 1. 모델 설정
# ================================================================

# Generation 모델 (RAG 답변 생성용)
GENERATION_MODELS = {
    "Qwen: Qwen3 VL 30B A3B Thinking": "qwen/qwen3-vl-30b-a3b-thinking",
    "OpenAI: gpt-oss-20b": "openai/gpt-oss-20b",
    "Microsoft: Phi 4 Reasoning Plus": "microsoft/phi-4-reasoning-plus",
    "NVIDIA: Llama 3.3 Nemotron Super 49B V1.5" : "nvidia/llama-3.3-nemotron-super-49b-v1.5", 
    "QwQ-32B" : "qwen/qwq-32b"}

# Judge 모델 (평가용 - 고성능)
JUDGE_MODELS = {
    "Gemini-2.5-Pro": "google/gemini-2.5-pro-preview",
    "GPT-4.1": "openai/gpt-4.1",
    "Grok-4": "x-ai/grok-4",
}


# ================================================================
# 2. LLM 호출 함수
# ================================================================

def call_llm(prompt: str, model: str, temperature: float = 0.1, max_retries: int = 3) -> str:
    """LLM 호출 (재시도 포함)"""
    
    for attempt in range(max_retries):
        try:
            response = ask_openrouter(prompt, model=model, temperature=temperature)
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      ⚠️ 재시도 {attempt + 1}/{max_retries}: {e}")
                time.sleep(2)
            else:
                return f"Error: {e}"
    return "Error: Max retries exceeded"


# ================================================================
# 3. RAG Chain (모델 교체 가능)
# ================================================================

def create_rag_chain_with_model(retriever, gen_model: str):
    """특정 generation 모델로 RAG chain 생성"""
    
    def format_docs(docs):
        return "\n\n".join(f"--- 문서 {i+1} ---\n{doc.page_content}" for i, doc in enumerate(docs))
    
    def rag_invoke(query: str) -> dict:
        # Retrieval
        docs = retriever.invoke(query, k=3) if hasattr(retriever, 'invoke') else retriever.invoke(query)[:3]
        context = format_docs(docs)
        
        # Prompt
        prompt = f"""주어진 맥락(Context) 정보를 사용하여 다음 질문에 답변해 주세요.
맥락에서 답을 찾을 수 없다면, "제공된 정보만으로는 답변하기 어렵습니다."라고 답하세요.
답변은 한국어로 간결하게 작성해주세요.

[맥락]
{context}

[질문]
{query}
"""
        # Generation
        answer = call_llm(prompt, model=gen_model)
        
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }
    
    return rag_invoke


# ================================================================
# 4. RAGAS 평가 프롬프트
# ================================================================

RAGAS_PROMPT = """당신은 RAG 시스템의 답변 품질을 평가하는 전문가입니다.

[Question]
{question}

[Context]
{context}

[Answer]
{answer}

다음 4가지 기준으로 평가하세요 (각 0.0~1.0):

1. Faithfulness (충실성): Answer가 Context에만 기반하는가? (hallucination 없는가?)
2. Answer Relevancy (답변 관련성): Answer가 Question에 적절히 대응하는가?
3. Context Relevancy (맥락 관련성): Context가 Question에 유용한가?
4. Completeness (완전성): Context의 핵심 정보가 Answer에 포함되었는가?

반드시 아래 JSON 형식으로만 응답하세요:
{{"faithfulness": 점수, "answer_relevancy": 점수, "context_relevancy": 점수, "completeness": 점수}}
"""


def parse_ragas_scores(response: str) -> Dict:
    """RAGAS 점수 파싱"""
    try:
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            # 점수 정규화
            for key in ["faithfulness", "answer_relevancy", "context_relevancy", "completeness"]:
                scores[key] = min(1.0, max(0.0, float(scores.get(key, 0))))
            scores["ragas_score"] = round(np.mean([
                scores["faithfulness"], 
                scores["answer_relevancy"],
                scores["context_relevancy"],
                scores["completeness"]
            ]), 3)
            return scores
    except:
        pass
    return {
        "faithfulness": 0, "answer_relevancy": 0, 
        "context_relevancy": 0, "completeness": 0, "ragas_score": 0
    }


# ================================================================
# 5. 전체 평가 실행기
# ================================================================

class MultiModelEvaluator:
    """다중 Generation × 다중 Judge 평가"""
    
    def __init__(self, retriever, gen_models: Dict, judge_models: Dict):
        self.retriever = retriever
        self.gen_models = gen_models
        self.judge_models = judge_models
    
    def evaluate_single(self, query: str, gen_model_name: str, gen_model_id: str, 
                       judge_model_name: str, judge_model_id: str) -> Dict:
        """단일 (query, gen_model, judge_model) 조합 평가"""
        
        # 1) RAG 실행
        rag_chain = create_rag_chain_with_model(self.retriever, gen_model_id)
        rag_result = rag_chain(query)
        
        # 2) Judge 평가
        eval_prompt = RAGAS_PROMPT.format(
            question=query,
            context=rag_result["context"],
            answer=rag_result["answer"]
        )
        judge_response = call_llm(eval_prompt, judge_model_id, temperature=0.0)
        scores = parse_ragas_scores(judge_response)
        
        return {
            "query": query,
            "gen_model": gen_model_name,
            "judge_model": judge_model_name,
            "answer": rag_result["answer"][:100] + "...",
            **scores
        }
    
    def run_full_evaluation(self, queries: List[str], 
                           selected_gen_models: List[str] = None,
                           selected_judge_models: List[str] = None) -> pd.DataFrame:
        """전체 평가 실행"""
        
        # 모델 선택
        gen_models = {k: v for k, v in self.gen_models.items() 
                      if selected_gen_models is None or k in selected_gen_models}
        judge_models = {k: v for k, v in self.judge_models.items() 
                        if selected_judge_models is None or k in selected_judge_models}
        
        total = len(queries) * len(gen_models) * len(judge_models)
        print(f"🚀 평가 시작: {len(queries)} queries × {len(gen_models)} gen × {len(judge_models)} judge = {total} 조합\n")
        
        results = []
        count = 0
        
        for query in queries:
            print(f"\n📝 Query: {query[:40]}...")
            
            for gen_name, gen_id in gen_models.items():
                print(f"   🤖 Gen: {gen_name}")
                
                for judge_name, judge_id in judge_models.items():
                    count += 1
                    print(f"      [{count}/{total}] Judge: {judge_name}...", end=" ")
                    
                    try:
                        result = self.evaluate_single(
                            query, gen_name, gen_id, judge_name, judge_id
                        )
                        print(f"✅ RAGAS: {result['ragas_score']:.3f}")
                        results.append(result)
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        results.append({
                            "query": query,
                            "gen_model": gen_name,
                            "judge_model": judge_name,
                            "answer": f"Error: {e}",
                            "faithfulness": 0, "answer_relevancy": 0,
                            "context_relevancy": 0, "completeness": 0, "ragas_score": 0
                        })
                    
                    time.sleep(0.5)  # Rate limit 방지
        
        return pd.DataFrame(results)


# ================================================================
# 6. 결과 분석 함수
# ================================================================

def analyze_multi_model_results(df: pd.DataFrame) -> Dict:
    """다중 모델 평가 결과 분석"""
    
    metrics = ["faithfulness", "answer_relevancy", "context_relevancy", "completeness", "ragas_score"]
    
    print("\n" + "=" * 80)
    print("📊 다중 모델 RAGAS 평가 결과")
    print("=" * 80)
    
    # 1. Generation 모델별 평균 (전체 Judge 평균)
    print("\n🤖 Generation 모델별 성능 (Judge 평균)")
    print("-" * 80)
    
    gen_summary = df.groupby("gen_model")[metrics].mean().round(3)
    gen_summary = gen_summary.sort_values("ragas_score", ascending=False)
    print(gen_summary.to_string())
    
    # Best Generation Model
    best_gen = gen_summary.index[0]
    print(f"\n   🏆 Best Generation: {best_gen} (RAGAS: {gen_summary.loc[best_gen, 'ragas_score']:.3f})")
    
    # 2. Judge 모델별 평균 점수 (평가 경향)
    print("\n\n⚖️ Judge 모델별 평가 경향")
    print("-" * 80)
    
    judge_summary = df.groupby("judge_model")[metrics].mean().round(3)
    print(judge_summary.to_string())
    
    # 3. Generation × Judge 교차 분석
    print("\n\n🔀 Generation × Judge 교차표 (RAGAS Score)")
    print("-" * 80)
    
    cross_table = df.pivot_table(
        index="gen_model", 
        columns="judge_model", 
        values="ragas_score", 
        aggfunc="mean"
    ).round(3)
    cross_table = cross_table.sort_values(cross_table.columns[0], ascending=False)
    print(cross_table.to_string())
    
    # 4. Judge 간 일치도
    print("\n\n🔍 Judge 간 일치도 분석")
    print("-" * 80)
    
    judge_agreement = df.groupby(["query", "gen_model"])["ragas_score"].std().mean()
    print(f"   평균 표준편차: {judge_agreement:.3f}")
    
    if judge_agreement < 0.1:
        print("   → Judge 간 높은 일치도 ✅")
    elif judge_agreement < 0.15:
        print("   → Judge 간 양호한 일치도")
    else:
        print("   → Judge 간 낮은 일치도 ⚠️ (추가 검토 필요)")
    
    # 5. 상위/하위 Generation 모델 상세
    print("\n\n📈 Generation 모델 순위")
    print("-" * 80)
    
    for rank, (model, row) in enumerate(gen_summary.iterrows(), 1):
        bar = "█" * int(row["ragas_score"] * 20) + "░" * (20 - int(row["ragas_score"] * 20))
        print(f"   {rank:2d}. {model:25s} {bar} {row['ragas_score']:.3f}")
    
    return {
        "gen_summary": gen_summary,
        "judge_summary": judge_summary,
        "cross_table": cross_table,
        "judge_agreement": judge_agreement,
        "best_gen_model": best_gen
    }


# ================================================================
# 7. 평가 쿼리
# ================================================================


eval_queries = [
    "선케어 시장 주요 경쟁사는?",
    "최근 선케어 시장의 주요 기술 트렌드는?",
    "경쟁사 선케어 UV 필터 트렌드 알려줘", 
    "UVMune 400 MCE 필터의 핵심 특징은?",
    "AAHCP 기술의 SPF 효율 향상 효과는?"]


# ================================================================
# 8. 실행
# ================================================================

if __name__ == "__main__":
    
    # 평가기 생성
    evaluator = MultiModelEvaluator(
        retriever=optimized_retriever,  # 앞서 구성한 최적 retriever
        gen_models=GENERATION_MODELS,
        judge_models=JUDGE_MODELS
    )
    
    # 전체 평가 (또는 일부 모델만 선택)
    results_df = evaluator.run_full_evaluation(
        queries=eval_queries,
        # 일부만 테스트할 경우:
        # selected_gen_models=["Qwen2.5-7B-Instruct", "Llama-3.1-8B", "Gemma-3-4B"],
        # selected_judge_models=["Gemini-2.5-Pro", "Claude-Opus-4.5"],
    )
    
    # 결과 분석
    analysis = analyze_multi_model_results(results_df)
    
    # 저장
    # results_df.to_csv("multi_model_ragas_evaluation.csv", index=False, encoding="utf-8-sig")
    
    print("\n✅ 평가 완료")
    
    
    #%%
    
    