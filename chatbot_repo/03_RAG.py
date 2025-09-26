# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:25:02 2025

@author: tmlab
"""

#%% 01. 임베딩 및 DB 로드
# -*- coding: utf-8 -*-
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
model_name = "nlpai-lab/KURE-v1"
model_kwargs = {'device': 'cpu'} # GPU가 없다면 'cpu'로 변경
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# ChromaDB 로드 및 Retriever 설정

chroma_persist_dir = r"C:\Users\PC1\OneDrive\프로젝트\250801_아모레\chroma_db"
db = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)
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

#%% 04.RAG 파이프라인 실행 (출력 방식 변경)

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