# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:25:02 2025

@author: tmlab
"""

import os
import pickle


# 'rb'는 바이너리 읽기 모드를 의미합니다.
with open(r"C:\Users\PC1\OneDrive\기술인텔리전스\프로젝트\수행중\아모레자문_박상현\df_sample_pages.pkl", "rb") as f:
    
    df_sample_pages = pickle.load(f)
    
    

#%%

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

print("\n추출된 내용을 LangChain Document로 래핑합니다...")
langchain_docs = []

for idx, row in df_sample_pages.iterrows():
    text = row['VQA_result_google/gemma-3-27b-it']
    if type(text) == float : continue
    
    # LangChain Document 객체 생성
    doc_instance = Document(
        page_content=text,
        metadata={
           "level1": row['level1'], 
           "level2": row['level2'], 
           "source": row['source'], 
          
            "page_number": row['page_number'], 
        }
    )
    langchain_docs.append(doc_instance)
    
print(f"총 {len(langchain_docs)}개의 LangChain Document를 생성했습니다.")

print("첫 번째 Document 예시:", langchain_docs[0]) # 생성된 Document 확인용

#%% 텍스트 분할, 임베딩 및 Vector Store 저장 (이전과 동일) ---

# 1. 텍스트 분할
print("\nDocument를 청크로 분할합니다...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(langchain_docs)
print(f"총 {len(split_documents)}개의 청크로 분할되었습니다.")

# 2. 임베딩
print("임베딩 모델을 로드합니다 (GPU 사용)...")
model_name = "nlpai-lab/KURE-v1"

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

#%% 벡터 저장 후 테스트
from langchain_community.vectorstores import Chroma # <--- Chroma로 변경

chroma_persist_dir = r"C:\Users\PC1\OneDrive\250801_아모레\chroma_db"

# 3. 벡터 저장 (Chroma)
print("벡터 저장을 시작합니다...")
# <--- Chroma.from_documents로 변경, persist_directory 지정 ---
vectorstore = Chroma.from_documents(
    documents=split_documents, 
    embedding=embeddings,
    # collection_name="amore_bge_m3_v1",               # ← 새 이름
    persist_directory=chroma_persist_dir
)
print(f"\n벡터 스토어를 '{chroma_persist_dir}' 폴더에 성공적으로 저장했습니다.")

# --- 4단계: 저장된 데이터로 검색 테스트 ---

# 1. 저장된 DB 다시 불러오기 (테스트를 위해)
db = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)

# 2. 검색할 질문(쿼리) 설정
# query = "올리브영의 사이트 클릭수는 얼마이고, 전년 대비 증가율은 어느 정도인가요?"
query = '코로나19 이후 소비자 트렌드에 대해 알려줘'
print(f"질문: {query}\n")

# 3. 유사도 높은 순서로 3개 문서 검색
# similarity_search_with_score는 유사도 점수(낮을수록 유사)도 함께 반환
searched_docs = db.similarity_search_with_score(query, k=3)

# 4. 검색 결과 출력
print("--- 검색 결과 ---")
if not searched_docs:
    print("검색 결과가 없습니다.")
else:
    for i, (doc, score) in enumerate(searched_docs):
        print(f"[결과 {i+1}] (유사도 점수: {score:.4f})")
        print(f"  - 내용: {doc.page_content[:1000]}...") # 내용 일부 출력
        print(f"  - 출처: {doc.metadata.get('source', 'N/A')}")
        print(f"  - 페이지: {doc.metadata.get('page_number', 'N/A')}")
        print(f"  - OCR 적용 여부: {doc.metadata.get('ocr_applied', 'N/A')}\n")