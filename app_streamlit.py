# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:38:19 2025

@author: PC1
"""


# app_streamlit.py
import streamlit as st
from typing import List, Tuple
from rag_core.core import RAGCore, ChunkConfig, EmbedConfig, RetrieveConfig

st.set_page_config(page_title="Search-only RAG", page_icon="🔎", layout="wide")
st.title("🔎 Search-only RAG (minimal)")

with st.sidebar:
    st.subheader("Settings")
    chunk_size = st.number_input("Chunk size (words)", 100, 2000, 800, 50)
    chunk_overlap = st.number_input("Chunk overlap (words)", 0, 800, 100, 10)
    top_k = st.number_input("Top-K", 1, 50, 5, 1)
    embed_model = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")

# 세션에 코어 인스턴스 유지
if "rag" not in st.session_state:
    st.session_state.rag = RAGCore(EmbedConfig(model_name=embed_model))

st.markdown("### 1) Upload documents (PDF/TXT)")
uploads = st.file_uploader("Upload files", type=["pdf", "txt"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Build / Rebuild Index", type="primary"):
        if not uploads:
            st.warning("Please upload at least one file.")
        else:
            files: List[Tuple[str, bytes]] = [(f.name, f.getvalue()) for f in uploads]
            # 임베딩 모델 변경 반영을 위해 새 인스턴스로 교체
            st.session_state.rag = RAGCore(EmbedConfig(model_name=embed_model))
            n = st.session_state.rag.build_index_from_files(
                files,
                ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            )
            st.success(f"Indexed {n} chunks from {len(uploads)} file(s).")

with col2:
    if st.button("Clear Index"):
        st.session_state.rag = RAGCore(EmbedConfig(model_name=embed_model))
        st.info("Cleared in-memory index.")

st.divider()
st.markdown("### 2) Ask")
q = st.text_input("Your query", placeholder="What does the document say about ...?")

if q:
    try:
        hits = st.session_state.rag.retrieve(q, RetrieveConfig(top_k=int(top_k)))
        st.markdown("#### Results")
        for i, h in enumerate(hits, 1):
            with st.expander(f"[{i}] score={h['score']:.4f} • {h['meta'].get('source','')}"):
                st.write(h["text"])
                st.caption(h["meta"])
    except Exception as e:
        st.error(str(e))