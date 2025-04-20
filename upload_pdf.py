# upload_pdf.py
import streamlit as st
from rag_utils import load_and_embed_docs
import os, time

st.title("📄 Upload PDF and Create Embeddings")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Creating embeddings..."):
        vectordb, total_chunks = load_and_embed_docs(file_path)
        for i in range(1, total_chunks + 1):
            st.progress(i / total_chunks)
            time.sleep(0.01)

    st.success(f"✅ Created embeddings for {total_chunks} chunks.")
    st.balloons()
