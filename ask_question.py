# ask_question.py
import streamlit as st
from streamlit_chat import message
from rag_utils import load_vector_store, query_gemini_with_rag

st.title("💬 Ask a Question")

vectordb = load_vector_store()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if vectordb:
    user_input = st.chat_input("Ask your question")
    explain = st.toggle("Explain like I'm 5")

    if user_input:
        response = query_gemini_with_rag(vectordb, user_input, explain)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

    for i, (role, msg) in enumerate(st.session_state.chat_history):
        message(msg, is_user=(role == "user"), key=f"{role}_{i}")
else:
    st.warning("⚠️ Please upload a PDF first.")
