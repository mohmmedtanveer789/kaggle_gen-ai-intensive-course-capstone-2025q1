import streamlit as st
import os, time
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

genai.configure(api_key=os.getenv("AIzaSyCivVWkICUmGBcz6ScFTH5AgTUtKHX0sSA"))


# rag_utils.py
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

genai.configure(api_key=os.getenv("AIzaSyCivVWkICUmGBcz6ScFTH5AgTUtKHX0sSA"))

def load_and_embed_docs(pdf_path, save_path="vector_store"):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_documents(docs, embedding=embeddings)

    vectordb.save_local(save_path)
    return vectordb, len(docs)

def load_vector_store(path="vector_store"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCivVWkICUmGBcz6ScFTH5AgTUtKHX0sSA"  # Add this line
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def query_gemini_with_rag(vectordb, user_query, explain_eli5=False):
    relevant_docs = vectordb.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    style = "Explain like I’m 5." if explain_eli5 else "Answer clearly like a tutor."

    prompt = f"""You are a helpful tutor. {style}
    Use the following context to answer:\n{context}\n\nQuestion: {user_query}"""

    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def generate_quiz_questions(topic):
    prompt = f"""
You are an AI tutor. Create a multiple choice quiz with exactly 5 questions on the topic: "{topic}".
Each question must have four options labeled A), B), C), D).
At the end of each question, provide the correct answer on a separate line like: "Answer: B".

Format strictly like:
1. Question text?
A) Option A
B) Option B
C) Option C
D) Option D
Answer: B

Repeat for 5 questions only. Do not add explanations or extra text.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def evaluate_quiz(questions, user_answers):
    correct = 0
    total = len(questions)
    feedback = []

    for i, (q, user_ans) in enumerate(zip(questions, user_answers)):
        correct_ans = q["answer"].strip()
        is_correct = user_ans.strip().startswith(correct_ans)

        if is_correct:
            correct += 1
            feedback.append(f"✅ Q{i+1}: Correct! ({user_ans})")
        else:
            feedback.append(
                f"❌ Q{i+1}: Incorrect. Your answer: {user_ans} | Correct: {correct_ans}"
            )

    score_percent = round((correct / total) * 100, 2)
    summary = f"\n**Your Score: {correct}/{total} ({score_percent}%)**\n"
    return "\n".join(feedback) + "\n\n" + summary




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
