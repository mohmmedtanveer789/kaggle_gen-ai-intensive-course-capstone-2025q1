import os
import streamlit as st
from streamlit_chat import message
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


st.title("📝 Generate Quiz")

# Step 1: Initialize session state
if "quiz" not in st.session_state:
    st.session_state.quiz = {
        "started": False,
        "questions": [],
        "answers": [],
        "current_q": 0,
    }

quiz = st.session_state.quiz

# Step 2: Generate quiz on topic input
if not quiz["started"]:
    topic = st.text_input("Enter a topic to generate quiz")

    if st.button("Start Quiz") and topic:
        raw = generate_quiz_questions(topic)
        lines = raw.strip().split("\n")
        parsed = []
        q = {}

        for line in lines:
            if line.strip().startswith(("1.", "2.", "3.", "4.", "5.")):
                if q: parsed.append(q)
                q = {"question": line.strip(), "options": [], "answer": ""}
            elif any(opt in line for opt in ["A)", "B)", "C)", "D)"]):
                q["options"].append(line.strip())
            elif "Answer:" in line:
                q["answer"] = line.split("Answer:")[-1].strip()
        if q: parsed.append(q)

        quiz["questions"] = parsed
        quiz["started"] = True
        quiz["answers"] = []
        quiz["current_q"] = 0
        st.rerun()

# Step 3: Show quiz questions
if quiz["started"] and quiz["current_q"] < len(quiz["questions"]):
    q = quiz["questions"][quiz["current_q"]]
    message(f"**Q{quiz['current_q']+1}: {q['question']}**", is_user=False)

    selected_option = st.radio(
        "Choose an answer:",
        options=q["options"],
        key=f"q_{quiz['current_q']}_option"
    )

    if st.button("Next"):
        quiz["answers"].append(selected_option)
        quiz["current_q"] += 1
        st.rerun()

# Step 4: Show report
if quiz["started"] and quiz["current_q"] >= len(quiz["questions"]):
    message("🎉 Quiz Completed!", is_user=False)
    report = evaluate_quiz(quiz["questions"], quiz["answers"])
    message(report, is_user=False)

    if st.button("Restart Quiz"):
        del st.session_state.quiz
        st.rerun()
