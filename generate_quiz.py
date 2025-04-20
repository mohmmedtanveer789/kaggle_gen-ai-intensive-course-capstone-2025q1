# generate_quiz.py
import streamlit as st
from streamlit_chat import message
from rag_utils import generate_quiz_questions, evaluate_quiz

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
