import streamlit as st

st.set_page_config(page_title="AI Tutor Bot", layout="wide")

st.title("📚 AI Tutor Bot")

pages = {
    "Tutor Workflow": [
        st.Page("upload_pdf.py", title="📄 Upload PDF"),
        st.Page("ask_question.py", title="💬 Ask Question"),
        st.Page("generate_quiz.py", title="📝 Generate Quiz"),
    ]
}

pg = st.navigation(pages)
pg.run()
