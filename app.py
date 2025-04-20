import streamlit as st

st.set_page_config(page_title="Streamlit Check", layout="centered")

st.write("Hello Tanveer")

st.title("✅ Streamlit is Working!")
st.subheader("A quick test of Streamlit components")

name = st.text_input("What's your name?", "Streamlit User")
if st.button("Say Hello"):
    st.success(f"Hello, {name}! 👋")

st.markdown("---")

st.write("Here's a simple chart:")
st.line_chart({
    "data": [1, 5, 2, 6, 9, 4]
})

st.markdown("---")

if st.checkbox("Show secret message"):
    st.info("🎉 You discovered the hidden message!")

st.markdown("Try changing this slider value:")
value = st.slider("Pick a number", 0, 100, 42)
st.write(f"You picked: {value}")
