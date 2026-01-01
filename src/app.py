import streamlit as st
import os
from rag_engine import get_chat_engine

# Page Config
st.set_page_config(
    page_title="USC MSEE Counselor AI",
    page_icon="🎓",
    layout="centered"
)

# Header
st.title("🎓 USC MSEE Student Helper")
st.markdown("Ask questions about policies, courses, and degree requirements. Answers are based *only* on official documents.")

# Initialize Chat Engine in Session State (so it doesn't reload on every click)
if "chat_engine" not in st.session_state:
    try:
        st.session_state.chat_engine = get_chat_engine()
        st.success("System Ready: Knowledge Base Loaded.")
    except Exception as e:
        st.error(f"Error loading system: {e}")
        st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help you with USC MSEE requirements. What would you like to know?"}]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question (e.g., 'What are the core requirements?')"):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting the handbook..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.markdown(response.response)
            
    # 3. Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response.response})