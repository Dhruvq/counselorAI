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

# Initialize Chat Engine in Session State (with Preloading)
if "chat_engine" not in st.session_state:
    # Create a status container to show progress
    with st.status("Initializing System...", expanded=True) as status:
        try:
            # Step 1: Load the Engine (Vector DB + Index)
            st.write("Loading Knowledge Base...")
            chat_engine = get_chat_engine()
            
            # Step 2: Warm Up / Preload
            # We send a dummy message to force the heavy LLM to load into RAM now
            # instead of waiting for the user's first input.
            st.write("Warming up AI Model (this prevents timeouts)...")
            chat_engine.chat("Just say hello.") 
            
            # Step 3: Save to session state
            st.session_state.chat_engine = chat_engine
            
            # Update status to finished
            status.update(label="System Ready!", state="complete", expanded=False)
            
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