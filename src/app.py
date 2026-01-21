import streamlit as st
from llama_index.core import Settings
from rag_engine import get_chat_engine

# Page Config
st.set_page_config(
    page_title="USC Counselor AI",
    page_icon="🎓",
    layout="centered"
)

# Header
st.title("🎓 USC MS ECE Student Helper")
st.markdown("Ask questions about policies, courses, and degree requirements. Answers are based *only* on official documents.")

# Sidebar Controls (For Testing & Diagnostics)
with st.sidebar:
    st.header("Controls")
    if st.button("Reset Conversation"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help you with USC requirements. What would you like to know?"}]
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()
        st.rerun()

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
            Settings.llm.complete("Just say hello.")
            
            # Step 3: Save to session state
            st.session_state.chat_engine = chat_engine
            
            # Update status to finished
            status.update(label="System Ready!", state="complete", expanded=False)
            
        except Exception as e:
            st.error(f"Error loading system: {e}")
            st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help you with USC requirements. What would you like to know?"}]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['file']}** (Page {source['page']}) - *Similarity: {source['score']}*")
                    st.caption(source['preview'])

# User Input
if prompt := st.chat_input("Ask a question (e.g., 'How many units do I need to graduate as a MSEE student?')"):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting the handbook..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.markdown(response.response)
            
            # Process and display sources
            sources_data = []
            if response.source_nodes:
                with st.expander("View Sources"):
                    for node in response.source_nodes:
                        file_name = node.metadata.get("file_name", "Unknown Document")
                        page_label = node.metadata.get("page_label", "N/A")
                        score = f"{node.score:.2f}" if node.score is not None else "N/A"
                        preview = node.get_content()[:200] + "..."
                        
                        st.markdown(f"**{file_name}** (Page {page_label}) - *Similarity: {score}*")
                        st.caption(preview)
                        
                        sources_data.append({"file": file_name, "page": page_label, "score": score, "preview": preview})
            
    # 3. Add assistant message to history
    message_data = {"role": "assistant", "content": response.response}
    if sources_data:
        message_data["sources"] = sources_data
    st.session_state.messages.append(message_data)