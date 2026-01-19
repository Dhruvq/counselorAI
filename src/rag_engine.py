import os
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# Configuration
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "usc_msee_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3.2"  

def load_index():
    """
    Loads the pre-built vector index from disk.
    """
    if not os.path.exists(CHROMA_DB_DIR):
        raise ValueError(f"Vector DB not found at {CHROMA_DB_DIR}. Run ingestion.py first.")

    # 1. Setup Embedding Model (Must match ingestion!)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.embed_model = embed_model

    # 2. Setup LLM (Ollama)
    # We use an env var for the URL so it works both locally and in Docker
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Auto-pull model if it doesn't exist
    try:
        tags_response = requests.get(f"{ollama_url}/api/tags")
        if tags_response.status_code == 200:
            existing_models = [m['name'] for m in tags_response.json().get('models', [])]
            if not any(LLM_MODEL in m for m in existing_models):
                print(f"Model '{LLM_MODEL}' not found. Pulling... (This may take a few minutes)")
                requests.post(f"{ollama_url}/api/pull", json={"name": LLM_MODEL, "stream": False})
                print(f"Model '{LLM_MODEL}' pulled successfully.")
    except Exception as e:
        print(f"Warning: Could not verify/pull Ollama model: {e}")

    Settings.llm = Ollama(
        model=LLM_MODEL, 
        base_url=ollama_url,
        request_timeout=600.0,
        temperature=0.1, # Low temperature = more factual/less creative
        context_window=8192,
        additional_kwargs={"num_ctx": 8192}
    )

    # 3. Connect to ChromaDB
    db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 4. Load Index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )
    return index

def get_chat_engine():
    """
    Creates the chat engine with strict system prompts.
    """
    index = load_index()
    
    # Custom System Prompt to enforce "Implacable Retrieval"
    custom_prompt = (
        "You are an academic advisor for the USC MSEE program. "
        "Your goal is to answer questions strictly based on the provided context. "
        "If the answer is not explicitly in the context, state: "
        "'I cannot find this information in the official documents. Please consult an academic advisor.' "
        "Do not make up policies or courses. "
        "Keep answers professional and concise."
    )
    
    return index.as_chat_engine(
        chat_mode="context", 
        system_prompt=custom_prompt,
        similarity_top_k=5, # Fetch more context since we have 8k window
        verbose=True
    )