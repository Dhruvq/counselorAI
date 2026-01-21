import os
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
import chromadb

# Configuration
CHROMA_DB_DIR = "./chroma_db"
DATA_DIR = "./data"
COLLECTION_NAME = "usc_msee_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3.2"  

def load_index():
    """
    Loads the pre-built vector index from disk.
    """

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
    
    # Auto-build index if empty (Robustness fix for fresh deployments)
    if chroma_collection.count() == 0:
        print("Vector DB is empty. Attempting to ingest data from ./data...")
        if os.path.exists(DATA_DIR):
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            if documents:
                return VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        
        # If we reach here, no data found
        raise ValueError("Vector DB is empty and no data found in ./data.")
        
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

def get_chat_engine():
    """
    Creates the chat engine with strict system prompts.
    """
    index = load_index()
    
    # Custom System Prompt to enforce "Implacable Retrieval"
    custom_prompt = (
        "You are an academic advisor for the USC MSEE program. "
        "Your goal is to answer questions strictly based on the provided context. "
        "Check all provided context snippets for relevant details before answering. "
        "If the user's question is vague, ambiguous, or could refer to multiple topics found in the context, "
        "ask a clarifying question to better understand their intent instead of guessing. "
        "If you find conflicting information in the context, mention both and ask for clarification. "
        "If the answer is not explicitly in the context, state: "
        "'I cannot find this information in the official documents. Please consult an academic advisor.' "
        "Do not make up policies or courses. "
        "Keep answers professional and concise."
    )
    
    # Re-ranker: Re-scores top 30 retrieved nodes to find the best 7
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=7
    )
    
    # --- Hybrid Search Setup ---
    # 1. Create Vector Retriever
    vector_retriever = index.as_retriever(similarity_top_k=30)
    
    # 2. Create BM25 (Keyword) Retriever
    # We fetch all documents from Chroma to build the keyword index in memory.
    # This ensures we can find exact matches like "EE 483" that vectors might miss.
    db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = db.get_collection(COLLECTION_NAME)
    data = collection.get() # Fetch all docs
    
    nodes = []
    if data and data['documents']:
        metadatas = data['metadatas'] if data['metadatas'] else [{}] * len(data['ids'])
        for id, text, meta in zip(data['ids'], data['documents'], metadatas):
            nodes.append(TextNode(id_=id, text=text, metadata=meta))
            
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=30)
    
    # 3. Fuse Retrievers (Hybrid)
    # Combines results from both Vector and Keyword search using Reciprocal Rank Fusion
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=30, # Total candidates to pass to the re-ranker
        num_queries=1,       # Use the original query (faster than generating variations)
        mode="reciprocal_rerank",
        use_async=False,
        verbose=True
    )

    return ContextChatEngine.from_defaults(
        retriever=fusion_retriever,
        system_prompt=custom_prompt,
        node_postprocessors=[reranker], # Filter them down to the best 7
        verbose=True
    )