import os
import logging
import sys
import chromadb

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Configure logging to look professional
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "./data"
CHROMA_DB_DIR = "./chroma_db"  # This folder will be created automatically
COLLECTION_NAME = "usc_msee_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # High performance, small size, runs locally

def ingest_data():
    # 1. Validation
    if not os.path.exists(DATA_DIR):
        logger.error(f"Directory '{DATA_DIR}' does not exist. Please create it and add PDFs.")
        return
    
    # Check if directory is empty
    if not os.listdir(DATA_DIR):
        logger.warning(f"Directory '{DATA_DIR}' is empty. No data to ingest.")
        return

    # 2. Load Data (PDFs)
    logger.info("Loading documents from data directory...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    logger.info(f"Loaded {len(documents)} document pages.")

    # 3. Setup Embedding Model
    # We use a local HuggingFace model so we don't need OpenAI API keys
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    Settings.llm = None  # We don't need an LLM for ingestion, only embeddings

    # 4. Setup Vector Database (ChromaDB)
    logger.info("Setting up ChromaDB client...")
    db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Create the Index
    logger.info("Creating vector index... (This may take a few minutes)")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    logger.info("Success! Data ingestion complete. Vector DB saved to ./chroma_db")

if __name__ == "__main__":
    ingest_data()