# CounselorAI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-orange)](https://www.llamaindex.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)](https://ollama.com/)

## Roadmap

* [x] **Initial proof of concept:** Creating a Local Retrieval-Augmented Generation (RAG) agent pipeline where the flow of **Ingestion**, **Storage:**, **Retrieval:**  and **Synthesis:** is followed and displayed on a basic UI. Accuracy in this stage in **NOT** a concern only that the pipeline behaves as expected.
* [ ] **Supply Refined Data:** For the best results we must provide as much relevant data in the initial stage of the RAG pipeline, this will improve usability and relevance to users significantly more than any post data accuracy improvements.
* [ ] **Dockerization:** Fully containerize the Streamlit app and ChromaDB for one-command deployment.
* [ ] **Accuracy:** Setting a low temperature(less creative, more encyclopedic knowledge), prompt engineering and other strategies can be implemented to improve accuracy within the bounds of the fairly simple llama3.2(~3B) model.
* [ ] **Hybrid Search:** Implement keyword search alongside vector search for better acronym recognition (e.g., "EE 483").
* [ ] **Citations:** Update UI to display the specific page number of the source PDF used for the answer.  

**CounselorAI** is a local, privacy-focused Retrieval-Augmented Generation (RAG) agent designed to assist USC Electrical Engineering (MSEE) students. It answers questions regarding degree requirements, course planning, and academic policies by strictly grounding responses in official university documentation (handbooks, catalogues).

> **Note:** This project runs 100% locally. No data is sent to external APIs (OpenAI/Anthropic), ensuring student privacy and zero inference costs.

---

## Key Features

* **Implacable Retrieval:** The agent uses a strict system prompt to refuse answering if the information is not found in the official documents, minimizing hallucinations.
* **Edge-Optimized:** Tuned to run on consumer hardware (e.g., Apple Silicon M1/M2) using quantization and the **Llama 3.2 (3B)** model.
* **Vector Search:** Utilizes **ChromaDB** for persistent, semantic search over PDF embeddings.
* **Interactive UI:** A clean, chat-based interface built with **Streamlit**.

## Technical Architecture

The system follows a standard RAG pipeline architecture:

1.  **Ingestion:** PDF documents are loaded, chunked, and embedded using `BAAI/bge-small-en`.
2.  **Storage:** Embeddings are stored locally in `ChromaDB`.
3.  **Retrieval:** User queries are matched against the vector store to retrieve the top-k relevant context nodes.
4.  **Synthesis:** The retrieved context + user query are sent to a local LLM (**Llama 3.2**) via **Ollama** to generate the final response.

## Project Structure

```text
counselorAI/
├── data/                    # (Ignored by Git) Place USC PDF handbooks here
├── src/
│   ├── app.py               # Streamlit Frontend UI
│   ├── ingestion.py         # ETL Pipeline: Reads PDFs -> Updates Vector DB
│   └── rag_engine.py        # Core Logic: Initialization of LlamaIndex & Ollama
├── requirements.txt         # Python dependencies
├── docker-compose.yaml      # Container orchestration (In Progress)
└── README.md                # Documentation
```

## Quick Start

### Prerequisites
* **Python 3.10+**
* **[Ollama](https://ollama.com/)** (Required for local inference)

### 1. Installation
Clone the repository and set up the environment:

```bash
git clone [https://github.com/Dhruvq/counselorAI.git](https://github.com/Dhruvq/counselorAI.git)
cd counselorAI

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure the Brain
Start Ollama and pull the optimized model (Llama 3.2 3B):

```bash
ollama pull llama3.2
```

### 3. Ingest Data
Place your official USC PDF documents (e.g., *Viterbi Graduate Handbook.pdf*) into the `data/` folder. Then run the ingestion pipeline:

```bash
python src/ingestion.py
```
*Output: `Success! Data ingestion complete. Vector DB saved to ./chroma_db`*

### 4. Run the Application

```bash
streamlit run src/app.py
```
