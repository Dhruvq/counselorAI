# CounselorAI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-orange)](https://www.llamaindex.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)](https://ollama.com/)
 

**CounselorAI** is a local, privacy-focused Retrieval-Augmented Generation (RAG) agent designed to assist USC's Electrical and Computer Engineering (MS ECE) students. It answers questions regarding degree requirements, course planning, and academic policies by strictly grounding responses in official university documentation.

> **Note:** At the current stage this project runs 100% locally. No data is sent to external APIs (OpenAI/Anthropic), ensuring student privacy and zero inference costs.

---

## Key Features

* **Implacable Retrieval:** The agent uses a strict system prompt to refuse answering if the information is not found in the official documents, minimizing hallucinations.
* **Edge-Optimized:** Tuned to run on consumer hardware (e.g., Apple Silicon M1/M2) using quantization and the **Llama 3.2 (3B)** model.
* **Vector Search:** Utilizes **ChromaDB** for persistent, semantic search over PDF embeddings.
* **Interactive UI:** A clean, chat-based interface built with **Streamlit**.

## Technical Architecture

The system follows a standard RAG pipeline architecture:

1.  **Ingestion:** PDF documents are loaded, chunked, and embedded using `BAAI/bge-small-en-v1.5`.
2.  **Storage:** Embeddings are stored locally in `ChromaDB`.
3.  **Retrieval:** User queries are matched against the vector store to retrieve the top-k relevant context nodes.
4.  **Synthesis:** The retrieved context + user query are sent to a local LLM (**Llama 3.2**) via **Ollama** to generate the final response.

## Project Structure

```text
counselorAI/
├── data/                    # USC PDF handbooks here
├── src/
│   ├── app.py               # Streamlit Frontend UI
│   ├── ingestion.py         # ETL Pipeline: Reads PDFs -> Updates Vector DB
│   └── rag_engine.py        # Core Logic: Initialization of LlamaIndex & Ollama
├── requirements.txt         # Python dependencies             
├── docker-compose.yaml      # Container file
├── Dockerfile  
└── README.md                # Documentation
```

## Quick Start

### Prerequisites
* **Python 3.12+**
* **[Ollama](https://ollama.com/)** (Required for local inference)

## Option 1: Docker Config (Recommended)
> **Note:** If you wish to add additional pdf files for the model to use as trusted sources of information, skip to **Option 2** to run the ingestion pipeline manually.
### 1. Setup Docker
* **Download:** Install [Docker Desktop](https://www.docker.com/) and sign in to your account.
* **Configure Resources:** To ensure the AI performs optimally, increase your memory allocation:
    1. Open Docker **Settings**.
    2. Navigate to **Resources** → **Advanced** → **Resource Allocation**.
    3. Set the **Memory Limit** to **8 GB**.

### 2. Launch the Application
Run the following commands in your terminal to clone the repository and build the container:
```bash 
git clone https://github.com/Dhruvq/counselorAI.git
cd counselorAI
docker-compose up --build
```
*Voila! You should be able to chat with your private, personal counselor!*

## Option 2: Virtual Environment Config (Legacy)

### 1. Installation
Clone the repository and set up the environment:

```bash
git clone https://github.com/Dhruvq/counselorAI.git
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
Add any additional relevant PDF documents(that you would trust as a source of information) into the `data/` folder(.pdf files only). Then run the ingestion pipeline:

```bash
python src/ingestion.py
```
*Output: `Success! Data ingestion complete. Vector DB saved to ./chroma_db`*

### 4. Run the Application

```bash
streamlit run src/app.py
```
## Preview:
<img width="735" height="362.5" alt="Screenshot 2026-01-22 at 10 02 03 PM" src="https://github.com/user-attachments/assets/14a54409-4af8-4b00-b24c-1ff19e14d501" />

## Roadmap to how the project was built:

* [x] **Initial proof of concept:** Created a Local Retrieval-Augmented Generation (RAG) agent pipeline where the flow of **Ingestion**, **Storage**, **Retrieval**  and **Synthesis** is followed and displayed on a basic UI. Accuracy in this stage was **NOT** a concern, only that the pipeline behaved as expected.

* [x] **Add a 'preload' step:** Added a dummy request to the model before letting the user make a request. This gave the model time to load into the computer's RAM, leading to fewer timeouts and a smoother user experience.

* [x] **Supply initial refined data:** Supplied a smaller batch of data to help test the system and ensure usability and relevance to users in a more confined manner before expanding to more comprehensive data.

* [x] **Dockerization:** Fully containerized the Streamlit app and ChromaDB for one-command deployment.

* [x] **Accuracy calibration:** Set a low temperature (less creative, more encyclopedic knowledge) and implemented prompt engineering strategies to improve accuracy within the bounds of the fairly simple llama3.2(~3B) model.

* [x] **Add more relevant Data:** Provided as much relevant data as possible in the initial stage of the RAG pipeline. This helped the system be as useful as possible to MS ECE students and increased the chances that a potential question could be answered without the user having to add sources of data.

* [x] **Improve accuracy on the larger data set:** Ensured that the larger dataset did not mess with the model's ability to retrieve relevant information. Furthermore, used advanced techniques (e.g., adding a Re-ranking step using a Cross-Encoder to rank relevant sources and only passing the top-k to the LLM) to improve the model's accuracy. This was done while keeping in mind that the system is designed to run on a local machine, which caps how compute-hungry the techniques can be.

* [x] **Hybrid Search:** Implemented keyword search alongside vector search for better acronym recognition (e.g., "EE 483"). For this, I combined the existing Vector Search (semantic understanding) with Keyword Search (BM25). This allowed the system to retrieve specific course codes (e.g., "EE 559") or acronyms that vector embeddings sometimes miss. These were the exact steps:
    * Fetched the text from ChromaDB at startup.
    * Built an in-memory BM25 index (Keyword Retriever).
    * Fused the Vector Retriever and BM25 Retriever using Reciprocal Rank Fusion (RRF).


* [x] **Citations:** Updated UI to display the specific page number of the source PDF used for the answer. 

* [x] **Optimization:** I optimized the retrieval and inference pipeline by reducing the similarity search top_k from 30 to 20 for better efficiency, updated the RAG engine to automatically recreate the Chroma database when it is empty, adding a startup check to detect GPU availability, and improving the warm-up stage so it only runs the LLM instead of the full top_k pipeline, reducing unnecessary overhead and startup compute time.

* [x] **Final end to end test:** Added a 'Reset Conversation' button for better testing. Reran the whole project top to bottom, checked for edge cases, got feedback from potential users and implement finishing touches in order to finalize the project.
