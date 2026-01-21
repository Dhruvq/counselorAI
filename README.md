# CounselorAI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-orange)](https://www.llamaindex.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)](https://ollama.com/)
 

**CounselorAI** is a local, privacy-focused Retrieval-Augmented Generation (RAG) agent intially designed to assist USC Electrical Engineering (MSEE) students. It answers questions regarding degree requirements, course planning, and academic policies by strictly grounding responses in official university documentation.

> **Note:** At the current stage this project runs 100% locally. No data is sent to external APIs (OpenAI/Anthropic), ensuring student privacy and zero inference costs.

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
├── data/                    # USC PDF handbooks here(More need to be added)
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
> **Note:** If you wish to add additional pdf files for the model to use as trusted sources of information, skip to **Option 2** since the ingestion pipeline will need to be rerun.
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

## Future Roadmap

* [x] **Initial proof of concept:** Creating a Local Retrieval-Augmented Generation (RAG) agent pipeline where the flow of **Ingestion**, **Storage**, **Retrieval**  and **Synthesis** is followed and displayed on a basic UI. Accuracy in this stage in **NOT** a concern only that the pipeline behaves as expected.

* [x] **Add a 'preload' step:** By sending a dummy request to the model before actually letting the user make a request, it give the model time to load into the computer's RAM leading to less timeouts and a smoother user experience.

* [x] **Supply initial refined data:** This smaller batch of data will help test the system to ensure usability and relevance to users in a more confined manner before expanding to more comprehensive data.

* [x] **Dockerization:** Fully containerize the Streamlit app and ChromaDB for one-command deployment.

* [x] **Accuracy calibration:** Setting a low temperature(less creative, more encyclopedic knowledge), prompt engineering and other strategies can be implemented to improve accuracy within the bounds of the fairly simple llama3.2(~3B) model.

* [x] **Add more relevant Data:** For the best results we must provide as much relevant data in the initial stage of the RAG pipeline, this will help the system be as useful as possible to MSEE students and increase the chances that a potential question can be answer without the user having themselves having to add sources of data.

* [x] **Improve accuracy on the larger data set:** Ensure that the larger dataset does not mess with the model's ability to retrive relevant information. Further, more advanced techniques(eg. adding a Re-ranking step which would use a Cross-Encoder to rank the revelant sources of information and only passing the top-k to the LLM) can be used to try and improve the model's accuracy. This needs to be done while keeping in mind that the systems is designed to run on a local machine, which caps how compute hungry the techniques can be.

* [x] **Hybrid Search:** Implement keyword search alongside vector search for better acronym recognition (e.g., "EE 483"). For this, I combined the existing Vector Search (semantic understanding) with Keyword Search (BM25). This is allows the system to retrive specific course codes (e.g., "EE 559") or acronyms that vector embeddings sometimes miss. These were the exact steps:
    * Fetched the text from ChromaDB at startup.
    * Built an in-memory BM25 index (Keyword Retriever).
    * Fused the Vector Retriever and BM25 Retriever using Reciprocal Rank Fusion (RRF).


* [x] **Citations:** Update UI to display the specific page number of the source PDF used for the answer. 

* [x] **Final end to end test:** Rerun the whole project top to bottom, check edge cases, get feedback from potential users and implement finishing touches in order to finalize the project.
