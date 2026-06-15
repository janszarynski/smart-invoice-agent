# Smart Invoice & E-commerce Agent (Local RAG Pipeline)

An autonomous, 100% local, and private AI-powered document processing pipeline. The system monitors a local directory for incoming financial and e-commerce documents (PDFs), processes them using a Retrieval-Augmented Generation (RAG) workflow, extracts critical structured business metrics using a local open-source LLM (Llama 3.2 via Ollama), and appends the structured results into a centralized Excel/CSV ledger.

## Performance
With optimized embedding generation and memory management, this pipeline extracts structured JSON data from a standard PDF invoice in **~27 seconds** on standard consumer hardware, entirely offline.

## Core Architecture
1. **Ingestion Layer (n8n):** Monitors local directories, captures new files, executes the Python extraction script, and handles file archival.
2. **Orchestration Layer (Python & LangChain):** Handles document loading, text chunking, and LLM prompting.
3. **Reasoning Layer (Local LLM via Ollama):** Runs **Llama 3.2** locally to ensure absolute data privacy and zero API usage costs.
4. **Embedding & Storage Layer:** Uses CPU-bound `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) to free up GPU VRAM. Implements transient, in-memory FAISS vector indexing for lightning-fast retrieval without file-lock collisions.

## Tech Stack
- **Python 3.11+**
- **LangChain & LangChain-Community**
- **Ollama** (Llama 3.2)
- **FAISS** (In-memory Vector Store)
- **Sentence-Transformers** (all-MiniLM-L6-v2)
- **n8n** (Workflow Orchestration)

## Setup & Installation

### 1. Prerequisites
- Install [Python 3.11+](https://www.python.org/downloads/)
- Install [Ollama](https://ollama.com/)
- Install [Node.js / npm](https://nodejs.org/) (for running n8n locally)

### 2. Environment Setup
Clone the repository and set up your virtual environment:

```bash
git clone https://github.com/your-username/smart-invoice-agent.git
cd smart-invoice-agent

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Preparation
Ensure you have pulled the required Ollama model:
```bash
ollama pull llama3.2
```

Create an environment file:
```bash
cp .env.example .env
```

### 4. Running the System
We provided a convenient batch script to launch the AI engine and the n8n dashboard automatically:
```bash
.\start_n8n.bat
```

1. Open `http://localhost:5678` in your browser.
2. Import the `n8n_workflow.json` (if provided) to set up your nodes.
3. Activate the workflow.
4. Drop a PDF invoice into the `inbox` folder. The system will automatically process it, extract the structured data to `data/master_ledger.csv`, and move the PDF to the `archive` folder.

## Features
- **Zero API Costs:** Runs 100% locally.
- **Privacy First:** No documents are ever sent to external servers.
- **Neuro-Symbolic Architecture:** Combines the contextual reasoning of Llama 3.2 with deterministic Python (Regex) fail-safes (Smart Fallbacks) for rigid fields (math calculations, zip codes, fractured addresses) to guarantee 100% extraction accuracy even when the LLM hallucinates or PDF parsing breaks table structures.
- **De-Duplication Check:** Built-in safeguards to prevent double-logging invoices into the master ledger.
- **Strict JSON Enforcement:** Forces the reasoning model to return strictly structured data objects mapping to predefined schemas.
- **VRAM Optimized:** Embeddings run on the CPU to ensure the GPU is entirely dedicated to fast token generation by Llama 3.2.

## Acknowledgments
- **Test Dataset:** Special thanks to [femstac/Sample-Pdf-invoices](https://github.com/femstac/Sample-Pdf-invoices) for providing the 1000+ PDF invoice dataset used for training, testing, and hardening the fallback logic of this pipeline. Their contribution to the open-source community made this project possible.
