# Smart Invoice & E-commerce Agent (Local RAG Pipeline)

## Project Overview
An autonomous, local AI-powered document processing and analysis pipeline. The system monitors a local directory for incoming financial and e-commerce documents (PDFs/CSVs), processes them using a Retrieval-Augmented Generation (RAG) workflow in Python, extracts critical structured business metrics using a local open-source LLM (via Ollama), and appends the structured results into a centralized ledger.

## Core Architecture
1. **Ingestion Layer (n8n):** Monitors local directories, captures new files, passes file paths to the execution script, and logs the structured JSON output into Excel/CSV files.
2. **Orchestration Layer (Python & LangChain):** Handles document loading, text chunking, embedding generation, vector database querying, and LLM prompting.
3. **Reasoning Layer (Local LLM via Ollama):** Runs open-source models (e.g., Llama 3/Mistral via Ollama) locally to ensure absolute data privacy and zero API usage costs.
4. **Storage Layer (Vector DB):** A lightweight, file-based vector database (FAISS or Chroma) to manage document embeddings locally without dedicated infrastructure.

---

## Tech Stack
- **Orchestration:** Python 3.11+, LangChain, LangChain-Community
- **Automation / Workflow:** n8n (Local Docker or npm setup)
- **Local LLM Engine:** Ollama (Llama 3 / Mistral)
- **Vector Database:** FAISS or Chroma (In-memory/local storage)
- **Environment Management:** `python-dotenv`

---

## Technical Backlog & Implementation Steps

### Phase 1: Environment Setup & Local LLM Factory
- [x] **Task 1.1:** Initialize the project environment. Create a virtual environment (`venv`), a `.env` template file, and a `requirements.txt` specifying `langchain`, `langchain-community`, `ollama`, and `python-dotenv`.
- [x] **Task 1.2:** Build a robust LLM factory module (`models.py`). This script must load the configured Ollama settings (e.g., `OLLAMA_BASE_URL` and `OLLAMA_MODEL` from `.env`) and return a `ChatOllama` instance.
- [x] **Task 1.3:** Create a baseline verification script (`main.py`) to test text completion through the local Ollama factory module.

### Phase 2: Local RAG Pipeline (Document Engine)
- [x] **Task 2.1:** Implement a document loading utility capable of ingestion for PDFs (`pypdf`) and CSV data. Use LangChain's `RecursiveCharacterTextSplitter` to chunk documents with appropriate token overlaps.
- [x] **Task 2.2:** Configure a local embedding strategy. Implement darmowy embedding generation (e.g., via `HuggingFaceEmbeddings` or Ollama’s native embedding API) and connect it to a local **FAISS** index that saves automatically to disk.
- [x] **Task 2.3:** Build the primary Retrieval Chain. The script must accept a natural language query, locate relevant text chunks within the vector store, inject them into a structured business prompt template, and output a clean response.

### Phase 3: Structured Extraction & Terminal Interface
- [x] **Task 3.1:** Enforce structured JSON output formatting from the LLM chain. The model must strictly extract structural entities: `invoice_date`, `vendor_name`, `tax_id`, `net_amount`, `vat_amount`, and `gross_amount`.
- [x] **Task 3.2:** Expose a clean command-line interface (CLI) using `argparse`. The backend must accept an absolute file path as an argument (e.g., `python main.py --path /docs/invoice.pdf`) and print only the final ustructured JSON dictionary to stdout.

### Phase 4: n8n Workflow Orchestration
- [x] **Task 4.1:** Establish a local n8n server. 
- [x] **Task 4.2:** Construct a visual workflow:
    - **Trigger:** *Local File Trigger node* watching a specific file directory.
    - **Execution:** *Execute Command node* invoking the Python CLI tool, passing the newly detected file path dynamically.
    - **Storage:** *Append to Excel/CSV node* that parses the returned JSON string and appends it cleanly as a new row in a master database.

### Phase 5: Performance Optimization & Reliability
- [x] **Task 5.1:** Switched embedding generation from Ollama to strictly CPU-bound `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) via Python to entirely free up GPU VRAM for the reasoning model.
- [x] **Task 5.2:** Upgraded the local reasoning model to `Llama 3.2` and optimized CLI execution to reduce end-to-end processing time per invoice from >2 minutes to ~27 seconds.
- [x] **Task 5.3:** Refactored `FAISS` to run entirely in-memory for on-the-fly invoice extraction, avoiding disk I/O and eliminating file lock collisions (`Permission denied`) when n8n runs parallel triggers.
- [x] **Task 5.4:** Added robust CLI stderr suppression (`2>NUL`) to prevent `n8n` from throwing false-positive errors on third-party progress bars (e.g., `tqdm`).

### Phase 6: Neuro-Symbolic AI & Smart Fallbacks (Architectural Principle)
- [x] **Task 6.1:** **NO HARDCODING RULE:** The system must adhere to an AI-first architecture. Hardcoding variables (e.g., forcing a specific Vendor Name or extracting Customer Name strictly from a filename without letting the LLM try) is strictly prohibited as it destroys the generic nature of the pipeline.
- [x] **Task 6.2:** **Context-Aware Fallbacks:** Instead of overriding the LLM with hardcoded text, implement deterministic Python fallback mechanisms (Regex) that dynamically parse the `PyMuPDF4LLM` Markdown context (e.g., scanning the `Bill To:` block for names or `Store` headers for vendors) when the LLM hallucinates or returns empty values.
- [x] **Task 6.3:** **Category Swap Fallback:** Automatically swap 'Category' and 'Sub-Category' if the LLM places known sub-categories (e.g., Fasteners, Phones) into the parent Category field.
- [x] **Task 6.4:** **Dynamic Notes Extractor:** Utilize a structural regex to extract any notes dynamically by locating text between the financial totals and the `Terms:`/`Order ID:` footers, abandoning any hardcoded phrase matches like "Thanks for your business!".
- [x] **Task 6.5:** **Strict Monetary Formatting:** Enforce a strict `{:.2f}` string formatting cast on all monetary float values in Python prior to CSV writing to prevent trailing zeros from dropping (e.g., forcing `47.3` into `47.30`).
- [x] **Task 6.6:** **Negative Prompt Optimization:** Remove all example data (like `Mar 06 2012`) from Pydantic `Field` descriptions and negative constraints within the prompt, as small LLMs (like Llama 3.2 3B) suffer from negative prompting hallucinations and copy forbidden examples blindly.
- [x] **Task 6.7:** **Ultimate Address and Mathematical Fallbacks:** Address structural hallucination and table-breaking issues on complex or "broken picture" PDFs by using a two-tier deterministc Regex pipeline to reconstruct addresses that span across pipe delimiters (`|`). Similarly, enforce mathematical fallback validation for rate and discount calculation to overwrite any numerical LLM hallucinations.
---

## Expected Output Format
The final Python system execution must return a standardized JSON string structured as follows to ensure seamless integration with n8n workflow nodes:

```json
{
  "metadata": {
    "file_name": "invoice_2026_05.pdf",
    "processed_at": "2026-05-24T18:25:00Z",
    "engine": "Ollama (Llama 3)"
  },
  "extracted_data": {
    "vendor_name": "Example Corp Sp. z o.o.",
    "tax_id": "PL1234567890",
    "invoice_date": "2026-05-20",
    "currency": "PLN",
    "net_amount": 1000.00,
    "vat_amount": 230.00,
    "gross_amount": 1230.00
  }
}