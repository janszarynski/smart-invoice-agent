import os
import sys
import csv
import shutil
import argparse
import datetime
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# -----------------------------------------------------------------------------
# SCHEMAS
# -----------------------------------------------------------------------------

class InvoiceMetadata(BaseModel):
    file_name: str = Field(description="The original name of the processed file.")
    processed_at: str = Field(description="The timestamp when the file was processed in ISO format (e.g. YYYY-MM-DDTHH:MM:SSZ).")
    engine: str = Field(description="The reasoning engine used (e.g. Ollama (qwen2.5)).")

class ExtractedData(BaseModel):
    row_id: str = Field(description="The Row ID of the invoice. Return ONLY the raw numeric digits.")
    order_id: str = Field(description="The unique Order ID found at the bottom of the invoice.")
    order_date: str = Field(description="The Order Date. Parse and return strictly in YYYY-MM-DD format.")
    ship_mode: str = Field(description="The shipping mode listed as Ship Mode.")
    
    # Customer & Vendor Details
    vendor_name: str = Field(description="The name of the vendor or issuer of the invoice.")
    customer_name: str = Field(description="The name of the customer under Bill To.")
    
    # Shipping Address (Ship To)
    ship_to_postal_code: str = Field(description="The postal code in the Ship To address.")
    ship_to_city: str = Field(description="The city in the Ship To address.")
    ship_to_state: str = Field(description="The state in the Ship To address.")
    ship_to_country: str = Field(description="The country in the Ship To address.")
    
    # Product Details (Line Item)
    product_name: str = Field(description="The full name of the product sold under Item.")
    category: str = Field(description="The category of the product, found under Item.")
    sub_category: str = Field(description="The sub-category of the product, found under Item.")
    product_id: str = Field(description="The product ID, found under Item.")
    
    # Quantities & Rates
    quantity: int = Field(description="The quantity of the item sold.")
    rate: float = Field(description="The rate/unit cost of the product as a float.")
    item_amount: float = Field(description="The amount for the line item as a float.")
    
    # Totals
    subtotal: float = Field(description="The subtotal of the invoice.")
    discount_percent: float = Field(description="The discount percentage as a float.")
    discount_amount: float = Field(description="The total discount amount subtracted.")
    shipping_fee: float = Field(description="The shipping fee as a float.")
    total_payable: float = Field(description="The total amount payable listed as Total or Balance Due.")
    notes: str = Field(description="Any notes, terms, or messages at the bottom of the invoice.")

class InvoiceExtractionResult(BaseModel):
    metadata: InvoiceMetadata
    extracted_data: ExtractedData


# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------

def get_llm():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Ensure Ollama is running and model is pulled: ollama run <model_name>
    return ChatOllama(model=model_name, base_url=base_url, temperature=0)


# -----------------------------------------------------------------------------
# DOCUMENT PROCESSOR
# -----------------------------------------------------------------------------

def load_document(file_path: str) -> List[Document]:
    """
    Loads a document (PDF or CSV) from a local path and returns a list of Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == ".pdf":
        md_text = pymupdf4llm.to_markdown(file_path)
        return [Document(page_content=md_text, metadata={"source": file_path})]
    elif ext == ".csv":
        # Polish/European spreadsheets often use windows-1250 or ISO encodings.
        # We try UTF-8 first, with robust fallback strategies.
        try:
            loader = CSVLoader(file_path, encoding="utf-8")
            return loader.load()
        except UnicodeDecodeError:
            try:
                loader = CSVLoader(file_path, encoding="windows-1250")
                return loader.load()
            except Exception:
                # Use langchain's built-in autodetection as a final fallback
                loader = CSVLoader(file_path, autodetect_encoding=True)
                return loader.load()
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF and CSV files are supported.")

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits a list of Documents into smaller chunks with overlaps.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)


# -----------------------------------------------------------------------------
# VECTOR STORE
# -----------------------------------------------------------------------------

def get_embeddings():
    """
    Initializes and returns local HuggingFace embeddings to prevent VRAM context switching in Ollama.
    """
    # all-MiniLM-L6-v2 is a tiny, lightning-fast embedding model that runs entirely on CPU/RAM
    # We force device='cpu' so PyTorch doesn't reserve VRAM and steal it from Ollama.
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}
    )

def save_vector_store(documents, store_dir: str = None, save_to_disk: bool = True):
    """
    Creates a new FAISS vector store from document chunks and optionally saves it locally.
    """
    if store_dir is None:
        store_dir = os.getenv("VECTOR_STORE_DIR", ".faiss_store")
        
    embeddings = get_embeddings()
    print(f"Creating local FAISS index...", file=sys.stderr)
    vector_store = FAISS.from_documents(documents, embeddings)
    
    if save_to_disk:
        vector_store.save_local(store_dir)
        print("FAISS index saved successfully.", file=sys.stderr)
        
    return vector_store

def load_vector_store(store_dir: str = None):
    """
    Loads an existing FAISS vector store from the local directory.
    Returns None if the directory does not exist or index is missing.
    """
    if store_dir is None:
        store_dir = os.getenv("VECTOR_STORE_DIR", ".faiss_store")
        
    if not os.path.exists(store_dir):
        return None
        
    embeddings = get_embeddings()
    print(f"Loading local FAISS index from: {store_dir}...", file=sys.stderr)
    # allow_dangerous_deserialization is required to load locally saved FAISS files.
    # It is safe because this is a purely local desktop workspace.
    try:
        vector_store = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        print(f"Warning: Failed to load vector store from {store_dir}: {e}", file=sys.stderr)
        return None


# -----------------------------------------------------------------------------
# ORCHESTRATION / MAIN PIPELINE
# -----------------------------------------------------------------------------

def parse_and_standardize_date(date_str: str) -> str:
    if not date_str:
        return ""
    date_str = date_str.strip()
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str, flags=re.IGNORECASE)
    for fmt in ("%b %d %Y", "%B %d %Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            dt = datetime.datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str

def clean_row_id(row_id_str: str) -> str:
    if not row_id_str:
        return ""
    return "".join(c for c in str(row_id_str) if c.isdigit())

def extract_row_id_from_filename(filename: str) -> str:
    if not filename:
        return ""
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if parts:
        digits = "".join(c for c in parts[-1] if c.isdigit())
        if digits:
            return digits
    return ""

def extract_customer_name_from_filename(filename: str) -> str:
    if not filename:
        return ""
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""

EXTRACTION_PROMPT = """You are an expert business assistant specializing in SuperStore retail and e-commerce invoice analysis.
Extract the required structured invoice fields from the retrieved document context.
Ensure all extracted monetary amounts and rates are returned as floating-point numbers.
Ensure all quantity fields are returned as integers.
If a field is completely missing and cannot be found in the context, return null.

CRITICAL INSTRUCTION: If a field is not present in the Retrieved Context, return null. Do NOT copy or hallucinate any values. Extracted values MUST perfectly match the text in the context.

Retrieved Context:
---------------------
{context}
---------------------

User Question: Extract the SuperStore invoice details from the context. Make sure to map:
1. row_id: The Invoice Number or Row ID. Return strictly the raw numeric digits.
2. vendor_name: The name of the vendor/issuer at the top.
3. customer_name: The name under Bill To.
4. order_date: The Date listed. Parse and return strictly in YYYY-MM-DD format.
5. ship_mode: The Ship Mode.
6. ship_to_postal_code: The numeric zip code listed under Ship To. If missing, leave EMPTY. Do NOT write city or state names here.
7. ship_to_city: The city listed under Ship To.
8. ship_to_state: The state listed under Ship To.
9. ship_to_country: The country listed under Ship To.
10. product_name: The full descriptive name of the product being purchased. Do NOT just write the category. Extract the actual full product name.
11. category: The product category listed. YOU MUST NOT LEAVE THIS BLANK. Extract it even if it's on the same line as the product.
12. sub_category: The product sub-category listed. YOU MUST NOT LEAVE THIS BLANK.
13. product_id: The Product ID code listed under Item. SEPARATE this from the category.
14. quantity: The quantity.
15. rate: The unit cost/rate as a float.
16. item_amount: The line item amount as a float.
17. subtotal: The Subtotal as a float. This is the sum before shipping and taxes. Do NOT confuse this with the final Total.
18. discount_percent: The percentage number in 'Discount'.
19. discount_amount: The discount value subtracted.
20. shipping_fee: The shipping fee listed under Shipping as a float. This is usually a small amount. Do NOT confuse this with the item amount.
21. total_payable: The total amount payable listed under Total or Balance Due.
22. order_id: The Order ID listed at the bottom under Terms.
23. notes: Any notes or terms at the bottom."""

def test_baseline():
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    print(f"Testing local LLM (Ollama) baseline with model={model_name}...", file=sys.stderr)
    llm = get_llm()
    try:
        response = llm.invoke("Hello, what is your purpose?")
        print("\nLLM Response:")
        print(response.content)
    except Exception as e:
        print(f"\nError communicating with Ollama: {e}", file=sys.stderr)
        print("\n[!] Please ensure that:", file=sys.stderr)
        print("  1. The Ollama application/service is running on your system.", file=sys.stderr)
        print(f"  2. You have pulled the model by running this in your terminal: ollama pull {model_name}", file=sys.stderr)

def ingest_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.", file=sys.stderr)
        return None
        
    print(f"Starting ingestion for file: {file_path}", file=sys.stderr)
    try:
        docs = load_document(file_path)
        print(f"Parsed {len(docs)} pages/records from the document.", file=sys.stderr)
        
        chunks = split_documents(docs)
        print(f"Split document into {len(chunks)} text chunks.", file=sys.stderr)
        
        save_vector_store(chunks)
        print("Ingestion and indexing completed successfully.\n", file=sys.stderr)
        return chunks
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        return None

def append_to_ledger(file_name: str, extracted_data: ExtractedData, engine: str):
    cleaned_row_id = clean_row_id(extracted_data.row_id)
    if not cleaned_row_id:
        cleaned_row_id = extract_row_id_from_filename(file_name)
    extracted_data.row_id = cleaned_row_id
    
    if not extracted_data.customer_name or str(extracted_data.customer_name).strip().upper() in ("N/A", "NULL", "NONE", ""):
        extracted_data.customer_name = extract_customer_name_from_filename(file_name)
        
    extracted_data.order_date = parse_and_standardize_date(extracted_data.order_date)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ledger_path = os.path.join(script_dir, "..", "data", "master_ledger.csv")
    
    if not os.path.exists(os.path.dirname(ledger_path)):
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
        
    file_exists = os.path.exists(ledger_path)
    
    headers = [
        "Processed At", "File Name", "Vendor Name", "Row ID", "Order ID", "Order Date", "Ship Mode",
        "Customer Name", "Postal Code", "City", "State", "Country",
        "Product Name", "Category", "Sub-Category", "Product ID",
        "Quantity", "Rate", "Item Amount", "Subtotal", "Discount %", "Discount Amount",
        "Shipping Fee", "Total Payable", "Notes", "Engine"
    ]
    
    if file_exists and os.path.getsize(ledger_path) > 0:
        try:
            with open(ledger_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                existing_headers = next(reader, None)
                
            if existing_headers and "Vendor Name" not in existing_headers:
                backup_path = os.path.join(os.path.dirname(ledger_path), "master_ledger_accounting_backup.csv")
                print(f"Old accounting schema detected. Backing up ledger to: {backup_path}...", file=sys.stderr)
                shutil.copy(ledger_path, backup_path)
                
                file_exists = False
                os.remove(ledger_path)
                print("SuperStore ledger initialized fresh.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to backup/pivot master ledger: {e}", file=sys.stderr)
            
    if file_exists and os.path.getsize(ledger_path) > 0:
        try:
            with open(ledger_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row_entry in reader:
                    same_file = row_entry.get("File Name") == file_name
                    same_row_id = False
                    if extracted_data.row_id and row_entry.get("Row ID"):
                        same_row_id = row_entry.get("Row ID") == str(extracted_data.row_id)
                    
                    if same_file or same_row_id:
                        reason = "file name already exists" if same_file else "duplicate Row ID detected"
                        print(f"[!] Duplicate detected ({reason}). Skipping ledger append to avoid double-logging.", file=sys.stderr)
                        return
        except Exception as e:
            print(f"Warning: Failed to check duplicates in ledger: {e}", file=sys.stderr)
    
    row = {
        "Processed At": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "File Name": file_name,
        "Vendor Name": extracted_data.vendor_name,
        "Row ID": extracted_data.row_id,
        "Order ID": extracted_data.order_id,
        "Order Date": extracted_data.order_date,
        "Ship Mode": extracted_data.ship_mode,
        "Customer Name": extracted_data.customer_name,
        "Postal Code": extracted_data.ship_to_postal_code,
        "City": extracted_data.ship_to_city,
        "State": extracted_data.ship_to_state,
        "Country": extracted_data.ship_to_country,
        "Product Name": extracted_data.product_name,
        "Category": extracted_data.category,
        "Sub-Category": extracted_data.sub_category,
        "Product ID": extracted_data.product_id,
        "Quantity": extracted_data.quantity,
        "Rate": f"{float(extracted_data.rate):.2f}" if extracted_data.rate is not None else "0.00",
        "Item Amount": f"{float(extracted_data.item_amount):.2f}" if extracted_data.item_amount is not None else "0.00",
        "Subtotal": f"{float(extracted_data.subtotal):.2f}" if extracted_data.subtotal is not None else "0.00",
        "Discount %": f"{float(extracted_data.discount_percent):.1f}" if extracted_data.discount_percent is not None else "0.0",
        "Discount Amount": f"{float(extracted_data.discount_amount):.2f}" if extracted_data.discount_amount is not None else "0.00",
        "Shipping Fee": f"{float(extracted_data.shipping_fee):.2f}" if extracted_data.shipping_fee is not None else "0.00",
        "Total Payable": f"{float(extracted_data.total_payable):.2f}" if extracted_data.total_payable is not None else "0.00",
        "Notes": extracted_data.notes if extracted_data.notes else "",
        "Engine": engine
    }
    
    try:
        with open(ledger_path, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists or os.path.getsize(ledger_path) == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"Successfully appended row to ledger: {ledger_path}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to append row to CSV ledger: {e}", file=sys.stderr)

def extract_invoice_data(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
    _, ext = os.path.splitext(file_path.lower())
    if ext != ".pdf":
        raise ValueError(f"Skipped file '{os.path.basename(file_path)}' because it is not a PDF invoice.")
        
    print(f"Ingesting {file_path} on the fly...", file=sys.stderr)
    docs = load_document(file_path)
    chunks = split_documents(docs)
    vector_store = save_vector_store(chunks, save_to_disk=False)
    
    print("Retrieving context from local index...", file=sys.stderr)
    results = vector_store.similarity_search("SuperStore invoice, customer name, product, totals, row ID, order ID", k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    
    print("Invoking Qwen local reasoning model with strict JSON formatting...", file=sys.stderr)
    llm = get_llm()
    structured_llm = llm.with_structured_output(ExtractedData)
    formatted_prompt = EXTRACTION_PROMPT.format(context=context)
    
    extracted_data = structured_llm.invoke(formatted_prompt)
    
    base_filename = os.path.basename(file_path)
    
    cleaned_row_id = clean_row_id(extracted_data.row_id)
    if not cleaned_row_id:
        cleaned_row_id = extract_row_id_from_filename(base_filename)
    extracted_data.row_id = cleaned_row_id
    
    # Remove "Ship To:" or "Bill To:" if LLM accidentally included it
    if extracted_data.customer_name:
        extracted_data.customer_name = str(extracted_data.customer_name).replace("Ship To:", "").replace("Bill To:", "").strip()
        
    # If Customer Name is empty, or contains digits (LLM hallucinated an address here)
    if not extracted_data.customer_name or re.search(r'\d', str(extracted_data.customer_name)):
        # Fallback 1: Extract strictly from the "Bill To" section in the document text
        match = re.search(r'Bill To:(?:<br>)?\s*\**([A-Za-z\s]+?)\**\s*(?:\||<br>|$)', context)
        if match and len(match.group(1).strip()) > 2:
            extracted_data.customer_name = match.group(1).strip()
        else:
            # Fallback 2: Filename
            extracted_data.customer_name = extract_customer_name_from_filename(base_filename)
        
    extracted_data.order_date = parse_and_standardize_date(extracted_data.order_date)
    
    # Address Shift Fallback: if postal code contains letters, it is hallucinated text
    if extracted_data.ship_to_postal_code and re.search(r'[a-zA-Z]', str(extracted_data.ship_to_postal_code)):
        hallucinated_text = str(extracted_data.ship_to_postal_code)
        extracted_data.ship_to_postal_code = "" # Clear invalid postal code
        if extracted_data.ship_to_state in ("India", "United States", "", None):
            clean_state = hallucinated_text.replace(", India", "").replace(", United States", "").strip(" ,")
            if clean_state:
                extracted_data.ship_to_state = clean_state
                
    # Address Shift Fallback: if City is empty but State has Country name, or City has State name
    if not extracted_data.ship_to_city and extracted_data.ship_to_country:
        # Pamiętaj, że czasem LLM w ogóle pominie miasto i stan, zostawiając same państwo
        if not extracted_data.ship_to_state or extracted_data.ship_to_state == extracted_data.ship_to_country:
            clean_context = re.sub(r'<[^>]+>|\*+', ' ', context)
            match = re.search(r'([A-Za-z\s]+),\s*([A-Za-z\s]+),\s*' + re.escape(extracted_data.ship_to_country), clean_context)
            if match:
                # bierzemy tylko ostatnią linijkę przed przecinkiem (często to imię i nazwisko \n miasto)
                city_raw = match.group(1).split('\n')[-1].strip()
                extracted_data.ship_to_city = city_raw
                extracted_data.ship_to_state = match.group(2).strip()
                
    # Ultimate Address Fallback (Kiedy cały adres przepadnie lub LLM się pogubi)
    match = re.search(r'Ship To:<br>\*\*([^*]+)\*\*(?:<br>\*\*([^*]+)\*\*)?', context)
    if match:
        part1 = match.group(1).replace('<br>', ' ').replace('\n', ' ').strip()
        part2 = match.group(2).replace('<br>', ' ').replace('\n', ' ').strip() if match.group(2) else ""
        full_address = part1 + " " + part2
        address_parts = [p.strip() for p in full_address.split(',')]
        if len(address_parts) >= 4:
            extracted_data.ship_to_postal_code = address_parts[-4]
            extracted_data.ship_to_city = address_parts[-3]
            extracted_data.ship_to_state = address_parts[-2]
            extracted_data.ship_to_country = address_parts[-1]
        elif len(address_parts) == 3:
            extracted_data.ship_to_city = address_parts[-3]
            extracted_data.ship_to_state = address_parts[-2]
            extracted_data.ship_to_country = address_parts[-1]
            
    # Ultimate Address Fallback V2 (Broken picture layouts)
    match2 = re.search(r'\|Ship Mode:\|.*?\|(?:<br>|\n)+((?:\|.*?\|(?:<br>|\n)+)*?)\|Item\|Quantity\|', context, re.DOTALL)
    if match2:
        block = match2.group(1)
        lines = block.split('\n')
        addr_parts_v2 = []
        for line in lines:
            line = line.replace('<br>', '').strip()
            if not line: continue
            # Usunięcie "Balance Due" oraz wszystkiego po nim, co pozwala odzyskać ukryty po lewej stronie stan/kraj
            line = re.sub(r'\|?\s*Balance Due:.*', '', line)
            parts = [p.strip() for p in line.strip('|').split('|') if p.strip()]
            if not parts: continue
            addr_parts_v2.append(parts[-1])
        full_addr = " ".join(addr_parts_v2).replace(' ,', ',')
        final_parts = [p.strip() for p in full_addr.split(',')]
        if len(final_parts) >= 4:
            extracted_data.ship_to_postal_code = final_parts[-4]
            extracted_data.ship_to_city = final_parts[-3]
            extracted_data.ship_to_state = final_parts[-2]
            extracted_data.ship_to_country = final_parts[-1]
        elif len(final_parts) == 3:
            extracted_data.ship_to_city = final_parts[-3]
            extracted_data.ship_to_state = final_parts[-2]
            extracted_data.ship_to_country = final_parts[-1]
            
    # Row ID Fallback
    if not extracted_data.row_id or len(str(extracted_data.row_id)) > 10:
        # Prawdziwy Row ID (np. 25445) zawsze znajduje się za znakiem # lub w nazwie pliku
        match = re.search(r'#\s*(\d{4,6})', context)
        if match:
            extracted_data.row_id = match.group(1)
        else:
            match = re.search(r'_(\d{4,6})\.pdf', base_filename, re.IGNORECASE)
            if match:
                extracted_data.row_id = match.group(1)
                
    # Standalone Postal Code Fallback
    if not extracted_data.ship_to_postal_code:
        # First try right after Ship To:
        match = re.search(r'Ship To:(?:<[^>]+>|\s|\*)*(\d{5})', context, re.IGNORECASE)
        if match:
            extracted_data.ship_to_postal_code = match.group(1)
        else:
            # Try to find a classic US zip and city pattern
            match = re.search(r'\b(\d{5}),\s+([A-Za-z\s]+)', context)
            if match:
                extracted_data.ship_to_postal_code = match.group(1)
                if not extracted_data.ship_to_city:
                    extracted_data.ship_to_city = match.group(2).strip()
                
    # Vendor Name Validation & Fallback
    # If LLM put the customer name as the vendor name, clear it
    if extracted_data.vendor_name and extracted_data.vendor_name == extracted_data.customer_name:
        extracted_data.vendor_name = ""
        
    if not extracted_data.vendor_name:
        # Fallback 1: Look for SuperStore at the beginning of the text
        if "SuperStore" in context[:500]:
            extracted_data.vendor_name = "SuperStore"
        else:
            # If not SuperStore, try to find a generic Store or Vendor name at the top
            match = re.search(r'^\s*\**([A-Za-z\s]+(?:Store|Corp|LLC|Inc\.?))\**', context, re.IGNORECASE)
            if match:
                extracted_data.vendor_name = match.group(1).strip()
        
    valid_modes = ["First Class", "Second Class", "Standard Class", "Same Day"]
    if not extracted_data.ship_mode or str(extracted_data.ship_mode).strip() not in valid_modes:
        for m in valid_modes:
            if m in context:
                extracted_data.ship_mode = m
                break
                
    if not extracted_data.product_id or not re.match(r'^[A-Z]{3}-[A-Z]{2}-\d{4,5}$', str(extracted_data.product_id).strip()):
        # Search the entire Markdown context as an ultimate fallback
        match = re.search(r'([A-Z]{3}-[A-Z]{2}-\d{4,5})', context)
        if match:
            extracted_data.product_id = match.group(1)
            
        # Clean from other fields just in case
        if extracted_data.product_id:
            if extracted_data.category:
                extracted_data.category = extracted_data.category.replace(extracted_data.product_id, "").strip(" ,")
            if extracted_data.sub_category:
                extracted_data.sub_category = extracted_data.sub_category.replace(extracted_data.product_id, "").strip(" ,")
                
    # Dynamic Product Name Fallback
    lines = context.strip().split('\n')
    extracted_product_name = ""
    for i, line in enumerate(lines):
        match = re.match(r'^\|\s*\**([^*\|]+?)\**\s*\|(?:\|)*\s*\d+\s*\|', line)
        if match:
            extracted_product_name = match.group(1).strip()
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                next_match = re.match(r'^\|\s*\**([^*\|]+?)\**\s*\|(?:\||\s)*$', next_line)
                if next_match:
                    next_text = next_match.group(1).strip()
                    if re.search(r'[A-Z]{3}-[A-Z]{2}-\d+', next_text) or "Furniture" in next_text or "Office Supplies" in next_text or "Technology" in next_text:
                        break
                    extracted_product_name += " " + next_text
                else:
                    break
            break
            
    if extracted_product_name:
        # If the LLM output contains product ID (merged lines bug)
        if extracted_data.product_id and str(extracted_data.product_id) in str(extracted_data.product_name):
            extracted_data.product_name = extracted_product_name
        # If the regex found a longer name (missed multiline bug)
        elif len(extracted_product_name) > len(str(extracted_data.product_name)) and extracted_product_name.startswith(str(extracted_data.product_name).strip(" ,")):
            extracted_data.product_name = extracted_product_name
            
    # Category / Sub-Category Split Fallback
    # If the LLM dumped everything into category separated by comma, split it.
    if extracted_data.category and ',' in str(extracted_data.category):
        parts = [p.strip() for p in str(extracted_data.category).split(',')]
        if len(parts) >= 2:
            extracted_data.sub_category = parts[0]
            extracted_data.category = parts[1]
    # Check the reverse just in case LLM dumped everything into sub_category
    if extracted_data.sub_category and ',' in str(extracted_data.sub_category):
        parts = [p.strip() for p in str(extracted_data.sub_category).split(',')]
        extracted_data.sub_category = parts[0]
        if len(parts) >= 2:
            valid_categories = ["Office Supplies", "Furniture", "Technology"]
            if not extracted_data.category or extracted_data.category not in valid_categories:
                extracted_data.category = parts[1]
            
    # Swap Category and Sub-category if the LLM placed them in the wrong order
    valid_categories = ["Office Supplies", "Furniture", "Technology"]
    if extracted_data.sub_category in valid_categories and extracted_data.category not in valid_categories:
        temp = extracted_data.category
        extracted_data.category = extracted_data.sub_category
        extracted_data.sub_category = temp
        
    if extracted_data.notes:
        notes_str = str(extracted_data.notes).replace('\n', ' ').replace('\r', ' ').strip()
        notes_str = re.split(r'(?i)\b(?:Terms|Order ID)\b', notes_str)[0].strip(': \n\r')
        extracted_data.notes = notes_str
        
    if not extracted_data.notes or str(extracted_data.notes).strip().lower() in ("null", "n/a", "none", ""):
        # Prawdziwy, strukturalny Fallback (Szukamy czegokolwiek, co znajduje się przed 'Terms:' lub 'Order ID:')
        match = re.search(r'([^>\|\n]+?)\s*(?:<br>|\n|\|)*\s*(?:Terms:|Order ID\s*:)', context, re.IGNORECASE)
        if match:
            potential_note = match.group(1).strip()
            # Sprawdzamy, czy to nie jest kwota (np. Total Payable)
            if len(potential_note) > 3 and not re.match(r'^[\$\d.,\s]+$', potential_note):
                extracted_data.notes = potential_note
                    
    # Mathematical Safeguard for Discount
    # Subtotal + Shipping Fee - Total Payable = Discount Amount
    calculated_discount = round(extracted_data.subtotal + extracted_data.shipping_fee - extracted_data.total_payable, 2)
    if calculated_discount > 0 and abs(calculated_discount - extracted_data.discount_amount) > 0.05:
        extracted_data.discount_amount = calculated_discount
    elif extracted_data.discount_percent > 0 and extracted_data.discount_amount == 0:
        extracted_data.discount_amount = round(extracted_data.item_amount * (extracted_data.discount_percent / 100.0), 2)
                    
    # Safeguard: LLM often copies total_payable into subtotal. Fix the math if obvious.
    if extracted_data.subtotal == extracted_data.total_payable and extracted_data.shipping_fee > 0:
        if extracted_data.item_amount > 0:
            extracted_data.subtotal = extracted_data.item_amount
            
    # Mathematical Safeguard for Rate (Fix hallucinated rate)
    if extracted_data.quantity > 0 and extracted_data.item_amount > 0:
        calculated_rate = round(extracted_data.item_amount / extracted_data.quantity, 2)
        if abs(calculated_rate - extracted_data.rate) > 0.05:
            extracted_data.rate = calculated_rate
            
    # Global Cleanup for Hallucinations on Empty/Broken Invoices
    hallucination_patterns = [
        r'^null$', r'^n/a$', r'^none$', r'^bill to:\s*null$',
        r'^ship to:\s*null$', r'^bill to:$', r'^ship to:$', r'^missing$'
    ]
    for field_name, value in extracted_data.__dict__.items():
        if isinstance(value, str):
            val_str = value.strip()
            for pattern in hallucination_patterns:
                if re.match(pattern, val_str, re.IGNORECASE):
                    setattr(extracted_data, field_name, "")
                    break
            
    engine_name = f"Ollama ({os.getenv('OLLAMA_MODEL', 'qwen2.5')})"
    processed_at_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    metadata = InvoiceMetadata(
        file_name=base_filename,
        processed_at=processed_at_str,
        engine=engine_name
    )
    
    extraction_result = InvoiceExtractionResult(
        metadata=metadata,
        extracted_data=extracted_data
    )
    
    append_to_ledger(base_filename, extracted_data, engine_name)
    
    return extraction_result.model_dump_json(indent=2)

def process_absolute_path(file_path: str):
    try:
        json_result = extract_invoice_data(file_path)
        print(json_result)
    except Exception as e:
        print(f"Error running structured extraction: {e}", file=sys.stderr)
        sys.exit(1)

def query_rag(query: str):
    print(f"Searching index for query: '{query}'", file=sys.stderr)
    vector_store = load_vector_store()
    
    if vector_store is None:
        print("\n[!] Error: No local vector index found. Please ingest a document first using:", file=sys.stderr)
        print("  python src/main.py --ingest <file_path>", file=sys.stderr)
        return
        
    try:
        results = vector_store.similarity_search(query, k=4)
        if not results:
            print("No relevant chunks found in the index.", file=sys.stderr)
            context = "No retrieved context."
        else:
            print(f"Retrieved {len(results)} relevant text chunks from FAISS.", file=sys.stderr)
            context = "\n\n".join([doc.page_content for doc in results])
            
        print("Sending prompt to Qwen reasoning model...", file=sys.stderr)
        PROMPT_TEMPLATE = """You are an expert business assistant specializing in invoice and document analysis.
Use the following pieces of retrieved context to answer the user's question.
If the answer cannot be found in the context, say "I cannot find the answer in the provided documents." Do not try to make up an answer.
Keep your answer factual, direct, and structured.

Retrieved Context:
---------------------
{context}
---------------------

User Question: {query}

Answer:"""
        formatted_prompt = PROMPT_TEMPLATE.format(context=context, query=query)
        
        llm = get_llm()
        response = llm.invoke(formatted_prompt)
        
        print("\n=== RAG Pipeline Response ===")
        print(response.content)
        print("=============================\n")
    except Exception as e:
        print(f"Error executing RAG query: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Smart Invoice & E-commerce Agent - RAG CLI")
    parser.add_argument("--ingest", "-i", type=str, help="Path to a PDF or CSV file to ingest and index locally.")
    parser.add_argument("--query", "-q", type=str, help="Natural language query to search the index and answer.")
    parser.add_argument("--path", "-p", type=str, help="Ingest a file dynamically and output ONLY raw structured JSON data to stdout.")
    
    args = parser.parse_args()
    
    if args.path is not None:
        process_absolute_path(args.path)
        return
        
    if args.ingest is None and args.query is None:
        test_baseline()
        return
        
    if args.ingest is not None:
        ingest_file(args.ingest)
        
    if args.query is not None:
        query_rag(args.query)

if __name__ == "__main__":
    main()
