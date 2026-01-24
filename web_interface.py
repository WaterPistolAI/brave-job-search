"""
Gradio web interface for managing and monitoring the job search system.
Provides dashboard, provider management, and database access.
"""

import gradio as gr
import sqlite3
import json
import os
import pandas as pd
import sys
import threading
import queue
import time
from datetime import datetime
from dotenv import load_dotenv
from job_processor import JobEmbedder, process_jobs_from_json
import logging
import brave_job_search
from ats_domains import SUPPORTED_ATS_DOMAINS
from pathlib import Path
import tempfile

load_dotenv()

# Configuration
DB_PATH = os.environ.get("DB_PATH", "job_search.db")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
ENV_FILE = ".env"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="web_interface.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Global variables for thread management
search_thread = None
processor_thread = None
search_cancelled = False
processor_cancelled = False
search_lock = threading.Lock()
processor_lock = threading.Lock()


def initialize_database():
    """Initialize the database with required tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Jobs table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            snippet TEXT,
            status TEXT DEFAULT 'pending',
            verified_at TIMESTAMP,
            scraped_at TIMESTAMP,
            job_description TEXT,
            requirements TEXT,
            benefits TEXT,
            location TEXT,
            company TEXT,
            salary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Add source_adapter column if it doesn't exist (for backward compatibility)
    try:
        cursor.execute(
            "ALTER TABLE jobs ADD COLUMN source_adapter TEXT DEFAULT 'unknown'"
        )
        logging.info("Added source_adapter column to jobs table")
    except sqlite3.OperationalError:
        # Column already exists, which is fine
        pass

    # Processing log table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS processing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            action TEXT,
            status TEXT,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    """
    )

    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_url ON jobs(url)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_log_job_id ON processing_log(job_id)"
    )

    conn.commit()
    conn.close()
    logging.info("Database initialized successfully")


def get_db_connection():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_env_config():
    """Load configuration from .env file."""
    config = {
        "BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", ""),
        "BRAVE_TARGET_QUERIES": os.environ.get("BRAVE_TARGET_QUERIES", "[]"),
        "BRAVE_ATS_DOMAINS": os.environ.get("BRAVE_ATS_DOMAINS", "[]"),
        "BRAVE_MONTHLY_LIMIT": os.environ.get("BRAVE_MONTHLY_LIMIT", "2000"),
        "BRAVE_RATE_LIMIT": os.environ.get("BRAVE_RATE_LIMIT", "1.1"),
        "BRAVE_FRESHNESS": os.environ.get("BRAVE_FRESHNESS", "pd"),
        "SCRAPE_RATE_LIMIT": os.environ.get("SCRAPE_RATE_LIMIT", "2.0"),
        "GOOGLE_SEARCH_API_KEY": os.environ.get("GOOGLE_SEARCH_API_KEY", ""),
        "GOOGLE_SEARCH_ID": os.environ.get("GOOGLE_SEARCH_ID", ""),
        "GOOGLE_SEARCH_RATE_LIMIT": os.environ.get("GOOGLE_SEARCH_RATE_LIMIT", "1.1"),
        "SEARCH_PROVIDER": os.environ.get("SEARCH_PROVIDER", "brave"),
        "SEARCH_PROVIDERS": os.environ.get("SEARCH_PROVIDERS", "brave"),
    }
    return config


def save_env_config(config):
    """Save configuration to .env file."""
    try:
        # Read existing .env file
        env_lines = []
        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, "r") as f:
                env_lines = f.readlines()

        # Update or add configuration
        updated_lines = []
        keys_to_update = {
            "BRAVE_TARGET_QUERIES": config["BRAVE_TARGET_QUERIES"],
            "BRAVE_ATS_DOMAINS": config["BRAVE_ATS_DOMAINS"],
            "BRAVE_MONTHLY_LIMIT": config["BRAVE_MONTHLY_LIMIT"],
            "BRAVE_RATE_LIMIT": config["BRAVE_RATE_LIMIT"],
            "BRAVE_FRESHNESS": config["BRAVE_FRESHNESS"],
            "SCRAPE_RATE_LIMIT": config["SCRAPE_RATE_LIMIT"],
            "GOOGLE_SEARCH_API_KEY": config["GOOGLE_SEARCH_API_KEY"],
            "GOOGLE_SEARCH_ID": config["GOOGLE_SEARCH_ID"],
            "GOOGLE_SEARCH_RATE_LIMIT": config["GOOGLE_SEARCH_RATE_LIMIT"],
            "SEARCH_PROVIDER": config["SEARCH_PROVIDER"],
            "SEARCH_PROVIDERS": config.get("SEARCH_PROVIDERS", "brave"),
        }

        # Keep existing lines that aren't being updated
        for line in env_lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith("#"):
                key = line_stripped.split("=")[0].strip()
                if key not in keys_to_update:
                    updated_lines.append(line)

        # Add updated configuration
        for key, value in keys_to_update.items():
            updated_lines.append(f"{key} = {value}\n")

        # Write back to .env file
        with open(ENV_FILE, "w") as f:
            f.writelines(updated_lines)

        # Update environment variables
        for key, value in keys_to_update.items():
            os.environ[key] = str(value)

        logging.info("Configuration saved successfully")
        return True
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")
        return False


def get_statistics():
    """Get job statistics."""
    conn = get_db_connection()

    stats = {
        "Total Jobs": conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
        "Pending": conn.execute(
            'SELECT COUNT(*) FROM jobs WHERE status = "pending"'
        ).fetchone()[0],
        "Verified": conn.execute(
            'SELECT COUNT(*) FROM jobs WHERE status = "verified"'
        ).fetchone()[0],
        "Scraped": conn.execute(
            'SELECT COUNT(*) FROM jobs WHERE status = "scraped"'
        ).fetchone()[0],
        "Embedded": conn.execute(
            'SELECT COUNT(*) FROM jobs WHERE status = "embedded"'
        ).fetchone()[0],
        "Inactive": conn.execute(
            'SELECT COUNT(*) FROM jobs WHERE status = "inactive"'
        ).fetchone()[0],
        "Failed": conn.execute(
            'SELECT COUNT(*) FROM jobs WHERE status LIKE "%_failed"'
        ).fetchone()[0],
    }

    conn.close()
    return stats


def get_recent_jobs():
    """Get recent jobs as a dataframe."""
    conn = get_db_connection()

    jobs = conn.execute(
        """
        SELECT id, title, company, location, status, created_at
        FROM jobs
        ORDER BY created_at DESC
        LIMIT 20
    """
    ).fetchall()

    conn.close()

    df = pd.DataFrame([dict(job) for job in jobs])
    return df


def get_jobs_filtered(status_filter="all", search_query=""):
    """Get jobs with filtering."""
    conn = get_db_connection()

    query = "SELECT id, title, company, location, status, url, created_at FROM jobs WHERE 1=1"
    params = []

    if status_filter != "all":
        query += " AND status = ?"
        params.append(status_filter)

    if search_query:
        query += " AND (title LIKE ? OR company LIKE ? OR location LIKE ?)"
        search_pattern = f"%{search_query}%"
        params.extend([search_pattern, search_pattern, search_pattern])

    query += " ORDER BY created_at DESC LIMIT 100"

    jobs = conn.execute(query, params).fetchall()
    conn.close()

    df = pd.DataFrame([dict(job) for job in jobs])

    # URLs are displayed as plain text in the dataframe
    # Users can view the full URL in the job details section

    return df


def get_job_details(job_id):
    """Get detailed information about a specific job."""
    conn = get_db_connection()

    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    logs = conn.execute(
        """
        SELECT * FROM processing_log 
        WHERE job_id = ? 
        ORDER BY timestamp DESC
    """,
        (job_id,),
    ).fetchall()

    conn.close()

    if not job:
        return "Job not found", ""

    job_dict = dict(job)
    logs_df = pd.DataFrame([dict(log) for log in logs])

    # Create clickable link for URL
    url_link = f"[{job_dict['url']}]({job_dict['url']})"

    details = f"""
    **Title:** {job_dict['title']}
    **Company:** {job_dict['company'] or 'N/A'}
    **Location:** {job_dict['location'] or 'N/A'}
    **Status:** {job_dict['status']}
    **URL:** {url_link}
    **Created:** {job_dict['created_at']}
    
    **Job Description:**
    {job_dict['job_description'] or 'N/A'}
    
    **Requirements:**
    {job_dict['requirements'] or 'N/A'}
    
    **Benefits:**
    {job_dict['benefits'] or 'N/A'}
    
    **Salary:**
    {job_dict['salary'] or 'N/A'}
    """

    return details, logs_df


def get_providers():
    """Get current ATS domains and search queries."""
    config = load_env_config()

    try:
        ats_domains = json.loads(config["BRAVE_ATS_DOMAINS"])
        target_queries = json.loads(config["BRAVE_TARGET_QUERIES"])
    except json.JSONDecodeError:
        ats_domains = []
        target_queries = []

    # Return domains, queries, and checkbox states
    checkbox_states = {
        domain: domain in ats_domains for domain in SUPPORTED_ATS_DOMAINS
    }
    return ats_domains, target_queries, checkbox_states


def update_ats_domains(*checkbox_values):
    """Update ATS domains based on checkbox selection."""
    config = load_env_config()

    # Get selected domains from checkboxes
    selected_domains = []
    for i, (domain, checked) in enumerate(
        zip(SUPPORTED_ATS_DOMAINS.keys(), checkbox_values)
    ):
        if checked:
            selected_domains.append(domain)

    config["BRAVE_ATS_DOMAINS"] = json.dumps(selected_domains)

    if save_env_config(config):
        logging.info(f"Updated ATS domains: {selected_domains}")
        return "\n".join(selected_domains)
    else:
        return "Error: Failed to save configuration"


def add_query(query):
    """Add a new search query."""
    if not query or not query.strip():
        return "Error: Query is required", get_providers()

    query = query.strip()
    config = load_env_config()

    try:
        target_queries = json.loads(config["BRAVE_TARGET_QUERIES"])
    except json.JSONDecodeError:
        target_queries = []

    if query in target_queries:
        return "Error: Query already exists", get_providers()

    target_queries.append(query)
    config["BRAVE_TARGET_QUERIES"] = json.dumps(target_queries)

    if save_env_config(config):
        logging.info(f"Added query: {query}")
        return f"Success: Added {query}", get_providers()
    else:
        return "Error: Failed to save configuration", get_providers()


def remove_query(query):
    """Remove a search query."""
    if not query:
        return "Error: Query is required", get_providers()

    config = load_env_config()

    try:
        target_queries = json.loads(config["BRAVE_TARGET_QUERIES"])
    except json.JSONDecodeError:
        target_queries = []

    if query not in target_queries:
        return "Error: Query not found", get_providers()

    target_queries.remove(query)
    config["BRAVE_TARGET_QUERIES"] = json.dumps(target_queries)

    if save_env_config(config):
        logging.info(f"Removed query: {query}")
        return f"Success: Removed {query}", get_providers()
    else:
        return "Error: Failed to save configuration", get_providers()


def update_config(brave_rate_limit, scrape_rate_limit, monthly_limit, freshness):
    """Update configuration settings."""
    config = load_env_config()

    config["BRAVE_RATE_LIMIT"] = str(brave_rate_limit)
    config["SCRAPE_RATE_LIMIT"] = str(scrape_rate_limit)
    config["BRAVE_MONTHLY_LIMIT"] = str(monthly_limit)
    config["BRAVE_FRESHNESS"] = freshness

    if save_env_config(config):
        logging.info("Configuration updated")
        return "Success: Configuration updated", load_env_config()
    else:
        return "Error: Failed to save configuration", load_env_config()


def semantic_search(query, n_results=5):
    """Perform semantic search on job descriptions."""
    if not query or not query.strip():
        return "Error: Query is required", None

    try:
        embedder = JobEmbedder()
        results = embedder.search_similar_jobs(query, n_results=n_results)

        # Get additional details from database
        conn = get_db_connection()
        for job in results:
            db_job = conn.execute(
                """
                SELECT job_description, requirements, benefits, salary
                FROM jobs WHERE id = ?
            """,
                (job["job_id"],),
            ).fetchone()

            if db_job:
                job["job_description"] = db_job["job_description"]
                job["requirements"] = db_job["requirements"]
                job["benefits"] = db_job["benefits"]
                job["salary"] = db_job["salary"]

        conn.close()

        if not results:
            return "No results found", None

        # Format results as dataframe
        df = pd.DataFrame(results)
        logging.info(f"Semantic search: '{query}' returned {len(results)} results")
        return f"Found {len(results)} similar jobs", df
    except Exception as e:
        logging.error(f"Error in semantic search: {e}")
        return f"Error: {str(e)}", None


def export_jobs():
    """Export jobs as CSV."""
    conn = get_db_connection()

    jobs = conn.execute(
        """
        SELECT id, url, title, snippet, status, job_description, 
               requirements, benefits, location, company, salary,
               created_at, updated_at
        FROM jobs
        ORDER BY created_at DESC
    """
    ).fetchall()

    conn.close()

    df = pd.DataFrame([dict(job) for job in jobs])
    return df


def run_job_search():
    """Run the job search function directly."""
    global search_thread, search_cancelled

    with search_lock:
        if search_thread and search_thread.is_alive():
            yield "❌ Job search is already running. Please wait or cancel the current operation."
            return

        search_cancelled = False

    # Create a thread-safe queue for output
    output_queue = queue.Queue()

    def search_worker():
        """Worker function that runs the job search."""
        try:
            # Capture stdout to display in the UI
            import io
            from contextlib import redirect_stdout, redirect_stderr

            output_buffer = io.StringIO()

            # Reload environment variables to get latest configuration
            from dotenv import load_dotenv

            load_dotenv(override=True)

            # Check which search providers to use
            search_providers_str = os.environ.get("SEARCH_PROVIDERS", "brave")
            search_providers = [
                p.strip() for p in search_providers_str.split(",") if p.strip()
            ]

            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                if len(search_providers) > 1:
                    # Use multi-search for multiple providers
                    import multi_search

                    multi_search.run_multi_search(providers=search_providers)
                elif "google" in search_providers:
                    import google_search

                    google_search.main()
                else:
                    # Reload brave_job_search module to get latest environment variables
                    import importlib

                    importlib.reload(brave_job_search)
                    brave_job_search.main()

            output = output_buffer.getvalue()
            if output:
                output_queue.put(output)
            output_queue.put(
                f"\n✅ Job search completed successfully using {', '.join(search_providers).upper()}!"
            )
        except Exception as e:
            output_queue.put(f"❌ Error running job search: {str(e)}")
        finally:
            output_queue.put(None)  # Signal completion

    # Start the worker thread
    with search_lock:
        search_thread = threading.Thread(target=search_worker)
        search_thread.start()

    # Yield output as it becomes available
    while True:
        try:
            output = output_queue.get(timeout=0.1)
            if output is None:
                break
            yield output
        except queue.Empty:
            # Check if cancelled
            with search_lock:
                if search_cancelled:
                    yield "\n⏸️ Job search was cancelled by user."
                    break
            continue

    with search_lock:
        search_thread = None


def cancel_job_search():
    """Cancel the running job search."""
    global search_cancelled

    with search_lock:
        if search_thread and search_thread.is_alive():
            search_cancelled = True
            return "⏸️ Job search cancellation requested..."
        else:
            return "ℹ️ No job search is currently running."


def run_job_processor():
    """Run the job processor function directly."""
    global processor_thread, processor_cancelled

    with processor_lock:
        if processor_thread and processor_thread.is_alive():
            yield "❌ Job processor is already running. Please wait or cancel the current operation."
            return

        processor_cancelled = False

    # Create a thread-safe queue for output
    output_queue = queue.Queue()

    def processor_worker():
        """Worker function that runs the job processor."""
        try:
            # Capture stdout to display in the UI
            import io
            from contextlib import redirect_stdout, redirect_stderr

            output_buffer = io.StringIO()

            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                process_jobs_from_json()

            output = output_buffer.getvalue()
            if output:
                output_queue.put(output)
            output_queue.put("\n✅ Job processing completed successfully!")
        except Exception as e:
            output_queue.put(f"❌ Error running job processor: {str(e)}")
        finally:
            output_queue.put(None)  # Signal completion

    # Start the worker thread
    with processor_lock:
        processor_thread = threading.Thread(target=processor_worker)
        processor_thread.start()

    # Yield output as it becomes available
    while True:
        try:
            output = output_queue.get(timeout=0.1)
            if output is None:
                break
            yield output
        except queue.Empty:
            # Check if cancelled
            with processor_lock:
                if processor_cancelled:
                    yield "\n⏸️ Job processor was cancelled by user."
                    break
            continue

    with processor_lock:
        processor_thread = None


def cancel_job_processor():
    """Cancel the running job processor."""
    global processor_cancelled

    with processor_lock:
        if processor_thread and processor_thread.is_alive():
            processor_cancelled = True
            return "⏸️ Job processor cancellation requested..."
        else:
            return "ℹ️ No job processor is currently running."


def extract_text_from_file(file_path):
    """Extract text from a file using Docling (supports PDF, DOCX, HTML, PPTX, images, etc.)."""
    file_path = Path(file_path)

    try:
        from docling.document_converter import DocumentConverter

        # Use Docling to convert the document
        converter = DocumentConverter()
        result = converter.convert(str(file_path))

        # Export to markdown
        markdown_text = result.document.export_to_markdown()

        logging.info(f"Successfully extracted text from {file_path.name} using Docling")
        return markdown_text
    except ImportError:
        logging.error("Docling not installed. Install with: pip install docling")
        return None
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return None


def embed_user_document(document_type, file_path):
    """Embed a user document (resume or cover letter) in ChromaDB."""
    if not file_path:
        return "Error: No file provided", None

    # Extract text from file
    text = extract_text_from_file(file_path)
    if not text:
        return "Error: Failed to extract text from file", None

    try:
        # Initialize embedder
        embedder = JobEmbedder()

        # Generate embedding
        embedding = embedder.embedding_adapter.embed([text])[0]

        # Store in ChromaDB with a separate collection for user documents
        collection_name = "user_documents"
        collection = embedder.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Use a consistent ID for each document type (resume or cover_letter)
        doc_id = f"{document_type}_current"

        # Delete ALL documents with this document_type to ensure clean overwrite
        try:
            # Get all documents with this document_type
            all_docs = collection.get()
            if all_docs and all_docs["metadatas"]:
                # Find all IDs with matching document_type
                ids_to_delete = []
                for i, metadata in enumerate(all_docs["metadatas"]):
                    if metadata.get("document_type") == document_type:
                        ids_to_delete.append(all_docs["ids"][i])

                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
                    logging.info(
                        f"Deleted {len(ids_to_delete)} existing {document_type} document(s)"
                    )
        except Exception as e:
            logging.warning(f"Error deleting existing documents: {e}")

        # Add the new document
        collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[
                {
                    "document_type": document_type,
                    "file_name": Path(file_path).name,
                    "uploaded_at": datetime.now().isoformat(),
                }
            ],
            ids=[doc_id],
        )

        logging.info(f"Successfully embedded {document_type} document")
        return f"✅ Successfully embedded {document_type} document", text
    except Exception as e:
        logging.error(f"Error embedding {document_type} document: {e}")
        return f"❌ Error: {str(e)}", None


def search_user_documents(query, n_results=5):
    """Search user documents (resume/cover letter) for relevant content."""
    if not query or not query.strip():
        return "Error: Query is required", None

    try:
        embedder = JobEmbedder()

        # Get the user documents collection
        collection = embedder.client.get_collection(name="user_documents")

        # Generate query embedding
        query_embedding = embedder.embedding_adapter.embed([query])[0]

        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )

        if not results["documents"] or not results["documents"][0]:
            return "No documents found", None

        # Format results
        documents = []
        for i in range(len(results["documents"][0])):
            documents.append(
                {
                    "document_type": results["metadatas"][0][i]["document_type"],
                    "file_name": results["metadatas"][0][i]["file_name"],
                    "uploaded_at": results["metadatas"][0][i]["uploaded_at"],
                    "content": (
                        results["documents"][0][i][:500] + "..."
                        if len(results["documents"][0][i]) > 500
                        else results["documents"][0][i]
                    ),
                }
            )

        df = pd.DataFrame(documents)
        logging.info(
            f"User document search: '{query}' returned {len(documents)} results"
        )
        return f"Found {len(documents)} relevant document sections", df
    except Exception as e:
        logging.error(f"Error searching user documents: {e}")
        return f"Error: {str(e)}", None


def analyze_resume_and_rank_jobs(n_results=10):
    """
    Analyze resume and cover letter with LLM, generate search queries,
    and rank job listings by relevance.
    """
    try:
        embedder = JobEmbedder()

        # Get user documents (resume and cover letter)
        user_collection = embedder.client.get_collection(name="user_documents")
        user_docs = user_collection.get()

        if not user_docs or not user_docs["documents"]:
            return (
                "Error: No resume or cover letter found. Please upload your documents first.",
                None,
            )

        # Extract resume and cover letter text
        resume_text = ""
        cover_letter_text = ""
        for i, metadata in enumerate(user_docs["metadatas"]):
            doc_type = metadata.get("document_type", "")
            if doc_type == "resume":
                resume_text = user_docs["documents"][i]
            elif doc_type == "cover_letter":
                cover_letter_text = user_docs["documents"][i]

        if not resume_text:
            return "Error: No resume found. Please upload your resume first.", None

        # Use LLM to analyze and generate search queries
        try:
            from openai import OpenAI

            # Check for OpenAI API key (separate from embedding key)
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
            openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

            if not openai_api_key:
                return (
                    "Error: OpenAI API key not found. Please set OPENAI_API_KEY in your .env file for LLM-based job matching. "
                    "Note: This is different from OPENAI_EMBEDDING_API_KEY which is used for embeddings.",
                    None,
                )

            client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

            # Create prompt for LLM
            prompt = f"""Analyze the following resume and cover letter to identify the candidate's key qualifications, skills, experience, and career goals. Then generate 5-7 specific search queries that would be most effective for finding relevant job listings.

IMPORTANT: Keep your analysis CONCISE (3-4 sentences maximum). Focus on the most important qualifications and career goals. Do not cut off mid-sentence. Ensure your analysis is fully finished before generating the search queries.

RESUME:
{resume_text[:25000]}

COVER LETTER:
{cover_letter_text[:20000] if cover_letter_text else "No cover letter provided"}

Please provide your analysis in the following format:

ANALYSIS:
[3-4 sentence summary of the candidate's profile, key skills, experience level, and career goals]

SEARCH QUERIES:
1. [Query 1]
2. [Query 2]
3. [Query 3]
4. [Query 4]
5. [Query 5]
6. [Query 6]
7. [Query 7]

Make the queries specific and targeted, focusing on the candidate's unique combination of skills and experience."""

            # Call LLM
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert career advisor and job search specialist. Analyze resumes and cover letters to identify key qualifications and generate effective job search queries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=16384,
            )

            llm_response = response.choices[0].message.content

            # Check if response is valid
            if not llm_response:
                return "Error: LLM returned empty response. Please try again.", None

            # Extract search queries from LLM response
            search_queries = []
            lines = llm_response.split("\n")
            in_queries_section = False
            for line in lines:
                if "SEARCH QUERIES:" in line:
                    in_queries_section = True
                    continue
                if in_queries_section:
                    # Look for numbered lines
                    if line.strip() and any(
                        line.strip().startswith(str(i)) for i in range(1, 10)
                    ):
                        # Extract the query (remove the number and any leading punctuation)
                        query = line.strip()
                        for i in range(1, 10):
                            if query.startswith(f"{i}."):
                                query = query[2:].strip()
                                break
                            elif query.startswith(f"{i} "):
                                query = query[2:].strip()
                                break
                        if query:
                            search_queries.append(query)

            if not search_queries:
                # Fallback: use the entire response as a single query
                search_queries = [llm_response]

            logging.info(f"Generated {len(search_queries)} search queries from LLM")

        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return f"Error analyzing resume with LLM: {str(e)}", None

        # Perform semantic search for each query
        all_results = {}
        for query in search_queries:
            try:
                results = embedder.search_similar_jobs(query, n_results=n_results)
                for job in results:
                    job_id = job["job_id"]
                    if job_id not in all_results:
                        all_results[job_id] = {
                            "job_id": job_id,
                            "url": job["url"],
                            "title": job["title"],
                            "company": job["company"],
                            "location": job["location"],
                            "distance": job.get("distance", 0),
                            "match_count": 0,
                            "queries_matched": [],
                        }
                    all_results[job_id]["match_count"] += 1
                    all_results[job_id]["queries_matched"].append(query)
                    # Use the minimum distance (best match)
                    if job.get("distance", 0) < all_results[job_id]["distance"]:
                        all_results[job_id]["distance"] = job.get("distance", 0)
            except Exception as e:
                logging.error(f"Error searching for query '{query}': {e}")
                continue

        # Sort by match count (descending) and then by distance (ascending)
        ranked_jobs = sorted(
            all_results.values(),
            key=lambda x: (-x["match_count"], x["distance"]),
        )

        # Get additional details from database
        conn = get_db_connection()
        for job in ranked_jobs:
            db_job = conn.execute(
                """
                SELECT job_description, requirements, benefits, salary
                FROM jobs WHERE id = ?
            """,
                (job["job_id"],),
            ).fetchone()

            if db_job:
                job["job_description"] = db_job["job_description"]
                job["requirements"] = db_job["requirements"]
                job["benefits"] = db_job["benefits"]
                job["salary"] = db_job["salary"]

        conn.close()

        if not ranked_jobs:
            return "No matching jobs found", None

        # Format results as dataframe
        df = pd.DataFrame(ranked_jobs)
        logging.info(
            f"Resume-based search: Found {len(ranked_jobs)} jobs ranked by relevance"
        )

        # Create summary message
        summary = f"""✅ Resume Analysis Complete!

{llm_response}

📊 Search Results:
- Found {len(ranked_jobs)} relevant jobs
- Ranked by number of matching queries and semantic similarity
- Top match: {ranked_jobs[0]['title']} at {ranked_jobs[0]['company']}"""

        return summary, df

    except Exception as e:
        logging.error(f"Error in resume-based job search: {e}")
        return f"Error: {str(e)}", None


def export_cover_letter_to_pdf(cover_letter_text, job_title, company):
    """Export cover letter to PDF file."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch

        # Create PDF file
        pdf_path = f"cover_letter_{company}_{job_title.replace(' ', '_')}.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)

        # Create styles
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="Custom",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=11,
                leading=14,
                spaceAfter=12,
            )
        )

        # Build content
        story = []
        lines = cover_letter_text.split("\n")
        for line in lines:
            if line.strip():
                story.append(Paragraph(line, styles["Custom"]))
                story.append(Spacer(1, 0.1 * inch))

        # Build PDF
        doc.build(story)

        logging.info(f"Successfully exported cover letter to {pdf_path}")
        return f"✅ Successfully exported cover letter to {pdf_path}", pdf_path
    except ImportError:
        return (
            "Error: reportlab not installed. Install with: pip install reportlab",
            None,
        )
    except Exception as e:
        logging.error(f"Error exporting to PDF: {e}")
        return f"Error exporting to PDF: {str(e)}", None


def generate_cover_letter_and_analyze(job_id):
    """
    Generate a personalized cover letter for a specific job and analyze
    pros/cons of the candidate's materials compared to the job.
    """
    try:
        # Get job details from database
        conn = get_db_connection()
        job = conn.execute(
            """
            SELECT id, title, company, location, job_description, requirements, benefits, salary, url
            FROM jobs WHERE id = ?
        """,
            (job_id,),
        ).fetchone()
        conn.close()

        if not job:
            return "Error: Job not found", None, None

        job_dict = dict(job)

        # Get user documents (resume and cover letter)
        embedder = JobEmbedder()
        user_collection = embedder.client.get_collection(name="user_documents")
        user_docs = user_collection.get()

        if not user_docs or not user_docs["documents"]:
            return (
                "Error: No resume or cover letter found. Please upload your documents first.",
                None,
                None,
            )

        # Extract resume and cover letter text
        resume_text = ""
        cover_letter_text = ""
        for i, metadata in enumerate(user_docs["metadatas"]):
            doc_type = metadata.get("document_type", "")
            if doc_type == "resume":
                resume_text = user_docs["documents"][i]
            elif doc_type == "cover_letter":
                cover_letter_text = user_docs["documents"][i]

        if not resume_text:
            return (
                "Error: No resume found. Please upload your resume first.",
                None,
                None,
            )

        # Use LLM to generate cover letter and analyze pros/cons
        try:
            from openai import OpenAI

            # Check for OpenAI API key
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
            openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

            if not openai_api_key:
                return (
                    "Error: OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.",
                    None,
                    None,
                )

            client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

            # Create prompt for LLM
            prompt = f"""You are an expert career coach and professional writer. Your task is to:

1. Generate a personalized cover letter for the candidate based on the job description
2. Analyze the pros and cons of the candidate's resume and current cover letter compared to this specific job

JOB DETAILS:
Title: {job_dict['title']}
Company: {job_dict['company']}
Location: {job_dict['location']}
Salary: {job_dict['salary'] or 'Not specified'}

JOB DESCRIPTION:
{job_dict['job_description'][:4000] if job_dict['job_description'] else 'Not provided'}

REQUIREMENTS:
{job_dict['requirements'][:3000] if job_dict['requirements'] else 'Not provided'}

BENEFITS:
{job_dict['benefits'][:2000] if job_dict['benefits'] else 'Not provided'}

CANDIDATE'S RESUME:
{resume_text[:4000]}

CANDIDATE'S CURRENT COVER LETTER:
{cover_letter_text[:3000] if cover_letter_text else 'No cover letter provided'}

Please provide your response in the following format:

=== PERSONALIZED COVER LETTER ===
[Write a compelling, personalized cover letter that highlights the candidate's relevant experience, skills, and achievements. Make it specific to this job and company. Use a professional tone and format.]

=== PROS AND CONS ANALYSIS ===

**STRENGTHS (Pros):**
[List 5-7 specific strengths of the candidate's materials in relation to this job. Focus on:
- Relevant skills and experience that match job requirements
- Notable achievements that demonstrate capability
- Strong qualifications that align with the role
- Any unique value propositions the candidate offers]

**WEAKNESSES (Cons):**
[List 3-5 areas where the candidate's materials could be improved or where there are gaps compared to this job. Focus on:
- Missing or underemphasized skills mentioned in the job
- Experience gaps that could be addressed
- Areas where the resume or cover letter could be more compelling
- Specific recommendations for improvement]

**OVERALL FIT:**
[Provide a brief assessment of how well the candidate fits this role (e.g., Excellent fit, Good fit, Moderate fit, or Poor fit) with a 1-2 sentence explanation]

**RECOMMENDATIONS:**
[Provide 3-5 specific, actionable recommendations for improving the candidate's application for this job]"""

            # Call LLM
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert career coach and professional writer with deep knowledge of job applications, resume writing, and cover letter crafting.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=3000,
            )

            llm_response = response.choices[0].message.content

            # Parse the response into sections
            sections = {}
            current_section = None
            current_content = []

            for line in llm_response.split("\n"):
                if line.strip().startswith("==="):
                    # Save previous section
                    if current_section:
                        sections[current_section] = "\n".join(current_content).strip()
                    # Start new section
                    current_section = line.strip().replace("=", "").strip()
                    current_content = []
                else:
                    if current_section:
                        current_content.append(line)

            # Save last section
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()

            # Extract cover letter
            cover_letter = sections.get("PERSONALIZED COVER LETTER", "")

            # Extract analysis
            analysis = llm_response.replace(
                sections.get("PERSONALIZED COVER LETTER", ""), ""
            ).strip()

            logging.info(f"Generated cover letter and analysis for job {job_id}")

            return (
                f"✅ Successfully generated cover letter and analysis for {job_dict['title']} at {job_dict['company']}",
                cover_letter,
                analysis,
            )

        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return f"Error generating cover letter: {str(e)}", None, None

    except Exception as e:
        logging.error(f"Error in cover letter generation: {e}")
        return f"Error: {str(e)}", None, None


# Global artifact storage
artifact_storage = {
    "artifacts": [],
    "current_index": 0,
}

# Global state for chat greeting
chat_greeting_state = {"greeting": None}


def resume_chat(message, history):
    """
    Chat interface for resume editing and cover letter generation.
    Returns response and optionally a cover letter artifact with metadata.
    """
    try:
        # Get user documents
        embedder = JobEmbedder()
        user_collection = embedder.client.get_collection(name="user_documents")
        user_docs = user_collection.get()

        if not user_docs or not user_docs["documents"]:
            return "Please upload your resume first in the Documents tab.", None

        # Extract resume and cover letter text
        resume_text = ""
        cover_letter_text = ""
        for i, metadata in enumerate(user_docs["metadatas"]):
            doc_type = metadata.get("document_type", "")
            if doc_type == "resume":
                resume_text = user_docs["documents"][i]
            elif doc_type == "cover_letter":
                cover_letter_text = user_docs["documents"][i]

        if not resume_text:
            return "No resume found. Please upload your resume first.", None

        # Use LLM to process the request
        try:
            from openai import OpenAI

            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
            openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

            if not openai_api_key:
                return (
                    "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.",
                    None,
                )

            client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

            # Build conversation history
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert career coach and resume writer. Your goal is to help the user represent themselves authentically and effectively, not to take over their voice.

## IMPORTANT: Response Format
- Always respond in **Markdown format** for better readability
- Use proper Markdown syntax: **bold**, *italic*, `code`, bullet points, numbered lists, headers (#, ##, ###)
- When generating cover letters, format them as clean Markdown with proper spacing
- Use code blocks (```) for examples or templates

## Resume Best Practices & Tips:

### Structure & Format:
- Use clear, professional formatting with consistent headings
- Start with a strong summary or objective (2-3 sentences)
- Use bullet points for experience and skills (easier to scan)
- Keep it to 1-2 pages unless you have 10+ years of experience
- Use action verbs to start bullet points (e.g., "Developed," "Led," "Created")

### Content Guidelines:
- Quantify achievements with numbers and metrics when possible
- Focus on impact and results, not just responsibilities
- Tailor content to the specific role/industry
- Include relevant keywords for ATS systems
- Be honest and authentic - don't exaggerate or misrepresent

### Experience Section:
- Use reverse chronological order
- For each role: Company, Title, Dates, Location
- 3-5 bullet points per role (focus on recent roles)
- Mix of responsibilities and achievements
- Use the STAR method: Situation, Task, Action, Result

### Skills Section:
- Separate technical and soft skills
- Group related skills together
- Include proficiency levels (e.g., "Advanced," "Intermediate")
- Focus on skills relevant to target roles

### Common Mistakes to Avoid:
- Typos and grammatical errors
- Vague or generic statements
- Including irrelevant information
- Using personal pronouns (I, me, my)
- Outdated contact information

## Cover Letter Best Practices:

### Structure:
- Header: Contact info, date, recipient info
- Salutation: Professional greeting
- Opening: State position and why interested
- Body: 2-3 paragraphs connecting experience to role
- Closing: Call to action and professional sign-off

### Content Guidelines:
- Research the company and mention specific details
- Connect your experience to the job requirements
- Show enthusiasm and genuine interest
- Keep it to 3-4 paragraphs (one page)
- Use the same voice and tone as your resume

### What to Include:
- Why you're interested in this specific role/company
- How your experience aligns with the job
- Specific examples of relevant achievements
- What you can contribute to the team
- Your unique value proposition

## Your Approach:

1. **Ask Proactive Questions**: When discussing resume/career, ask about:
   - Biggest strengths and areas for improvement
   - Key achievements they're proud of
   - Challenges they've overcome
   - Career goals and aspirations
   - What makes them unique

2. **Maintain Their Voice**: 
   - Don't rewrite in your style - help them improve their own
   - Suggest improvements but let them choose the wording
   - Ask for their input on changes
   - Respect their authentic voice and personality

3. **Provide Specific Feedback**:
   - Point out strengths to build on
   - Identify areas for improvement with specific examples
   - Suggest concrete, actionable changes
   - Explain WHY changes are recommended

4. **Generate Authentic Content**:
   - When writing cover letters, use their actual experience
   - Incorporate their achievements and skills
   - Match their communication style
   - Ask for their feedback and revisions

Remember: Your role is to guide and empower, not to take over. Help them tell their story in the most compelling way possible while staying true to who they are.""",
                }
            ]

            # Add resume context
            messages.append(
                {
                    "role": "system",
                    "content": f"USER'S RESUME:\n{resume_text[:3000]}",
                }
            )

            if cover_letter_text:
                messages.append(
                    {
                        "role": "system",
                        "content": f"USER'S CURRENT COVER LETTER:\n{cover_letter_text[:2000]}",
                    }
                )

            # Add conversation history
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})

            # Add current message
            messages.append({"role": "user", "content": message})

            # Call LLM
            response = client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )

            response_text = response.choices[0].message.content

            # Check if response contains a cover letter
            cover_letter_artifact = None
            if (
                "cover letter" in message.lower()
                or "cover letter" in response_text.lower()
            ):
                # Extract cover letter from response
                lines = response_text.split("\n")
                cover_letter_lines = []
                in_cover_letter = False

                for line in lines:
                    if "cover letter" in line.lower() or line.strip().startswith(
                        "Dear"
                    ):
                        in_cover_letter = True
                    if in_cover_letter:
                        cover_letter_lines.append(line)
                    if in_cover_letter and line.strip().endswith("Sincerely,"):
                        break

                if cover_letter_lines:
                    # Generate unique artifact ID
                    artifact_id = f"CL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

                    # Create artifact with metadata
                    cover_letter_artifact = {
                        "id": artifact_id,
                        "content": "\n".join(cover_letter_lines),
                        "timestamp": datetime.now().isoformat(),
                        "context": (
                            message[:100] + "..." if len(message) > 100 else message
                        ),
                    }

                    # Store artifact
                    artifact_storage["artifacts"].append(cover_letter_artifact)
                    artifact_storage["current_index"] = (
                        len(artifact_storage["artifacts"]) - 1
                    )

                    logging.info(
                        f"Created artifact {artifact_id} with {len(cover_letter_lines)} lines"
                    )

            return response_text, cover_letter_artifact

        except Exception as e:
            logging.error(f"Error in resume chat: {e}")
            return f"Error: {str(e)}", None

    except Exception as e:
        logging.error(f"Error in resume chat: {e}")
        return f"Error: {str(e)}", None


def navigate_artifacts(direction):
    """Navigate through stored artifacts."""
    if not artifact_storage["artifacts"]:
        return None, "No artifacts available"

    current = artifact_storage["current_index"]

    if direction == "next":
        new_index = min(current + 1, len(artifact_storage["artifacts"]) - 1)
    elif direction == "prev":
        new_index = max(current - 1, 0)
    else:
        new_index = current

    artifact_storage["current_index"] = new_index
    artifact = artifact_storage["artifacts"][new_index]

    return artifact, f"Artifact {new_index + 1} of {len(artifact_storage['artifacts'])}"


def get_current_artifact():
    """Get the current artifact."""
    if not artifact_storage["artifacts"]:
        return None, "No artifacts available"

    current = artifact_storage["current_index"]
    artifact = artifact_storage["artifacts"][current]

    return artifact, f"Artifact {current + 1} of {len(artifact_storage['artifacts'])}"


# Initialize database on module import
initialize_database()

# Create Gradio interface
with gr.Blocks(title="Job Search Manager") as demo:
    gr.Markdown("# 🔍 Job Search Manager")
    gr.Markdown(
        "Manage and monitor your job search system with provider management, database access, and semantic search."
    )

    with gr.Tabs():
        # Dashboard Tab
        with gr.Tab("📊 Dashboard"):
            with gr.Row():
                stats_btn = gr.Button("Refresh Statistics", variant="primary")

            with gr.Row():
                stats_display = gr.JSON(label="Job Statistics")

            with gr.Row():
                recent_jobs_df = gr.Dataframe(label="Recent Jobs", interactive=False)

            stats_btn.click(get_statistics, outputs=stats_display)
            stats_btn.click(get_recent_jobs, outputs=recent_jobs_df)

            # Load initial data
            demo.load(get_statistics, outputs=stats_display)
            demo.load(get_recent_jobs, outputs=recent_jobs_df)

        # Jobs Tab
        with gr.Tab("💼 Jobs"):
            with gr.Row():
                status_filter = gr.Dropdown(
                    choices=[
                        "all",
                        "pending",
                        "verified",
                        "scraped",
                        "embedded",
                        "inactive",
                        "scrape_failed",
                        "embed_failed",
                    ],
                    value="all",
                    label="Status Filter",
                )
                search_query = gr.Textbox(
                    label="Search",
                    placeholder="Search by title, company, or location...",
                )
                filter_btn = gr.Button("Filter", variant="primary")

            with gr.Row():
                jobs_df = gr.Dataframe(label="Jobs", interactive=False)

            with gr.Row():
                job_id_input = gr.Number(label="Job ID", precision=0)
                view_job_btn = gr.Button("View Details", variant="secondary")

            with gr.Row():
                job_details = gr.Markdown(label="Job Details")
                job_logs = gr.Dataframe(label="Processing Log", interactive=False)

            filter_btn.click(
                get_jobs_filtered, inputs=[status_filter, search_query], outputs=jobs_df
            )

            view_job_btn.click(
                get_job_details, inputs=job_id_input, outputs=[job_details, job_logs]
            )

            # Load initial data
            demo.load(
                get_jobs_filtered, inputs=[status_filter, search_query], outputs=jobs_df
            )

        # Providers Tab
        with gr.Tab("⚙️ Providers & Queries"):
            with gr.Row():
                gr.Markdown("### ATS Domains")

            with gr.Row():
                gr.Markdown(
                    "Select the ATS domains you want to search for jobs. Each domain has specific scraping and verification logic."
                )

            # Create checkboxes for each supported ATS domain
            with gr.Row():
                ats_checkboxes = {}
                for domain, config in SUPPORTED_ATS_DOMAINS.items():
                    checkbox = gr.Checkbox(
                        label=f"{config['name']} ({domain})",
                        value=False,
                        info=config.get("description", ""),
                    )
                    ats_checkboxes[domain] = checkbox

            with gr.Row():
                update_ats_btn = gr.Button("Update ATS Domains", variant="primary")

            with gr.Row():
                selected_domains_display = gr.Textbox(
                    label="Selected Domains", interactive=False, lines=3
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Search Queries")

            with gr.Row():
                query_input = gr.Textbox(
                    label="Add Query", placeholder="e.g., (GenAI | LLM) Engineer"
                )
                add_query_btn = gr.Button("Add Query", variant="primary")

            with gr.Row():
                queries_list = gr.Textbox(
                    label="Current Queries", interactive=False, lines=5
                )
                query_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                remove_query_input = gr.Textbox(
                    label="Remove Query", placeholder="Enter query to remove"
                )
                remove_query_btn = gr.Button("Remove Query", variant="stop")

            # Event handlers for ATS domains
            update_ats_btn.click(
                update_ats_domains,
                inputs=list(ats_checkboxes.values()),
                outputs=selected_domains_display,
            )

            # Event handlers for queries
            add_query_btn.click(
                add_query, inputs=query_input, outputs=[query_status, queries_list]
            )

            remove_query_btn.click(
                remove_query,
                inputs=remove_query_input,
                outputs=[query_status, queries_list],
            )

            # Load initial data
            def load_providers():
                ats_domains, target_queries, checkbox_states = get_providers()
                # Return checkbox states, queries, and selected domains
                checkbox_values = [
                    checkbox_states[domain] for domain in SUPPORTED_ATS_DOMAINS
                ]
                return checkbox_values + [
                    "\n".join(target_queries),
                    "\n".join(ats_domains),
                ]

            demo.load(
                load_providers,
                outputs=list(ats_checkboxes.values())
                + [queries_list, selected_domains_display],
            )

        # Configuration Tab
        with gr.Tab("🔧 Configuration"):
            with gr.Row():
                gr.Markdown("### Search Provider")

            with gr.Row():
                gr.Markdown(
                    "Select the search providers you want to use. Multiple providers can be selected simultaneously."
                )

            with gr.Row():
                search_provider_brave = gr.Checkbox(
                    label="Brave Search", value=True, info="Use Brave Search API"
                )
                search_provider_google = gr.Checkbox(
                    label="Google Search",
                    value=False,
                    info="Use Google Custom Search API",
                )

            with gr.Row():
                gr.Markdown("### Brave Search Configuration")

            with gr.Row():
                brave_rate_limit = gr.Number(
                    label="Brave API Rate Limit (seconds)", value=1.1, step=0.1
                )
                monthly_limit = gr.Number(
                    label="Monthly API Limit", value=2000, step=100
                )
                freshness = gr.Dropdown(
                    choices=["pd", "pw", "pm", "py"],
                    value="pd",
                    label="Freshness (pd=past day, pw=past week, pm=past month, py=past year)",
                )

            with gr.Row():
                gr.Markdown("### Google Search Configuration")

            with gr.Row():
                google_api_key = gr.Textbox(
                    label="Google Search API Key",
                    placeholder="Enter your Google Search API key",
                    type="password",
                )
                google_search_id = gr.Textbox(
                    label="Google Search Engine ID",
                    placeholder="Enter your Google Search Engine ID",
                )

            with gr.Row():
                google_rate_limit = gr.Number(
                    label="Google API Rate Limit (seconds)", value=1.1, step=0.1
                )

            with gr.Row():
                gr.Markdown("### Scraping Configuration")

            with gr.Row():
                scrape_rate_limit = gr.Number(
                    label="Scrape Rate Limit (seconds)", value=2.0, step=0.1
                )

            with gr.Row():
                update_config_btn = gr.Button("Update Configuration", variant="primary")
                config_status = gr.Textbox(label="Status", interactive=False)

            def update_all_config(
                search_provider_brave,
                search_provider_google,
                brave_rate_limit,
                monthly_limit,
                freshness,
                google_api_key,
                google_search_id,
                google_rate_limit,
                scrape_rate_limit,
            ):
                """Update all configuration settings."""
                config = load_env_config()

                # Determine search provider(s)
                providers = []
                if search_provider_brave:
                    providers.append("brave")
                if search_provider_google:
                    providers.append("google")

                # Set SEARCH_PROVIDER (use first provider for backward compatibility)
                config["SEARCH_PROVIDER"] = providers[0] if providers else "brave"
                config["SEARCH_PROVIDERS"] = ",".join(providers)

                config["BRAVE_RATE_LIMIT"] = str(brave_rate_limit)
                config["BRAVE_MONTHLY_LIMIT"] = str(monthly_limit)
                config["BRAVE_FRESHNESS"] = freshness
                config["GOOGLE_SEARCH_API_KEY"] = google_api_key
                config["GOOGLE_SEARCH_ID"] = google_search_id
                config["GOOGLE_SEARCH_RATE_LIMIT"] = str(google_rate_limit)
                config["SCRAPE_RATE_LIMIT"] = str(scrape_rate_limit)

                if save_env_config(config):
                    logging.info("Configuration updated")
                    return f"Success: Configuration updated (Providers: {', '.join(providers)})"
                else:
                    return "Error: Failed to save configuration"

            update_config_btn.click(
                update_all_config,
                inputs=[
                    search_provider_brave,
                    search_provider_google,
                    brave_rate_limit,
                    monthly_limit,
                    freshness,
                    google_api_key,
                    google_search_id,
                    google_rate_limit,
                    scrape_rate_limit,
                ],
                outputs=[config_status],
            )

        # Search Tab
        with gr.Tab("🔎 Semantic Search"):
            with gr.Row():
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., Remote Python developer with ML experience",
                    scale=4,
                )
                n_results = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1, label="Number of Results"
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)

            with gr.Row():
                search_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                search_results = gr.Dataframe(label="Similar Jobs", interactive=False)

            search_btn.click(
                semantic_search,
                inputs=[search_input, n_results],
                outputs=[search_status, search_results],
            )

        # Export Tab
        with gr.Tab("📥 Export"):
            with gr.Row():
                gr.Markdown("### Export Jobs Data")

            with gr.Row():
                export_btn = gr.Button("Export to CSV", variant="primary")

            with gr.Row():
                export_df = gr.Dataframe(label="Exported Jobs", interactive=False)

            export_btn.click(export_jobs, outputs=export_df)

        # Documents Tab
        with gr.Tab("📄 Documents"):
            with gr.Row():
                gr.Markdown("### Upload Your Documents")

            with gr.Row():
                gr.Markdown(
                    "Upload your resume and cover letter to enable semantic search against your qualifications. "
                    "Supported formats: **PDF, DOCX, HTML, PPTX, images, and more** (powered by Docling)"
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Resume")

            with gr.Row():
                resume_upload = gr.File(
                    label="Upload Resume",
                    file_types=[
                        ".pdf",
                        ".docx",
                        ".doc",
                        ".html",
                        ".pptx",
                        ".txt",
                        ".rst",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ],
                    file_count="single",
                )
                embed_resume_btn = gr.Button("Embed Resume", variant="primary")

            with gr.Row():
                resume_status = gr.Textbox(label="Status", interactive=False)
                resume_preview = gr.Textbox(
                    label="Resume Preview (first 500 chars)",
                    interactive=False,
                    lines=10,
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Cover Letter")

            with gr.Row():
                cover_letter_upload = gr.File(
                    label="Upload Cover Letter",
                    file_types=[
                        ".pdf",
                        ".docx",
                        ".doc",
                        ".html",
                        ".pptx",
                        ".txt",
                        ".rst",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ],
                    file_count="single",
                )
                embed_cover_letter_btn = gr.Button("Embed Cover Letter", variant="primary")

            with gr.Row():
                cover_letter_status = gr.Textbox(label="Status", interactive=False)
                cover_letter_preview = gr.Textbox(
                    label="Cover Letter Preview (first 500 chars)",
                    interactive=False,
                    lines=10,
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Search Your Documents")

            with gr.Row():
                doc_search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., What are my Python skills?",
                    scale=4,
                )
                doc_n_results = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1, label="Number of Results"
                )
                doc_search_btn = gr.Button("Search Documents", variant="primary", scale=1)

            with gr.Row():
                doc_search_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                doc_search_results = gr.Dataframe(
                    label="Document Search Results", interactive=False
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### AI-Powered Job Matching")

            with gr.Row():
                gr.Markdown(
                    "Let an AI analyze your resume and cover letter to find the most relevant jobs. "
                    "The system will generate targeted search queries and rank job listings by relevance."
                )

            with gr.Row():
                n_jobs = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=10,
                    step=5,
                    label="Number of Jobs to Return",
                )
                match_jobs_btn = gr.Button(
                    "🎯 Find Matching Jobs", variant="primary", size="lg"
                )

            with gr.Row():
                match_status = gr.Textbox(label="Status", interactive=False, lines=20)

            with gr.Row():
                match_results = gr.Dataframe(
                    label="Ranked Job Matches", interactive=False
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Cover Letter Generator & Analysis")

            with gr.Row():
                gr.Markdown(
                    "Generate a personalized cover letter for any job and get an analysis of how well your materials match the requirements."
                )

            with gr.Row():
                cover_letter_job_id = gr.Number(
                    label="Job ID",
                    precision=0,
                    placeholder="Enter job ID from Jobs tab",
                )
                generate_cover_letter_btn = gr.Button(
                    "📝 Generate Cover Letter & Analysis", variant="primary"
                )

            with gr.Row():
                cover_letter_status = gr.Textbox(
                    label="Status", interactive=False, lines=2
                )

            with gr.Row():
                generated_cover_letter = gr.Textbox(
                    label="Generated Cover Letter",
                    interactive=False,
                    lines=20,
                    placeholder="Your personalized cover letter will appear here...",
                )

            with gr.Row():
                pros_cons_analysis = gr.Textbox(
                    label="Pros & Cons Analysis",
                    interactive=False,
                    lines=20,
                    placeholder="Analysis of your materials compared to the job will appear here...",
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Tips")

            with gr.Row():
                gr.Markdown(
                    """
                - **Resume**: Upload your resume in PDF, DOCX, DOC, HTML, PPTX, TXT, RST, or image format
                - **Cover Letter**: Upload your cover letter in PDF, DOCX, DOC, HTML, PPTX, TXT, RST, or image format
                - **Embed**: Click the embed button to process and store your document in the vector database
                - **Search**: Use semantic search to find relevant sections in your documents
                - **AI Matching**: Let AI analyze your documents and find the best job matches automatically
                - **Cover Letter Generator**: Enter a Job ID to get a personalized cover letter and analysis
                - Documents are stored separately from job listings for easy reference
                - Powered by Docling for accurate text extraction from multiple formats
                - AI features require OpenAI API key (OPENAI_API_KEY)
                """
                )

            # Event handlers
            def embed_resume_and_start_chat(file):
                """Embed resume and return status to trigger chat start."""
                status, preview = embed_user_document("resume", file)
                # Get initial greeting from AI and store in global state
                greeting, _ = resume_chat("", [])
                chat_greeting_state["greeting"] = greeting
                return (
                    status,
                    preview,
                    "Resume embedded! Go to AI Resume Coach tab to start chatting.",
                )

            embed_resume_btn.click(
                embed_resume_and_start_chat,
                inputs=[resume_upload],
                outputs=[resume_status, resume_preview, resume_status],
            )

            embed_cover_letter_btn.click(
                lambda file: embed_user_document("cover_letter", file),
                inputs=[cover_letter_upload],
                outputs=[cover_letter_status, cover_letter_preview],
            )

            doc_search_btn.click(
                search_user_documents,
                inputs=[doc_search_input, doc_n_results],
                outputs=[doc_search_status, doc_search_results],
            )

            match_jobs_btn.click(
                analyze_resume_and_rank_jobs,
                inputs=[n_jobs],
                outputs=[match_status, match_results],
            )

            generate_cover_letter_btn.click(
                generate_cover_letter_and_analyze,
                inputs=[cover_letter_job_id],
                outputs=[
                    cover_letter_status,
                    generated_cover_letter,
                    pros_cons_analysis,
                ],
            )

        # AI Resume Coach Tab
        with gr.Tab("💬 AI Resume Coach"):
            with gr.Row():
                gr.Markdown("### Interactive Resume Coaching & Cover Letter Generation")

            with gr.Row():
                gr.Markdown(
                    "Chat with an AI career coach to improve your resume and generate personalized cover letters. "
                    "Cover letters generated in the chat will appear as artifacts in the panel on the right."
                )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("<center><h2>💬 Chat with AI Coach</h2></center>")

                    # Chat history display
                    chat_history = gr.Chatbot(
                        label="Conversation",
                        height=500,
                    )

                    # Chat input
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="Your Message",
                            placeholder="Type your message here...",
                            scale=4,
                            show_label=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                        start_chat_btn = gr.Button(
                            "Start Chat", variant="secondary", scale=1
                        )

                    # Hidden state for chat history
                    chat_state = gr.State([])

                with gr.Column(scale=1):
                    gr.Markdown("<center><h2>📄 Cover Letter Artifacts</h2></center>")

                    with gr.Row():
                        artifact_counter = gr.Textbox(
                            label="Artifact",
                            value="No artifacts",
                            interactive=False,
                            scale=2,
                        )

                    with gr.Row():
                        prev_btn = gr.Button("◀ Previous", variant="secondary", scale=1)
                        next_btn = gr.Button("Next ▶", variant="secondary", scale=1)

                    with gr.Row():
                        artifact_id_display = gr.Textbox(
                            label="Artifact ID",
                            interactive=False,
                            placeholder="ID will appear here...",
                        )

                    with gr.Row():
                        artifact_context = gr.Textbox(
                            label="Context",
                            interactive=False,
                            lines=2,
                            placeholder="Request context will appear here...",
                        )

                    cover_letter_artifact = gr.Markdown(
                        label="Generated Cover Letter",
                        value="Cover letters generated in the chat will appear here...",
                    )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Export Cover Letter to PDF")

            with gr.Row():
                gr.Markdown(
                    "Export the cover letter from the artifacts panel to a PDF file for easy sharing and printing."
                )

            with gr.Row():
                export_pdf_btn = gr.Button("📄 Export to PDF", variant="primary")
                pdf_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Tips")

            with gr.Row():
                gr.Markdown(
                    """
                - **Chat**: Ask questions about resume improvement, cover letter writing, and career advice
                - **Cover Letters**: Request cover letters for specific jobs or roles, and they'll appear in the artifacts panel
                - **Export**: Click "Export to PDF" to save the cover letter from the artifacts panel
                - **Context**: The AI has access to your uploaded resume and cover letter for personalized advice
                - **Conversation**: The AI remembers your conversation history for context-aware responses
                - Requires OpenAI API key (OPENAI_API_KEY) and uploaded resume
                """
                )

            # Event handlers
            def send_message(message, history):
                """Send a message and get AI response."""
                if not message or not message.strip():
                    return history, history

                # Add user message to history
                history.append((message, None))

                # Get AI response
                response, artifact = resume_chat(message, history)

                # Add AI response to history
                history[-1] = (message, response)

                # Update artifact display if there's a new artifact
                if artifact:
                    counter, artifact_id, context, content = update_artifact_display(
                        artifact
                    )
                    return history, history, counter, artifact_id, context, content

                return history, history, "No artifacts", "", "", ""

            def start_chat_on_embed():
                """Auto-start chat when resume is embedded."""
                # Get initial greeting from AI
                greeting, _ = resume_chat("", [])
                return [(None, greeting)]

            # Wire up send button and chat input
            send_btn.click(
                send_message,
                inputs=[chat_input, chat_state],
                outputs=[
                    chat_history,
                    chat_state,
                    artifact_counter,
                    artifact_id_display,
                    artifact_context,
                    cover_letter_artifact,
                ],
            )

            chat_input.submit(
                send_message,
                inputs=[chat_input, chat_state],
                outputs=[
                    chat_history,
                    chat_state,
                    artifact_counter,
                    artifact_id_display,
                    artifact_context,
                    cover_letter_artifact,
                ],
            )

            # Wire up Start Chat button to load greeting
            start_chat_btn.click(
                lambda: [
                    (
                        None,
                        chat_greeting_state.get(
                            "greeting",
                            "Hello! I'm your AI career coach. How can I help you with your resume or cover letter today?",
                        ),
                    )
                ],
                outputs=[chat_history],
            )

            def update_artifact_display(artifact_dict):
                """Update artifact display from dictionary."""
                if artifact_dict is None:
                    return "No artifacts", "", "", ""

                # Format content as Markdown for better readability
                content = artifact_dict.get("content", "")

                # Convert URLs to clickable Markdown links
                import re

                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

                def replace_url(match):
                    url = match.group(0)
                    return f"[{url}]({url})"

                markdown_content = re.sub(url_pattern, replace_url, content)

                # Preserve line breaks
                markdown_content = markdown_content.replace("\n\n", "\n\n").replace(
                    "\n", "\n"
                )

                return (
                    f"Artifact {artifact_storage['current_index'] + 1} of {len(artifact_storage['artifacts'])}",
                    artifact_dict.get("id", ""),
                    artifact_dict.get("context", ""),
                    markdown_content,
                )

            def on_chat_response(response, artifact_dict):
                """Handle chat response and update artifact display."""
                if artifact_dict:
                    return update_artifact_display(artifact_dict)
                return "No artifacts", "", "", ""

            # Navigation handlers
            prev_btn.click(
                lambda: navigate_artifacts("prev"),
                outputs=[cover_letter_artifact, artifact_counter],
            ).then(
                lambda: get_current_artifact(),
                outputs=[cover_letter_artifact, artifact_counter],
            ).then(
                lambda artifact: update_artifact_display(artifact),
                inputs=[cover_letter_artifact],
                outputs=[
                    artifact_counter,
                    artifact_id_display,
                    artifact_context,
                    cover_letter_artifact,
                ],
            )

            next_btn.click(
                lambda: navigate_artifacts("next"),
                outputs=[cover_letter_artifact, artifact_counter],
            ).then(
                lambda: get_current_artifact(),
                outputs=[cover_letter_artifact, artifact_counter],
            ).then(
                lambda artifact: update_artifact_display(artifact),
                inputs=[cover_letter_artifact],
                outputs=[
                    artifact_counter,
                    artifact_id_display,
                    artifact_context,
                    cover_letter_artifact,
                ],
            )

            def export_cover_letter_pdf(cover_letter_text):
                if not cover_letter_text or not cover_letter_text.strip():
                    return (
                        "Error: No cover letter to export. Generate a cover letter in the chat first.",
                        None,
                    )

                # Try to extract job title and company from cover letter
                lines = cover_letter_text.split("\n")
                job_title = "Position"
                company = "Company"

                for line in lines:
                    if "position" in line.lower() or "role" in line.lower():
                        job_title = line.strip()
                    if "company" in line.lower():
                        company = line.strip()

                return export_cover_letter_to_pdf(cover_letter_text, job_title, company)

            export_pdf_btn.click(
                export_cover_letter_pdf,
                inputs=[cover_letter_artifact],
                outputs=[pdf_status, gr.File(visible=False)],
            )

        # Run Commands Tab
        with gr.Tab("🚀 Run Commands"):
            with gr.Row():
                gr.Markdown("### Execute Job Search and Processing Scripts")

            with gr.Row():
                gr.Markdown(
                    "Run the job search and processing scripts directly from the interface. Output will be displayed in real-time."
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Job Search")

            with gr.Row():
                search_btn = gr.Button(
                    "🔍 Run Job Search", variant="primary", size="lg"
                )
                search_cancel_btn = gr.Button(
                    "⏹️ Stop Search", variant="stop", size="lg"
                )

            with gr.Row():
                search_output = gr.Textbox(
                    label="Job Search Output",
                    placeholder="Click 'Run Job Search' to start...",
                    lines=15,
                    interactive=False,
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Job Processing")

            with gr.Row():
                process_btn = gr.Button(
                    "⚙️ Run Job Processor", variant="primary", size="lg"
                )
                process_cancel_btn = gr.Button(
                    "⏹️ Stop Processor", variant="stop", size="lg"
                )

            with gr.Row():
                process_output = gr.Textbox(
                    label="Job Processor Output",
                    placeholder="Click 'Run Job Processor' to start...",
                    lines=15,
                    interactive=False,
                )

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### Tips")

            with gr.Row():
                gr.Markdown(
                    """
                - **Job Search**: Searches for jobs using your configured providers and queries. Results are saved to `job_results.json`.
                - **Job Processor**: Verifies, scrapes, and embeds jobs from `job_results.json`. Results are saved to `job_search.db` and `chroma_db`.
                - Make sure your providers and queries are configured in the **Providers & Queries** tab before running.
                - Check the **Dashboard** tab after processing to see updated statistics.
                """
                )

            # Event handlers
            search_btn.click(run_job_search, outputs=search_output)
            search_cancel_btn.click(cancel_job_search, outputs=search_output)

            process_btn.click(run_job_processor, outputs=process_output)
            process_cancel_btn.click(cancel_job_processor, outputs=process_output)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft()
    )
