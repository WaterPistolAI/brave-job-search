import sqlite3
import httpx
import time
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
from scraping_adapters import adapter_registry
from embedding_adapters import get_embedding_adapter
from ats_domains import get_expired_indicators, get_domain_config
from pathos.multiprocessing import ProcessingPool as Pool
from urllib.parse import urlparse
import json
from ratelimit import limits, sleep_and_retry
from queue import Queue

load_dotenv()

# --- CONFIGURATION ---
DB_PATH = os.environ.get("DB_PATH", "job_search.db")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Check for OPENAI_EMBEDDING_* first, then fall back to OPENAI_*
OPENAI_API_KEY = os.environ.get("OPENAI_EMBEDDING_API_KEY") or os.environ.get(
    "OPENAI_API_KEY", ""
)
OPENAI_BASE_URL = os.environ.get("OPENAI_EMBEDDING_BASE_URL") or os.environ.get(
    "OPENAI_BASE_URL", ""
)
OPENAI_EMBEDDING_DIMENSION = os.environ.get("OPENAI_EMBEDDING_DIMENSION")

SCRAPE_RATE_LIMIT = float(
    os.environ.get("SCRAPE_RATE_LIMIT", "2.0")
)  # seconds between requests
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))
GLOBAL_RATE_LIMIT = int(
    os.environ.get("GLOBAL_RATE_LIMIT", "5")
)  # requests per minute per domain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="job_processor.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class JobDatabase:
    """Handles SQLite database operations for job listings."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        """Create database tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

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

        self.conn.commit()
        logging.info("Database initialized successfully")

    def insert_job(self, job_data: Dict) -> int:
        """Insert a new job into the database."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT OR IGNORE INTO jobs (url, title, snippet, source_adapter)
                VALUES (?, ?, ?, ?)
            """,
                (
                    job_data["url"],
                    job_data["title"],
                    job_data["snippet"],
                    job_data.get("source_adapter", "unknown"),
                ),
            )
            self.conn.commit()
            job_id = cursor.lastrowid
            if job_id:
                logging.info(
                    f"Inserted job: {job_data['title']} from {job_data.get('source_adapter', 'unknown')} (ID: {job_id})"
                )
            return job_id
        except sqlite3.Error as e:
            logging.error(f"Error inserting job: {e}")
            return None

    def update_job_status(self, job_id: int, status: str, message: str = None):
        """Update job status and log the action."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE jobs 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (status, job_id),
            )

            # Log the action
            cursor.execute(
                """
                INSERT INTO processing_log (job_id, action, status, message)
                VALUES (?, 'status_update', ?, ?)
            """,
                (job_id, status, message),
            )

            self.conn.commit()
            logging.info(f"Updated job {job_id} status to {status}")
        except sqlite3.Error as e:
            logging.error(f"Error updating job status: {e}")

    def update_job_details(self, job_id: int, details: Dict):
        """Update job details after scraping."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE jobs 
                SET job_description = ?,
                    requirements = ?,
                    benefits = ?,
                    location = ?,
                    company = ?,
                    salary = ?,
                    scraped_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (
                    details.get("job_description"),
                    details.get("requirements"),
                    details.get("benefits"),
                    details.get("location"),
                    details.get("company"),
                    details.get("salary"),
                    job_id,
                ),
            )
            self.conn.commit()
            logging.info(f"Updated job {job_id} details")
        except sqlite3.Error as e:
            logging.error(f"Error updating job details: {e}")

    def get_pending_jobs(self, limit: int = None) -> List[Dict]:
        """Get jobs that need to be verified."""
        cursor = self.conn.cursor()
        query = "SELECT id, url, title, snippet FROM jobs WHERE status = 'pending'"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        jobs = []
        for row in cursor.fetchall():
            jobs.append(
                {"id": row[0], "url": row[1], "title": row[2], "snippet": row[3]}
            )
        return jobs

    def get_verified_jobs(self, limit: int = None) -> List[Dict]:
        """Get verified jobs that need to be scraped."""
        cursor = self.conn.cursor()
        query = "SELECT id, url, title FROM jobs WHERE status = 'verified'"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        jobs = []
        for row in cursor.fetchall():
            jobs.append({"id": row[0], "url": row[1], "title": row[2]})
        return jobs

    def get_scraped_jobs(self, limit: int = None) -> List[Dict]:
        """Get scraped jobs that need to be embedded."""
        cursor = self.conn.cursor()
        query = """
            SELECT id, url, title, job_description, company, location 
            FROM jobs 
            WHERE status = 'scraped' AND job_description IS NOT NULL
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        jobs = []
        for row in cursor.fetchall():
            jobs.append(
                {
                    "id": row[0],
                    "url": row[1],
                    "title": row[2],
                    "job_description": row[3],
                    "company": row[4],
                    "location": row[5],
                }
            )
        return jobs

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class RateLimiter:
    """Rate limiter for per-domain request limiting using ratelimit package."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        # Calculate the minimum period between requests in seconds
        self.min_period = 60 / requests_per_minute

    @sleep_and_retry
    @limits(calls=1, period=int(60 / 5))  # Default: 5 requests per minute
    def make_request(self, url: str) -> Tuple[int, str]:
        """
        Make an HTTP request with rate limiting.
        Returns (status_code, html_content).
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = httpx.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT, follow_redirects=True
        )

        return response.status_code, response.text


def verify_job_with_rate_limit(
    job: Dict, rate_limiter: RateLimiter
) -> Tuple[bool, str, Optional[str]]:
    """
    Verify if a job listing is still active with rate limiting.
    Returns (is_active, message, html_content).
    """
    try:
        # Make rate-limited request
        status_code, html_content = rate_limiter.make_request(job["url"])

        if status_code == 404:
            return False, "Page not found (404)", None
        elif status_code == 410:
            return False, "Page gone (410)", None
        elif status_code >= 400:
            return False, f"HTTP {status_code}", None

        # Parse HTML content
        soup = BeautifulSoup(html_content, "html.parser")
        page_text = soup.get_text().lower()

        # Get domain-specific expired indicators
        parsed_url = urlparse(job["url"])
        domain = parsed_url.netloc.lower()

        # Try to get domain-specific indicators
        expired_indicators = get_expired_indicators(domain)

        # Check for domain-specific CSS selectors first
        domain_config = get_domain_config(domain)
        if domain_config and "expired_selectors" in domain_config:
            for selector_config in domain_config["expired_selectors"]:
                selector = selector_config["selector"]
                case_sensitive = selector_config.get("case_sensitive", False)
                pattern_type = selector_config.get("pattern_type", "exact")

                try:
                    element = soup.select_one(selector)
                    if element:
                        element_text = element.get_text()
                        if not case_sensitive:
                            element_text = element_text.lower()

                        # Handle different pattern types
                        if pattern_type == "contains":
                            # Pattern matching (e.g., "current openings at *")
                            text_pattern = selector_config.get("text_pattern", "")
                            if not case_sensitive:
                                text_pattern = text_pattern.lower()

                            if text_pattern in element_text:
                                return (
                                    False,
                                    f"Job appears closed: Pattern '{text_pattern}' found in element '{selector}'",
                                    None,
                                )
                        else:
                            # Exact text matching (default)
                            expected_text = selector_config.get("text", "")
                            if not case_sensitive:
                                expected_text = expected_text.lower()

                            if expected_text in element_text:
                                return (
                                    False,
                                    f"Job appears closed: '{expected_text}' found in element '{selector}'",
                                    None,
                                )
                except Exception as e:
                    logging.warning(f"Error checking selector {selector}: {e}")
                    continue

        # If no domain-specific indicators found, use common ones
        if not expired_indicators:
            expired_indicators = [
                "position is closed",
                "no longer accepting applications",
                "position filled",
                "this position has been filled",
                "we are no longer accepting",
                "application closed",
                "position no longer available",
                "this role is no longer available",
                "applications are closed",
                "hiring is complete",
                "position has been filled",
                "role is closed",
                "this job is no longer available",
                "this posting has expired",
                "this position is no longer open",
            ]

        # Check for closed indicators in page text
        for indicator in expired_indicators:
            if indicator in page_text:
                return False, f"Job appears closed: '{indicator}'", None

        # Check for common job posting elements
        job_elements = [
            "apply",
            "application",
            "submit",
            "job description",
            "requirements",
            "qualifications",
            "responsibilities",
        ]
        has_job_elements = any(elem in page_text for elem in job_elements)

        if not has_job_elements:
            return False, "No job posting elements found", None

        # Job is active, return HTML content for scraping
        return True, "Job appears active", html_content

    except httpx.TimeoutException:
        return False, "Request timeout", None
    except Exception as e:
        return False, f"Error: {str(e)}", None


def convert_html_to_markdown(html_content: str) -> str:
    """Convert HTML content to markdown using markitdown."""
    try:
        from markitdown import MarkItDown
        import tempfile
        import os

        md = MarkItDown()

        # Save HTML content to a temporary file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html_content)
            temp_file_path = f.name

        try:
            # Convert the temporary file
            result = md.convert(temp_file_path)
            return result.text_content
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except ImportError:
        logging.warning("markitdown not available, using BeautifulSoup text extraction")
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logging.error(f"Error converting HTML to markdown: {e}")
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n", strip=True)


def scrape_job_with_markdown(url: str, html_content: str) -> Dict:
    """
    Scrape job details from HTML content using domain-specific adapters.
    Returns a dictionary with job details.
    """
    details = {
        "job_description": "",
        "requirements": "",
        "benefits": "",
        "location": "",
        "company": "",
        "salary": "",
    }

    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Get the appropriate adapter for this URL
        adapter = adapter_registry.get_adapter(url)
        logging.info(f"Using adapter '{adapter.__class__.__name__}' for {url}")

        # Use the adapter to scrape the job details
        # The adapter already extracts clean text from specific divs
        details = adapter.scrape(soup, url)

        logging.info(f"Successfully scraped job from {url}")
        return details

    except Exception as e:
        logging.error(f"Error scraping job from {url}: {e}")
        return details


def process_domain_jobs(
    domain: str, jobs: List[Dict], rate_limiter: RateLimiter, result_queue: Queue
):
    """
    Process jobs for a specific domain with rate limiting.
    Results are put into the result queue.
    """
    logging.info(f"Processing {len(jobs)} jobs for domain: {domain}")

    results = []
    for job in jobs:
        try:
            # Verify the job
            is_active, message, html_content = verify_job_with_rate_limit(
                job, rate_limiter
            )

            result = {
                "id": job["id"],
                "url": job["url"],
                "title": job["title"],
                "is_active": is_active,
                "message": message,
                "html_content": html_content,
            }

            if is_active and html_content:
                # Scrape the job details
                details = scrape_job_with_markdown(job["url"], html_content)
                result["details"] = details

            results.append(result)
            logging.info(f"Processed job {job['id']}: {job['title']} - {message}")

        except Exception as e:
            logging.error(f"Error processing job {job['id']}: {e}")
            results.append(
                {
                    "id": job["id"],
                    "url": job["url"],
                    "title": job["title"],
                    "is_active": False,
                    "message": f"Error: {str(e)}",
                    "html_content": None,
                }
            )

    result_queue.put((domain, results))


def process_jobs_multiprocess(
    jobs: List[Dict], rate_limit: int = GLOBAL_RATE_LIMIT
) -> List[Dict]:
    """
    Process jobs using ThreadPool with per-domain rate limiting.
    """
    # Group jobs by domain
    domain_jobs = {}
    for job in jobs:
        parsed_url = urlparse(job["url"])
        domain = parsed_url.netloc.lower()
        if domain not in domain_jobs:
            domain_jobs[domain] = []
        domain_jobs[domain].append(job)

    logging.info(f"Processing {len(jobs)} jobs across {len(domain_jobs)} domains")

    # Create a queue for results
    result_queue = Queue()

    # Create a thread for each domain using ThreadPool
    with Pool(processes=len(domain_jobs)) as pool:
        # Submit tasks for each domain
        pool.starmap(
            process_domain_jobs,
            [
                (domain, domain_job_list, RateLimiter(rate_limit), result_queue)
                for domain, domain_job_list in domain_jobs.items()
            ],
        )

    # Collect results from queue
    all_results = []
    while not result_queue.empty():
        domain, results = result_queue.get()
        all_results.extend(results)
        logging.info(f"Collected {len(results)} results from domain: {domain}")

    return all_results


class JobEmbedder:
    """Handles embedding job descriptions and storing in ChromaDB."""

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        provider: str = EMBEDDING_PROVIDER,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.chroma_path = chroma_path
        self.provider = provider
        self.model_name = model_name
        self.client = None
        self.collection = None
        self.embedding_adapter = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and embedding adapter."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.chroma_path)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="job_descriptions", metadata={"hnsw:space": "cosine"}
            )

            # Initialize embedding adapter
            adapter_kwargs = {"model_name": self.model_name}

            if self.provider.lower() == "openai":
                if OPENAI_API_KEY:
                    adapter_kwargs["api_key"] = OPENAI_API_KEY
                if OPENAI_BASE_URL:
                    adapter_kwargs["base_url"] = OPENAI_BASE_URL
                if OPENAI_EMBEDDING_DIMENSION:
                    adapter_kwargs["dimension"] = int(OPENAI_EMBEDDING_DIMENSION)

            self.embedding_adapter = get_embedding_adapter(
                provider=self.provider, **adapter_kwargs
            )

            logging.info(
                f"ChromaDB and {self.provider} embedding adapter initialized successfully"
            )
        except Exception as e:
            logging.error(f"Error initializing embedder: {e}")
            raise

    def embed_job(self, job_id: int, job_data: Dict):
        """
        Embed a job description and store in ChromaDB.
        """
        try:
            # Prepare text for embedding
            text_to_embed = f"{job_data['title']}\n{job_data.get('company', '')}\n{job_data.get('location', '')}\n{job_data['job_description']}"

            # Generate embedding using adapter
            embedding = self.embedding_adapter.embed([text_to_embed])[0]

            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[text_to_embed],
                metadatas=[
                    {
                        "job_id": job_id,
                        "url": job_data["url"],
                        "title": job_data["title"],
                        "company": job_data.get("company", ""),
                        "location": job_data.get("location", ""),
                        "scraped_at": datetime.now().isoformat(),
                    }
                ],
                ids=[f"job_{job_id}"],
            )

            logging.info(f"Embedded job {job_id}: {job_data['title']}")
            return True

        except Exception as e:
            logging.error(f"Error embedding job {job_id}: {e}")
            return False

    def search_similar_jobs(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for similar jobs using a text query.
        """
        try:
            # Generate query embedding using adapter
            query_embedding = self.embedding_adapter.embed([query])[0]

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )

            # Format results
            jobs = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    jobs.append(
                        {
                            "job_id": results["metadatas"][0][i]["job_id"],
                            "url": results["metadatas"][0][i]["url"],
                            "title": results["metadatas"][0][i]["title"],
                            "company": results["metadatas"][0][i]["company"],
                            "location": results["metadatas"][0][i]["location"],
                            "distance": (
                                results["distances"][0][i]
                                if "distances" in results
                                else None
                            ),
                        }
                    )

            return jobs

        except Exception as e:
            logging.error(f"Error searching similar jobs: {e}")
            return []


def process_jobs_from_json(json_file: str = "job_results.json"):
    """
    Main function to process jobs from JSON file:
    1. Load jobs from JSON
    2. Insert into database
    3. Verify each job (with multiprocessing and per-domain rate limiting)
    4. Scrape verified jobs (convert to markdown)
    5. Embed scraped jobs
    """
    # Initialize components
    db = JobDatabase()
    embedder = JobEmbedder()

    try:
        # Load jobs from JSON
        logging.info(f"Loading jobs from {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            jobs = json.load(f)

        logging.info(f"Found {len(jobs)} jobs in JSON file")

        # Step 1: Insert jobs into database
        logging.info("Step 1: Inserting jobs into database")
        for job in jobs:
            db.insert_job(job)

        # Step 2: Get pending jobs and verify with multiprocessing
        logging.info("Step 2: Verifying jobs with multiprocessing")
        pending_jobs = db.get_pending_jobs()
        logging.info(f"Found {len(pending_jobs)} jobs to verify")

        # Initialize verification_results to empty list
        verification_results = []

        if pending_jobs:
            # Process jobs with multiprocessing and per-domain rate limiting
            verification_results = process_jobs_multiprocess(
                pending_jobs, GLOBAL_RATE_LIMIT
            )

            # Update database with verification results
            for result in verification_results:
                if result["is_active"]:
                    db.update_job_status(result["id"], "verified", result["message"])
                    logging.info(f"Job {result['id']} verified: {result['title']}")

                    # Update job details if available
                    if "details" in result and result["details"]:
                        db.update_job_details(result["id"], result["details"])
                        db.update_job_status(
                            result["id"],
                            "scraped",
                            "Successfully scraped and converted to markdown",
                        )
                        logging.info(f"Scraped job {result['id']}: {result['title']}")
                else:
                    db.update_job_status(result["id"], "inactive", result["message"])
                    logging.warning(
                        f"Job {result['id']} inactive: {result['title']} - {result['message']}"
                    )
        else:
            logging.info("No pending jobs to verify")

        # Step 3: Embed scraped jobs
        logging.info("Step 3: Embedding scraped jobs")
        scraped_jobs = db.get_scraped_jobs()
        logging.info(f"Found {len(scraped_jobs)} jobs to embed")

        for job in scraped_jobs:
            success = embedder.embed_job(job["id"], job)
            if success:
                db.update_job_status(job["id"], "embedded", "Successfully embedded")
                logging.info(f"Embedded job {job['id']}: {job['title']}")
            else:
                db.update_job_status(job["id"], "embed_failed", "Embedding failed")
                logging.error(f"Failed to embed job {job['id']}: {job['title']}")

        # Summary
        logging.info("=" * 50)
        logging.info("PROCESSING COMPLETE")
        logging.info(f"Total jobs processed: {len(jobs)}")
        logging.info(
            f"Verified: {len([j for j in verification_results if j['is_active']])}"
        )
        logging.info(f"Scraped: {len(scraped_jobs)}")
        logging.info("=" * 50)

    except Exception as e:
        logging.error(f"Error in process_jobs_from_json: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    process_jobs_from_json()
