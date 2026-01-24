# 🚀 Job Search Manager - Comprehensive Job Search System

A sophisticated job search system that combines multiple search engines, intelligent job processing, and semantic search capabilities to help you find and manage high-quality job opportunities. This system searches across specific Applicant Tracking Systems (ATS) for high-intent job roles like **Founding Engineer**, **GenAI Developer**, and **DevRel**.

## ✨ Key Features

### Search Capabilities
* **Multi-Provider Search:** Supports Brave Search, Google Custom Search, and other providers
* **ATS Targeting:** Searches directly on `greenhouse.io`, `lever.co`, `smartrecruiters.com`, and more
* **Rate Limit Protection:** Respects API rate limits to avoid being blocked
* **Usage Tracking:** Maintains local usage tracking to stay within API limits
* **Freshness Filter:** Automatically filters for recently posted roles
* **Advanced Boolean Logic:** Uses complex search operators to maximize result quality

### Job Processing Pipeline
* **Job Verification:** Checks if job listings are still active
* **Intelligent Scraping:** Extracts detailed job information using domain-specific adapters
* **Quality Control:** Handles various ATS platforms with specialized scraping methods
* **Rate Limiting:** Respects job board rate limits to avoid blocking

### Embedding & Search
* **Multiple Embedding Providers:** Supports local embeddings (sentence-transformers) and OpenAI-compatible endpoints
* **Semantic Search:** Find jobs similar to your preferences using vector embeddings
* **Vector Storage:** Uses ChromaDB for efficient similarity search
* **Flexible Models:** Supports various embedding models with configurable dimensions

### Web Interface
* **Interactive Dashboard:** Real-time statistics and job monitoring
* **Job Management:** Browse, filter, and view detailed job information
* **Configuration Management:** Easy provider and query management
* **Semantic Search UI:** Visual interface for similarity-based job search
* **Direct Script Execution:** Run search and processing scripts from the UI
* **Export Capabilities:** Export job data to CSV for offline analysis

---

## 🛠️ Setup

### 1. Prerequisites

* Python 3.8+
* A Brave Search API Key ([Get it here](https://api-dashboard.search.brave.com/app/dashboard)) - *Free tier available*
* Google Custom Search API Key (optional) - *For Google search support*
* OpenAI API Key (optional) - *For premium embeddings*

### 2. Installation

```bash
# Clone this repository
git clone https://github.com/WaterPistolAI/brave-job-search.git
cd brave-job-search

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the `.env.example` file to create your `.env` file. **Note:** Complex lists are wrapped in single quotes for `ast.literal_eval` parsing.

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys and configuration:

```env
# Required: Brave Search API Key (get from https://api-dashboard.search.brave.com)
BRAVE_API_KEY=your_brave_api_key_here

# Search Configuration
BRAVE_MONTHLY_LIMIT=2000
BRAVE_RATE_LIMIT=1.1
BRAVE_FRESHNESS=pd  # pd=past day, pw=past week, pm=past month, py=past year

# ATS Domains to target
BRAVE_ATS_DOMAINS='["boards.greenhouse.io", "jobs.lever.co", "jobs.smartrecruiters.com", "jobs.workable.com", "wd1.myworkdayjobs.com", "jobs.jobvite.com", "careers.icims.com"]'

# Search Queries (Boolean logic supported)
BRAVE_TARGET_QUERIES='["(GenAI | Generative AI | LLM) (Engineer | Developer)", "(Developer Advocacy | Developer Relations | DevRel | Developer Advocate)", "(Data Scientist | Data Science | Machine Learning)"]'

# Database Configuration
DB_PATH=job_search.db
CHROMA_PATH=./chroma_db

# Scraping Configuration
SCRAPE_RATE_LIMIT=2.0  # Seconds between requests
REQUEST_TIMEOUT=30     # Request timeout in seconds

# Embedding Configuration (default: local embeddings)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional: OpenAI Embeddings (uncomment to use)
# EMBEDDING_PROVIDER=openai
# OPENAI_EMBEDDING_API_KEY=your_openai_api_key
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 🚀 Usage

### Method 1: Web Interface (Recommended)

Start the interactive web interface for easy management:

```bash
python web_interface.py
```

The interface will be available at `http://localhost:7860`. The web interface provides tabs for:
* Dashboard: Real-time statistics and recent jobs
* Jobs Management: Browse and filter job listings
* Providers & Queries: Manage ATS domains and search queries
* Configuration: Adjust system settings
* Semantic Search: Find similar jobs using vector embeddings
* Export: Export job data to CSV
* Run Commands: Execute search and processing scripts

### Method 2: Command Line

Run the search and processing separately:

```bash
# Step 1: Search for jobs
python brave-job-search.py

# Step 2: Process and verify jobs
python job_processor.py
```

### How it works

1. **Search Phase:** `brave-job-search.py` queries job boards using configured providers and queries, saving results to `job_results.json`
2. **Processing Pipeline:** `job_processor.py` loads jobs and processes them through multiple stages:
   - **Verification:** Checks if each job is still active
   - **Scraping:** Extracts detailed job information using domain-specific adapters
   - **Embedding:** Creates vector embeddings of job descriptions
   - **Storage:** Saves data to SQLite database and ChromaDB vector store
3. **Search & Analysis:** Use the web interface or programmatic API to search and analyze jobs

---

## 📊 Search Logic & Configuration

### Search Operators

The system uses advanced search operators to maximize result quality:

| Operator | Usage | Benefit |
| --- | --- | --- |
| `site:` | `site:jobs.lever.co` | Restricts search to specific job board platforms |
| ` | ` (OR) | `"GenAI" | "LLM"` | Matches multiple terms with boolean logic |
| `freshness` | `pd` (past day), `pw` (past week) | Filters by posting date |
| `()` | `("GenAI" | "LLM") Engineer` | Groups boolean expressions |

### Supported ATS Platforms

The system includes specialized adapters for:
* Greenhouse (`boards.greenhouse.io`)
* Lever (`jobs.lever.co`)
* SmartRecruiters (`jobs.smartrecruiters.com`)
* Workable (`jobs.workable.com`)
* Workday (`wd1.myworkdayjobs.com`)
* Jobvite (`jobs.jobvite.com`)
* iCIMS (`careers.icims.com`)
* Generic fallback for other platforms

---

## 🔧 Advanced Features

### Embedding Providers

**Local Embeddings (Default):**
* Free, private, fast processing
* Models: `all-MiniLM-L6-v2` (384d), `all-mpnet-base-v2` (768d), and more
* No API costs or privacy concerns

**OpenAI Embeddings:**
* Higher quality embeddings
* Models: `text-embedding-3-small` (1536d), `text-embedding-3-large` (3072d)
* Cost per 1K tokens (about $0.02/1K tokens for small model)

**Custom OpenAI-Compatible Endpoints:**
* Support for Ollama, vLLM, and other local LLM servers
* Configurable embedding dimensions
* Privacy with local processing

### Semantic Search

Find jobs similar to your preferences using natural language queries:
```python
from job_processor import JobEmbedder

embedder = JobEmbedder()
results = embedder.search_similar_jobs("Remote Python developer with machine learning experience", n_results=5)
```

---

## ⚠️ Important Limitations & Best Practices

* **API Quotas:** Brave Search Free tier allows 2,000 requests/month. Configure rate limits appropriately.
* **Rate Limits:** Don't modify rate limits without understanding the consequences - you may get blocked.
* **Scraping Ethics:** Respect robots.txt and rate limits of job boards.
* **Data Freshness:** Regularly re-run the processing pipeline to maintain current job data.
* **Embedding Switching:** When changing embedding providers/models, delete `./chroma_db` and re-process jobs.

---

## 📁 Project Structure

```
brave-job-search/
├── brave_job_search.py      # Main search functionality
├── job_processor.py        # Job verification, scraping, and embedding
├── web_interface.py        # Interactive web dashboard
├── embedding_adapters.py   # Multiple embedding provider support
├── scraping_adapters.py    # Domain-specific job scraping
├── job_search.db           # SQLite database for job data
├── chroma_db/              # Vector store for semantic search
├── job_results.json        # Temporary search results storage
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests. Areas of interest include:
* Additional ATS platform adapters
* New search provider integrations
* Improved scraping techniques
* Enhanced web interface features
* Better error handling and recovery

---

## 📝 License

This project is licensed under the GNU General Public License v3.0. See [LICENSE.md](LICENSE.md) for details.
