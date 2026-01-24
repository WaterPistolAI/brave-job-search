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
from dotenv import load_dotenv
from job_processor import JobEmbedder, process_jobs_from_json
import logging
import brave_job_search
from ats_domains import SUPPORTED_ATS_DOMAINS

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

    details = f"""
    **Title:** {job_dict['title']}
    **Company:** {job_dict['company'] or 'N/A'}
    **Location:** {job_dict['location'] or 'N/A'}
    **Status:** {job_dict['status']}
    **URL:** {job_dict['url']}
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
            yield output
        yield f"\n✅ Job search completed successfully using {', '.join(search_providers).upper()}!"
    except Exception as e:
        yield f"❌ Error running job search: {str(e)}"


def run_job_processor():
    """Run the job processor function directly."""
    try:
        # Capture stdout to display in the UI
        import io
        from contextlib import redirect_stdout, redirect_stderr

        output_buffer = io.StringIO()

        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            process_jobs_from_json()

        output = output_buffer.getvalue()
        if output:
            yield output
        yield "\n✅ Job processing completed successfully!"
    except Exception as e:
        yield f"❌ Error running job processor: {str(e)}"


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

            process_btn.click(run_job_processor, outputs=process_output)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft()
    )
