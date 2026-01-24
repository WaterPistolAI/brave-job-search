# Job Search Manager - Web Interface

A Gradio-based web interface for managing and monitoring your job search system. Provides an intuitive dashboard for provider management, database access, and semantic search.

## Features

- **📊 Dashboard**: Real-time statistics and recent job listings
- **💼 Jobs Management**: Browse, filter, and view detailed job information
- **⚙️ Provider Management**: Add/remove ATS domains and search queries
- **🔧 Configuration**: Adjust rate limits and API settings
- **🔎 Semantic Search**: Find similar jobs using vector embeddings
- **📥 Export**: Export job data to CSV
- **🚀 Run Commands**: Execute job search and processing scripts directly from the UI

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Interface

Start the web interface:

```bash
python web_interface.py
```

The interface will be available at `http://localhost:7860`

## Usage Guide

### Dashboard Tab

The Dashboard provides an overview of your job search system:

- **Statistics**: View total jobs, pending, verified, scraped, embedded, inactive, and failed jobs
- **Recent Jobs**: See the 20 most recently added jobs
- **Refresh**: Click "Refresh Statistics" to update the dashboard

### Jobs Tab

Browse and manage your job listings:

1. **Filter Jobs**:
   - Use the status dropdown to filter by job status (pending, verified, scraped, embedded, etc.)
   - Use the search box to find jobs by title, company, or location
   - Click "Filter" to apply filters

2. **View Job Details**:
   - Enter a Job ID from the jobs table
   - Click "View Details" to see full job information
   - View the processing log to track the job's journey through the pipeline

### Providers & Queries Tab

Manage your ATS domains and search queries:

#### ATS Domains

1. **Add a Domain**:
   - Enter the domain (e.g., `boards.greenhouse.io`)
   - Click "Add Domain"
   - The domain will be added to your `.env` file

2. **Remove a Domain**:
   - Enter the domain to remove
   - Click "Remove Domain"
   - The domain will be removed from your `.env` file

#### Search Queries

1. **Add a Query**:
   - Enter your search query (e.g., `(GenAI | LLM) Engineer`)
   - Click "Add Query"
   - The query will be added to your `.env` file

2. **Remove a Query**:
   - Enter the query to remove
   - Click "Remove Query"
   - The query will be removed from your `.env` file

**Note**: Changes to providers and queries are saved to your `.env` file and will be used by `brave-job-search.py` on the next run.

### Configuration Tab

Adjust system settings:

- **Brave API Rate Limit**: Seconds between Brave API requests (default: 1.1)
- **Scrape Rate Limit**: Seconds between scraping requests (default: 2.0)
- **Monthly API Limit**: Maximum API calls per month (default: 2000)
- **Freshness**: Time filter for search results:
  - `pd`: Past day
  - `pw`: Past week
  - `pm`: Past month
  - `py`: Past year

Click "Update Configuration" to save changes to your `.env` file.

### Semantic Search Tab

Find jobs similar to your search query using vector embeddings:

1. **Enter Search Query**:
   - Type a natural language description (e.g., "Remote Python developer with ML experience")
   - Adjust the number of results (1-20)
   - Click "Search"

2. **View Results**:
   - Results are ranked by similarity to your query
   - Each result shows job ID, title, company, location, URL, and distance score
   - Lower distance scores indicate higher similarity

**Note**: Semantic search only works on jobs that have been embedded (status = "embedded").

### Export Tab

Export your job data:

1. Click "Export to CSV"
2. The exported data includes:
   - Job ID, URL, title, snippet
   - Status, job description, requirements, benefits
   - Location, company, salary
   - Created and updated timestamps

### Run Commands Tab

Execute the job search and processing scripts directly from the interface:

#### Job Search

1. Click "🔍 Run Job Search"
2. Watch the real-time output in the text box
3. The script will:
   - Search for jobs using your configured providers and queries
   - Save results to `job_results.json`
   - Display progress and any errors
4. When complete, you'll see a success message

#### Job Processing

1. Click "⚙️ Run Job Processor"
2. Watch the real-time output in the text box
3. The script will:
   - Load jobs from `job_results.json`
   - Verify each job is still active
   - Scrape detailed job information
   - Embed job descriptions in the vector store
   - Save all data to `job_search.db` and `chroma_db`
4. When complete, you'll see a success message

**Tips**:
- Make sure your providers and queries are configured in the **Providers & Queries** tab before running
- Check the **Dashboard** tab after processing to see updated statistics
- Output is displayed in real-time, so you can monitor progress
- Use the copy button to save output for troubleshooting

## Database Schema

The interface connects to your SQLite database (`job_search.db`) with the following tables:

### Jobs Table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| url | TEXT | Job listing URL (unique) |
| title | TEXT | Job title |
| snippet | TEXT | Search result snippet |
| status | TEXT | Processing status |
| job_description | TEXT | Full job description |
| requirements | TEXT | Job requirements |
| benefits | TEXT | Job benefits |
| location | TEXT | Job location |
| company | TEXT | Company name |
| salary | TEXT | Salary information |
| created_at | TIMESTAMP | Record creation time |
| updated_at | TIMESTAMP | Last update time |

### Processing Log Table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| job_id | INTEGER | Foreign key to jobs table |
| action | TEXT | Action performed |
| status | TEXT | Status of action |
| message | TEXT | Additional message |
| timestamp | TIMESTAMP | When action occurred |

## Job Status Values

Jobs can have the following statuses:

- **pending**: Job loaded, awaiting verification
- **verified**: Job is active, awaiting scraping
- **inactive**: Job is no longer active
- **scraped**: Job details extracted, awaiting embedding
- **scrape_failed**: Scraping failed
- **embedded**: Job description embedded in vector store
- **embed_failed**: Embedding failed

## Troubleshooting

### Interface won't start

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that port 7860 is not already in use
- Verify your `.env` file exists and is properly formatted

### Can't see any jobs

- Run `brave-job-search.py` first to generate `job_results.json`
- Run `python job_processor.py` to process the results
- Check the database file exists: `job_search.db`

### Semantic search returns no results

- Ensure jobs have been processed and embedded (status = "embedded")
- Check that ChromaDB vector store exists in `./chroma_db`
- Try a more general search query

### Provider/Query changes not saving

- Check that you have write permissions for the `.env` file
- Verify the `.env` file is not locked by another process
- Check the `web_interface.log` for error messages

### Database connection errors

- Ensure `job_search.db` exists in the current directory
- Check that the database file is not corrupted
- Verify the DB_PATH environment variable is set correctly

## Advanced Usage

### Custom Port

To run on a different port:

```python
# In web_interface.py, change the last line to:
demo.launch(server_name="0.0.0.0", server_port=8080, share=False)
```

### Public Access

To share the interface publicly:

```python
# In web_interface.py, change the last line to:
demo.launch(share=True)
```

This will generate a public URL that you can share with others.

### Custom Theme

Gradio supports multiple themes. To change the theme:

```python
# In web_interface.py, change the theme parameter:
with gr.Blocks(title="Job Search Manager", theme=gr.themes.Glass()) as demo:
```

Available themes:
- `gr.themes.Soft()` (default)
- `gr.themes.Glass()`
- `gr.themes.Monochrome()`
- `gr.themes.Base()`

## Integration with Other Scripts

The web interface works seamlessly with the other scripts in this project:

1. **brave-job-search.py**: Generates `job_results.json` with job listings
2. **job_processor.py**: Processes jobs, verifies, scrapes, and embeds them
3. **web_interface.py**: Provides a UI to manage and monitor everything

### Typical Workflow

**Option 1: Using the Web Interface (Recommended)**

```bash
# Start the web interface
python web_interface.py

# Open http://localhost:7860 in your browser

# Use the "Run Commands" tab to:
# 1. Run Job Search
# 2. Run Job Processor
# 3. Monitor real-time output
```

**Option 2: Command Line**

```bash
# Step 1: Search for jobs
python brave-job-search.py

# Step 2: Process the results
python job_processor.py

# Step 3: Start the web interface
python web_interface.py

# Step 4: Open http://localhost:7860 in your browser
```

## Security Considerations

- The interface runs on `0.0.0.0` by default, making it accessible from your local network
- Do not expose the interface to the public internet without authentication
- Your `.env` file contains sensitive information (API keys)
- Consider using a reverse proxy with authentication for production use

## Performance Tips

- For large job databases, use filters to reduce the number of jobs displayed
- Semantic search is faster with fewer results (adjust the slider)
- Close the interface when not in use to free up resources
- Consider archiving old jobs to improve performance

## Logging

All interface actions are logged to `web_interface.log`:

```log
2026-01-23 14:20:00 - INFO - Configuration saved successfully
2026-01-23 14:20:05 - INFO - Added provider: boards.greenhouse.io
2026-01-23 14:20:10 - INFO - Semantic search: 'Python developer' returned 5 results
```

Check this file for troubleshooting and audit purposes.

## License

This web interface is part of the brave-job-search project. See LICENSE.md for details.