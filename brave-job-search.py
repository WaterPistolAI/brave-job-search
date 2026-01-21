import httpx
import json
import time
import os
import ast
import html
import re
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "YOUR_BRAVE_API_KEY")
print(
    f"DEBUG: API_KEY loaded: {'YES' if BRAVE_API_KEY != 'YOUR_BRAVE_API_KEY' else 'NO (using default)'}"
)
BRAVE_BASE_URL = os.environ.get(
    "BRAVE_BASE_URL", "https://api.search.brave.com/res/v1/web/search"
)
BRAVE_USAGE_FILE = os.environ.get("USAGE_FILE", "api_usage.json")
BRAVE_RATE_LIMIT = float(os.environ.get("BRAVE_RATE_LIMIT", "1.1"))
BRAVE_MONTHLY_LIMIT = int(os.environ.get("BRAVE_MONTHLY_LIMIT", "2000"))
BRAVE_JOB_SEARCH_LOGFILE = os.environ.get(
    "BRAVE_JOB_SEARCH_LOGFILE", "brave-job-search.log"
)

logging.basicConfig(
    level=logging.INFO,
    filename=BRAVE_JOB_SEARCH_LOGFILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def load_list_from_env(key, default_val):
    raw_value = os.environ.get(key)
    if not raw_value:
        return default_val
    try:
        # This converts "[1, 2]" string into [1, 2] list
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing {key}: {e}. Falling back to default.")
        return default_val


BRAVE_ATS_DOMAINS = load_list_from_env("BRAVE_ATS_DOMAINS", ["boards.greenhouse.io"])
BRAVE_TARGET_QUERIES = load_list_from_env(
    "BRAVE_TARGET_QUERIES", ['"Founding Engineer"']
)


def get_monthly_usage():
    """Tracks usage across script restarts."""
    if not os.path.exists(BRAVE_USAGE_FILE):
        return {"count": 0, "month": datetime.now().month}

    with open(BRAVE_USAGE_FILE, "r") as f:
        data = json.load(f)
        # Reset if it's a new month
        if data["month"] != datetime.now().month:
            return {"count": 0, "month": datetime.now().month}
        return data


def update_usage(count):
    with open(BRAVE_USAGE_FILE, "w") as f:
        json.dump({"count": count, "month": datetime.now().month}, f)


def fetch_brave_jobs(query):
    usage = get_monthly_usage()
    if usage["count"] >= BRAVE_MONTHLY_LIMIT:
        print("CRITICAL: Monthly API limit reached. Stopping.")
        return None

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": query}

    # Respect the 1 req/sec limit strictly
    time.sleep(BRAVE_RATE_LIMIT)

    try:
        response = httpx.get(BRAVE_BASE_URL, headers=headers, params=params)

        # Track usage from Brave's actual response headers
        # Format usually: "1, 14523" (per-sec, per-month)
        remaining_header = response.headers.get("X-RateLimit-Remaining", "")
        if "," in remaining_header:
            monthly_left = int(remaining_header.split(",")[1].strip())
            usage["count"] = BRAVE_MONTHLY_LIMIT - monthly_left
        else:
            usage["count"] += 1

        update_usage(usage["count"])

        if response.status_code == 200:
            data = response.json()
            results = data.get("web", {}).get("results", [])
            print(f"DEBUG: Query '{query}' returned {len(results)} results")
            if not results:
                print(
                    f"DEBUG: Full response for empty results: {json.dumps(data, indent=2)}"
                )
            return results
        else:
            print(
                f"DEBUG: Query '{query}' failed with status {response.status_code}: {response.text}"
            )
            return []
    except Exception as e:
        print(f"Request failed: {e}")
        return []


def main():
    # Test with reference query
    print("Testing with reference query 'brave search'")
    test_results = fetch_brave_jobs("brave search")
    print(f"Test results: {len(test_results)} items")

    all_jobs = []
    for query_string in BRAVE_TARGET_QUERIES:
        for domain in BRAVE_ATS_DOMAINS:
            # Combining the wildcard/OR string with the site restriction
            full_query = f"site:{domain} {query_string}"

            logging.info(f"Searching: {full_query}")
            results = fetch_brave_jobs(full_query)
            logging.info(f"Found {len(results)} results for {full_query}")

            if results:
                for res in results:
                    snippet = html.unescape(res.get("description", ""))
                    snippet = re.sub(r"<[^>]+>", "", snippet)
                    all_jobs.append(
                        {
                            "title": res.get("title"),
                            "url": res.get("url"),
                            "snippet": snippet,
                        }
                    )

    with open("job_results.json", "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
