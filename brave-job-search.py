import httpx
import json
import time
import os
from datetime import datetime

# --- CONFIGURATION ---
API_KEY = os.environ.get("BRAVE_API_KEY", "YOUR_BRAVE_API_KEY")
print(
    f"DEBUG: API_KEY loaded: {'YES' if API_KEY != 'YOUR_BRAVE_API_KEY' else 'NO (using default)'}"
)
BASE_URL = "https://api.search.brave.com/res/v1/web/search"
USAGE_FILE = "api_usage.json"
MONTHLY_LIMIT = 2000

ATS_DOMAINS = [
    "jobs.ashbyhq.com",
    "boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.smartrecruiters.com",
    "wd1.myworkdayjobs.com",
    "jobs.bamboohr.com",
    "jobs.jobvite.com",
    "careers.icims.com",
    "apply.jazz.co",
    "careers.workable.com",
]


TARGET_QUERIES = [
    # '("Founding" | "First") * Engineer',  # Catches Founding ML, Founding Software, etc.
    '("GenAI" | "Generative AI" | "LLM") (Engineer | Developer)',
    '("Developer Advocacy" | "Developer Relations" | "DevRel" | "Developer Advocate")',
]


def get_monthly_usage():
    """Tracks usage across script restarts."""
    if not os.path.exists(USAGE_FILE):
        return {"count": 0, "month": datetime.now().month}

    with open(USAGE_FILE, "r") as f:
        data = json.load(f)
        # Reset if it's a new month
        if data["month"] != datetime.now().month:
            return {"count": 0, "month": datetime.now().month}
        return data


def update_usage(count):
    with open(USAGE_FILE, "w") as f:
        json.dump({"count": count, "month": datetime.now().month}, f)


def fetch_brave_jobs(query):
    usage = get_monthly_usage()
    if usage["count"] >= MONTHLY_LIMIT:
        print("CRITICAL: Monthly API limit reached. Stopping.")
        return None

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": API_KEY,
    }
    params = {"q": query}

    # Respect the 1 req/sec limit strictly
    time.sleep(1.1)

    try:
        response = httpx.get(BASE_URL, headers=headers, params=params)

        # Track usage from Brave's actual response headers
        # Format usually: "1, 14523" (per-sec, per-month)
        remaining_header = response.headers.get("X-RateLimit-Remaining", "")
        if "," in remaining_header:
            monthly_left = int(remaining_header.split(",")[1].strip())
            usage["count"] = MONTHLY_LIMIT - monthly_left
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
    for query_string in TARGET_QUERIES:
        for domain in ATS_DOMAINS:
            # Combining the wildcard/OR string with the site restriction
            full_query = f"site:{domain} {query_string}"

            print(f"Searching: {full_query}")
            results = fetch_brave_jobs(full_query)

            if results:
                for res in results:
                    all_jobs.append(
                        {
                            "title": res.get("title"),
                            "url": res.get("url"),
                            "snippet": res.get("description"),
                        }
                    )

    with open("job_results.json", "w") as f:
        json.dump(all_jobs, f, indent=4)


if __name__ == "__main__":
    main()
