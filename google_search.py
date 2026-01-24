"""
Google Custom Search API adapter for job searching.
Provides an alternative to Brave Search API.
"""

import httpx
import time
import json
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_ID = os.environ.get("GOOGLE_SEARCH_ID", "")
GOOGLE_SEARCH_RATE_LIMIT = float(os.environ.get("GOOGLE_SEARCH_RATE_LIMIT", "1.1"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="google_search.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class GoogleSearchAPI:
    """Google Custom Search API client for job searching."""

    def __init__(
        self,
        api_key: str = GOOGLE_SEARCH_API_KEY,
        search_engine_id: str = GOOGLE_SEARCH_ID,
        rate_limit: float = GOOGLE_SEARCH_RATE_LIMIT,
    ):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.base_url = "https://www.googleapis.com/customsearch/v1"

        if not self.api_key:
            raise ValueError(
                "Google Search API key is required. Set GOOGLE_SEARCH_API_KEY environment variable."
            )

        if not self.search_engine_id:
            raise ValueError(
                "Google Search Engine ID is required. Set GOOGLE_SEARCH_ID environment variable."
            )

        logging.info("Google Search API initialized")

    def _rate_limit_wait(self):
        """Wait to respect rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def search(
        self,
        query: str,
        num_results: int = 10,
        start_index: int = 1,
        date_restrict: Optional[str] = None,
        site_search: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search for jobs using Google Custom Search API.

        Args:
            query: Search query string
            num_results: Number of results to return (1-10)
            start_index: Starting index for results (for pagination)
            date_restrict: Restrict results by date (e.g., "d7" for past 7 days)
            site_search: Restrict search to specific site

        Returns:
            List of job results with title, url, and snippet
        """
        self._rate_limit_wait()

        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(num_results, 10),  # Google API max is 10
            "start": start_index,
        }

        # Add optional parameters
        if date_restrict:
            params["dateRestrict"] = date_restrict

        if site_search:
            params["siteSearch"] = site_search
            params["siteSearchFilter"] = "i"  # Include results from this site

        try:
            response = httpx.get(
                self.base_url,
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()

            # Extract search results
            results = []
            if "items" in data:
                for item in data["items"]:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                        }
                    )

            logging.info(
                f"Google Search: Found {len(results)} results for query: {query}"
            )
            return results

        except httpx.HTTPStatusError as e:
            logging.error(
                f"Google Search API error: {e.response.status_code} - {e.response.text}"
            )
            return []
        except Exception as e:
            logging.error(f"Error in Google Search: {e}")
            return []

    def search_jobs(
        self,
        queries: List[str],
        ats_domains: List[str],
        num_results_per_query: int = 10,
        date_restrict: str = "d7",  # Past 7 days by default
    ) -> List[Dict]:
        """
        Search for jobs across multiple queries and ATS domains.

        Args:
            queries: List of search queries
            ats_domains: List of ATS domains to search within
            num_results_per_query: Number of results per query
            date_restrict: Date restriction for results

        Returns:
            Combined list of job results from all searches
        """
        all_results = []

        for query in queries:
            for domain in ats_domains:
                # Search for jobs on this ATS domain
                results = self.search(
                    query=query,
                    num_results=num_results_per_query,
                    date_restrict=date_restrict,
                    site_search=domain,
                )
                all_results.extend(results)

                logging.info(
                    f"Google Search: Query '{query}' on {domain} returned {len(results)} results"
                )

        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append(result)

        logging.info(f"Google Search: Total unique results: {len(unique_results)}")
        return unique_results


def main():
    """Main function to run Google job search."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Search for jobs using Google Custom Search API"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default='["(GenAI | Generative AI | LLM) ( Application | Applied ) (Engineer | Developer)", "(Data) (Scientist | Science)"]',
        help="JSON array of search queries",
    )
    parser.add_argument(
        "--ats-domains",
        type=str,
        default='["jobs.ashbyhq.com", "boards.greenhouse.io", "jobs.lever.co", "jobs.smartrecruiters.com", "apply.workable.com", "wd1.myworkdayjobs.com", "jobs.jobvite.com", "careers.icims.com", "www.levels.fyi/jobs"]',
        help="JSON array of ATS domains to search",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="google_job_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=10,
        help="Number of results per query",
    )
    parser.add_argument(
        "--date-restrict",
        type=str,
        default="d7",
        help="Date restriction (e.g., d7 for past 7 days, w1 for past week)",
    )

    args = parser.parse_args()

    try:
        # Parse JSON arguments
        queries = json.loads(args.queries)
        ats_domains = json.loads(args.ats_domains)

        logging.info(
            f"Starting Google job search with {len(queries)} queries and {len(ats_domains)} domains"
        )

        # Initialize Google Search API
        google_search = GoogleSearchAPI()

        # Search for jobs
        results = google_search.search_jobs(
            queries=queries,
            ats_domains=ats_domains,
            num_results_per_query=args.num_results,
            date_restrict=args.date_restrict,
        )

        # Save results to file
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved {len(results)} results to {args.output}")
        print(f"✅ Google job search completed! Found {len(results)} unique jobs.")
        print(f"📄 Results saved to: {args.output}")

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON arguments: {e}")
        print(f"❌ Error parsing JSON arguments: {e}")
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
