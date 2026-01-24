"""
Multi-adapter job search that combines results from multiple search providers.
Supports Brave Search API and Google Custom Search API.
"""

import json
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="multi_search.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_multi_search(
    providers: List[str] = None,
    output_file: str = "job_results.json",
) -> List[Dict]:
    """
    Run job search using multiple providers and merge results.

    Args:
        providers: List of providers to use (e.g., ["brave", "google"])
                   If None, uses all configured providers
        output_file: Output file for merged results

    Returns:
        Combined list of job results from all providers
    """
    if providers is None:
        # Default to both if both are configured
        providers = []
        if os.environ.get("BRAVE_API_KEY"):
            providers.append("brave")
        if os.environ.get("GOOGLE_SEARCH_API_KEY") and os.environ.get(
            "GOOGLE_SEARCH_ID"
        ):
            providers.append("google")

    if not providers:
        logging.error("No search providers configured")
        print("❌ Error: No search providers configured")
        return []

    logging.info(f"Starting multi-provider search with providers: {providers}")
    print(f"🔍 Starting multi-provider search with: {', '.join(providers)}")

    all_results = []

    # Run each provider
    for provider in providers:
        try:
            if provider == "brave":
                logging.info("Running Brave Search")
                print("📡 Running Brave Search...")
                import brave_job_search

                # Temporarily change output file to avoid conflicts
                original_output = "job_results.json"
                brave_output = "brave_job_results_temp.json"

                # Run Brave search
                brave_job_search.main()

                # Load results
                try:
                    with open(original_output, "r", encoding="utf-8") as f:
                        brave_results = json.load(f)
                    all_results.extend(brave_results)
                    logging.info(f"Brave Search: Found {len(brave_results)} results")
                    print(f"✅ Brave Search: Found {len(brave_results)} results")
                except FileNotFoundError:
                    logging.warning("Brave Search: No results file found")
                    print("⚠️  Brave Search: No results found")

            elif provider == "google":
                logging.info("Running Google Search")
                print("🔎 Running Google Search...")
                import google_search

                # Run Google search
                google_search.main()

                # Load results
                google_output = "google_job_results.json"
                try:
                    with open(google_output, "r", encoding="utf-8") as f:
                        google_results = json.load(f)
                    all_results.extend(google_results)
                    logging.info(f"Google Search: Found {len(google_results)} results")
                    print(f"✅ Google Search: Found {len(google_results)} results")
                except FileNotFoundError:
                    logging.warning("Google Search: No results file found")
                    print("⚠️  Google Search: No results found")

        except Exception as e:
            logging.error(f"Error running {provider} search: {e}")
            print(f"❌ Error running {provider} search: {e}")

    # Remove duplicates based on URL
    seen_urls = {}
    unique_results = []
    for result in all_results:
        url = result["url"]
        if url not in seen_urls:
            seen_urls[url] = result
            unique_results.append(result)
        else:
            # Keep the result from the first provider that found it
            logging.info(
                f"Duplicate URL found: {url} (from {result.get('source_adapter', 'unknown')})"
            )

    # Count results by provider
    provider_counts = {}
    for result in unique_results:
        provider = result.get("source_adapter", "unknown")
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

    logging.info(f"Multi-search complete: {len(unique_results)} unique results")
    logging.info(f"Results by provider: {provider_counts}")

    # Save merged results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_results, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved {len(unique_results)} results to {output_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("📊 MULTI-SEARCH SUMMARY")
    print("=" * 50)
    print(f"Total unique results: {len(unique_results)}")
    print("\nResults by provider:")
    for provider, count in provider_counts.items():
        print(f"  - {provider}: {count} results")
    print(f"\n📄 Results saved to: {output_file}")
    print("=" * 50)

    return unique_results


def main():
    """Main function to run multi-provider job search."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run job search using multiple providers"
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="brave,google",
        help="Comma-separated list of providers (e.g., 'brave,google')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="job_results.json",
        help="Output file for merged results",
    )

    args = parser.parse_args()

    # Parse providers
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    # Validate providers
    valid_providers = ["brave", "google"]
    invalid_providers = [p for p in providers if p not in valid_providers]
    if invalid_providers:
        print(f"❌ Error: Invalid providers: {', '.join(invalid_providers)}")
        print(f"Valid providers: {', '.join(valid_providers)}")
        return

    # Run multi-search
    try:
        results = run_multi_search(providers=providers, output_file=args.output)
        if results:
            print(f"\n✅ Multi-search completed successfully!")
        else:
            print(f"\n⚠️  No results found")
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
