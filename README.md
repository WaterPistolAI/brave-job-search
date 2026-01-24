This README is designed to help you (or any other developers) set up and run the search script while respecting the tight API constraints of the Brave Search Free Tier.

---

# 🚀 Brave Job Search

A high-efficiency Python script that uses the **Brave Search API** to search across specific Applicant Tracking Systems (ATS) for high-intent job roles like **Founding Engineer**, **GenAI Developer**, and **DevRel**.

### ✨ Key Features

* **ATS Targeting:** Searches directly on `greenhouse.io`, `lever.co`, and more.
* **Rate Limit Protection:** Strictly respects the **1 request per second** limit.
* **Usage Tracking:** Maintains a local `api_usage.json` to ensure you never exceed the **2,000 requests/month** limit.
* **Freshness Filter:** Automatically filters for roles posted in the **last week** to ensure leads are active.
* **Wildcard Support:** Uses complex boolean logic (OR/Wildcards) to maximize result density per API call.

---

## 🛠️ Setup

### 1. Prerequisites

* Python 3.8+
* A Brave Search API Key ([Get it here](https://api-dashboard.search.brave.com/app/dashboard))

### 2. Installation

```bash
# Clone this repository (or save the script)
git clone https://github.com/WaterPistolAI/brave-job-search.git
cd brave-job-search

# Install dependencies
pip install httpx python-dotenv

```

### 3. Environment Configuration

Create a `.env` file in the root directory. **Note:** Complex lists are wrapped in single quotes for `ast.literal_eval` parsing.

```env
BRAVE_API_KEY=YOUR_ACTUAL_KEY_HERE
BRAVE_MONTHLY_LIMIT=2000
BRAVE_USAGE_FILE=api_usage.json

# ATS Domains to target
BRAVE_ATS_DOMAINS='["boards.greenhouse.io", "jobs.lever.co", "jobs.smartrecruiters.com", "jobs.workable.com"]'

# Search Queries (Boolean logic supported)
BRAVE_TARGET_QUERIES='["(\"GenAI\" | \"Generative AI\" | \"LLM\") (Engineer | Developer)", "(\"Developer Advocacy\" | \"Developer Relations\" | \"DevRel\")"]'

```

---

## 🚀 Usage

Run the script:

```bash
python brave-job-search.py

```

### How it works

1. **Usage Check:** The script checks `api_usage.json` to see if you have credits remaining.
2. **The Loop:** It iterates through every `Target Query` against every `ATS Domain`.
3. **The Wait:** A `time.sleep(1.1)` is enforced between every call to prevent 429 errors.
4. **The Output:** Results are saved to `job_results.json` with the title, direct URL, and a snippet of the job description.

---

## 📊 Search Logic Explanation

The script uses advanced search operators to minimize "noise" and maximize your API credits:

| Operator | Usage | Benefit |
| --- | --- | --- |
| `site:` | `site:jobs.lever.co` | Restricts search to specific job board platforms. |
| ` | ` (OR) | `"GenAI" |
| `*` | `"Founding * Engineer"` | Matches "Founding **AI** Engineer" or "Founding **Software** Engineer". |
| `freshness` | `pw` (Past Week) | Only returns jobs posted recently. |

---

## ⚠️ Important Limitations

* **API Quota:** The Free tier allows 2,000 requests per month. With 10 ATS domains and 2 target queries, one "run" of this script uses **20 requests**.
* **Rate Limits:** Do not remove the `time.sleep(1.1)` or you will receive a `429 Too Many Requests` response from Brave.

---

## 📝 License

The GNU General Public License is a free, copyleft license for software and other kinds of works.
