"""
Domain-specific scraping adapters for different ATS platforms.
Each adapter handles the unique HTML structure of a specific job board.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from bs4 import BeautifulSoup
import re


class BaseScraperAdapter(ABC):
    """Base class for domain-specific scraping adapters."""

    def __init__(self):
        self.domain = None

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Check if this adapter can handle the given URL."""
        pass

    @abstractmethod
    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        """
        Scrape job details from the parsed HTML.
        Returns a dictionary with job details.
        """
        pass

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        text = " ".join(text.split())
        return text.strip()


class GreenhouseAdapter(BaseScraperAdapter):
    """Adapter for Greenhouse job boards (boards.greenhouse.io)."""

    def __init__(self):
        super().__init__()
        self.domain = "boards.greenhouse.io"

    def can_handle(self, url: str) -> bool:
        return "boards.greenhouse.io" in url

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # Extract company name from URL or page
        company_match = re.search(r"boards\.greenhouse\.io/([^/]+)", url)
        if company_match:
            details["company"] = company_match.group(1).replace("-", " ").title()

        # Job description - Greenhouse typically uses specific sections
        # Look for the main content section
        main_content = soup.find("div", class_="section") or soup.find(
            "div", id="content"
        )

        if main_content:
            # Extract all paragraphs from the main content
            paragraphs = main_content.find_all("p")
            description_parts = []
            requirements_parts = []
            benefits_parts = []

            current_section = "description"

            for p in paragraphs:
                text = self._clean_text(p.get_text())
                if not text:
                    continue

                # Detect section headers
                text_lower = text.lower()
                if any(
                    keyword in text_lower
                    for keyword in [
                        "requirement",
                        "qualification",
                        "what you need",
                        "what we need",
                    ]
                ):
                    current_section = "requirements"
                    continue
                elif any(
                    keyword in text_lower
                    for keyword in [
                        "benefit",
                        "perk",
                        "what we offer",
                        "compensation",
                        "package",
                    ]
                ):
                    current_section = "benefits"
                    continue

                # Add to appropriate section
                if current_section == "description":
                    description_parts.append(text)
                elif current_section == "requirements":
                    requirements_parts.append(text)
                elif current_section == "benefits":
                    benefits_parts.append(text)

            details["job_description"] = "\n".join(description_parts)
            details["requirements"] = "\n".join(requirements_parts)
            details["benefits"] = "\n".join(benefits_parts)

        # Location - Greenhouse typically has a specific location element
        location_elem = soup.find("div", class_="location") or soup.find(
            "span", class_="location"
        )
        if location_elem:
            details["location"] = self._clean_text(location_elem.get_text())

        # Try to find location in the header or sidebar
        if not details["location"]:
            location_patterns = [
                soup.find("div", class_="sidebar"),
                soup.find("div", class_="header"),
            ]
            for pattern in location_patterns:
                if pattern:
                    location_text = pattern.get_text()
                    if "location" in location_text.lower():
                        # Extract location text
                        lines = location_text.split("\n")
                        for line in lines:
                            if "location" in line.lower():
                                details["location"] = self._clean_text(
                                    line.replace("location", "").replace(":", "")
                                )
                                break

        # Salary - Look for salary information
        salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
        page_text = soup.get_text().lower()
        for keyword in salary_keywords:
            if keyword in page_text:
                # Find the element containing this keyword
                element = soup.find(
                    text=lambda text: text and keyword.lower() in text.lower()
                )
                if element:
                    parent = element.parent
                    if parent:
                        text = self._clean_text(parent.get_text())
                        if "$" in text or any(c.isdigit() for c in text):
                            details["salary"] = text
                            break

        return details


class LeverAdapter(BaseScraperAdapter):
    """Adapter for Lever job boards (jobs.lever.co)."""

    def __init__(self):
        super().__init__()
        self.domain = "jobs.lever.co"

    def can_handle(self, url: str) -> bool:
        return "jobs.lever.co" in url

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # Extract company name from URL
        company_match = re.search(r"jobs\.lever\.co/([^/]+)", url)
        if company_match:
            details["company"] = company_match.group(1).replace("-", " ").title()

        # Lever typically uses specific sections
        # Job description
        desc_section = soup.find("div", class_="description") or soup.find(
            "section", class_="description"
        )
        if desc_section:
            details["job_description"] = self._clean_text(desc_section.get_text())

        # Requirements
        req_section = soup.find("div", class_="requirements") or soup.find(
            "section", class_="requirements"
        )
        if req_section:
            details["requirements"] = self._clean_text(req_section.get_text())

        # Benefits
        benefit_section = soup.find("div", class_="benefits") or soup.find(
            "section", class_="benefits"
        )
        if benefit_section:
            details["benefits"] = self._clean_text(benefit_section.get_text())

        # Location
        location_elem = soup.find("span", class_="location") or soup.find(
            "div", class_="location"
        )
        if location_elem:
            details["location"] = self._clean_text(location_elem.get_text())

        # Salary
        salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
        for keyword in salary_keywords:
            element = soup.find(
                text=lambda text: text and keyword.lower() in text.lower()
            )
            if element:
                parent = element.parent
                if parent:
                    text = self._clean_text(parent.get_text())
                    if "$" in text or any(c.isdigit() for c in text):
                        details["salary"] = text
                        break

        return details


class WorkdayAdapter(BaseScraperAdapter):
    """Adapter for Workday job boards."""

    def __init__(self):
        super().__init__()
        self.domain = "workday.com"

    def can_handle(self, url: str) -> bool:
        return "workday.com" in url or "myworkdayjobs.com" in url

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # Workday uses data attributes and specific classes
        # Job description
        desc_elem = soup.find(
            "div", {"data-automation-id": "jobDescription"}
        ) or soup.find("div", class_="job-description")
        if desc_elem:
            details["job_description"] = self._clean_text(desc_elem.get_text())

        # Requirements
        req_elem = soup.find(
            "div", {"data-automation-id": "qualifications"}
        ) or soup.find("div", class_="qualifications")
        if req_elem:
            details["requirements"] = self._clean_text(req_elem.get_text())

        # Location
        location_elem = soup.find(
            "div", {"data-automation-id": "location"}
        ) or soup.find("span", class_="location")
        if location_elem:
            details["location"] = self._clean_text(location_elem.get_text())

        # Company - often in the header
        company_elem = soup.find("div", class_="company-name") or soup.find(
            "span", class_="company"
        )
        if company_elem:
            details["company"] = self._clean_text(company_elem.get_text())

        # Salary
        salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
        for keyword in salary_keywords:
            element = soup.find(
                text=lambda text: text and keyword.lower() in text.lower()
            )
            if element:
                parent = element.parent
                if parent:
                    text = self._clean_text(parent.get_text())
                    if "$" in text or any(c.isdigit() for c in text):
                        details["salary"] = text
                        break

        return details


class SmartRecruitersAdapter(BaseScraperAdapter):
    """Adapter for SmartRecruiters job boards."""

    def __init__(self):
        super().__init__()
        self.domain = "smartrecruiters.com"

    def can_handle(self, url: str) -> bool:
        return "smartrecruiters.com" in url

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # SmartRecruiters uses specific data attributes
        # Job description
        desc_elem = soup.find("div", {"data-test": "job-description"}) or soup.find(
            "div", class_="job-description"
        )
        if desc_elem:
            details["job_description"] = self._clean_text(desc_elem.get_text())

        # Requirements
        req_elem = soup.find("div", {"data-test": "requirements"}) or soup.find(
            "div", class_="requirements"
        )
        if req_elem:
            details["requirements"] = self._clean_text(req_elem.get_text())

        # Location
        location_elem = soup.find("div", {"data-test": "location"}) or soup.find(
            "span", class_="location"
        )
        if location_elem:
            details["location"] = self._clean_text(location_elem.get_text())

        # Company
        company_elem = soup.find("div", {"data-test": "company-name"}) or soup.find(
            "span", class_="company"
        )
        if company_elem:
            details["company"] = self._clean_text(company_elem.get_text())

        # Salary
        salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
        for keyword in salary_keywords:
            element = soup.find(
                text=lambda text: text and keyword.lower() in text.lower()
            )
            if element:
                parent = element.parent
                if parent:
                    text = self._clean_text(parent.get_text())
                    if "$" in text or any(c.isdigit() for c in text):
                        details["salary"] = text
                        break

        return details


class WorkableAdapter(BaseScraperAdapter):
    """Adapter for Workable job boards (jobs.workable.com)."""

    def __init__(self):
        super().__init__()
        self.domain = "jobs.workable.com"

    def can_handle(self, url: str) -> bool:
        return "jobs.workable.com" in url or "workable.com" in url

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # Extract company name from the company link
        company_link = soup.find("a", class_=lambda x: x and "companyName__link" in x)
        if company_link:
            details["company"] = self._clean_text(company_link.get_text())

        # Fallback: extract company name from URL
        if not details["company"]:
            company_match = re.search(r"jobs\.workable\.com/([^/]+)", url)
            if company_match:
                details["company"] = company_match.group(1).replace("-", " ").title()

        # Extract location from data-ui attribute
        location_elem = soup.find("span", {"data-ui": "overview-location"})
        if location_elem:
            details["location"] = self._clean_text(location_elem.get_text())

        # Extract job description from parsed HTML content
        desc_content = soup.find(
            "div", {"data-ui": "job-breakdown-description-parsed-html"}
        )
        if desc_content:
            details["job_description"] = self._clean_text(desc_content.get_text())

        # Extract requirements - look for section with "requirement" or "qualification" in heading
        sections = soup.find_all("section")
        for section in sections:
            heading = section.find("h3")
            if heading:
                heading_text = heading.get_text().lower()
                if any(
                    keyword in heading_text
                    for keyword in [
                        "requirement",
                        "qualification",
                        "what we'd love to see",
                    ]
                ):
                    content = section.find(
                        "div", class_=lambda x: x and "parsedHtml__content" in x
                    )
                    if content:
                        details["requirements"] = self._clean_text(content.get_text())
                        break

        # Extract benefits - look for section with "benefit" or "what we offer" in heading
        for section in sections:
            heading = section.find("h3")
            if heading:
                heading_text = heading.get_text().lower()
                if any(
                    keyword in heading_text
                    for keyword in ["benefit", "what we offer", "what we offer"]
                ):
                    content = section.find(
                        "div", class_=lambda x: x and "parsedHtml__content" in x
                    )
                    if content:
                        details["benefits"] = self._clean_text(content.get_text())
                        break

        # Salary - look for salary information in the job details
        salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
        for keyword in salary_keywords:
            element = soup.find(
                text=lambda text: text and keyword.lower() in text.lower()
            )
            if element:
                parent = element.parent
                if parent:
                    text = self._clean_text(parent.get_text())
                    if "$" in text or any(c.isdigit() for c in text):
                        details["salary"] = text
                        break

        return details


class JobviteAdapter(BaseScraperAdapter):
    """Adapter for Jobvite job boards (careers.jobvite.com)."""

    def __init__(self):
        super().__init__()
        self.domain = "careers.jobvite.com"

    def can_handle(self, url: str) -> bool:
        return "careers.jobvite.com" in url or "jobvite.com" in url

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # Extract company name from URL
        company_match = re.search(r"careers\.jobvite\.com/([^/]+)", url)
        if company_match:
            details["company"] = company_match.group(1).replace("-", " ").title()

        # Jobvite uses specific data attributes
        # Job description
        desc_elem = soup.find("div", {"data-testid": "job-description"}) or soup.find(
            "div", class_="job-description"
        )
        if desc_elem:
            details["job_description"] = self._clean_text(desc_elem.get_text())

        # Requirements
        req_elem = soup.find("div", {"data-testid": "requirements"}) or soup.find(
            "div", class_="requirements"
        )
        if req_elem:
            details["requirements"] = self._clean_text(req_elem.get_text())

        # Location
        location_elem = soup.find("div", {"data-testid": "location"}) or soup.find(
            "span", class_="location"
        )
        if location_elem:
            details["location"] = self._clean_text(location_elem.get_text())

        # Salary
        salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
        for keyword in salary_keywords:
            element = soup.find(
                text=lambda text: text and keyword.lower() in text.lower()
            )
            if element:
                parent = element.parent
                if parent:
                    text = self._clean_text(parent.get_text())
                    if "$" in text or any(c.isdigit() for c in text):
                        details["salary"] = text
                        break

        return details


class ICIMSAdapter(BaseScraperAdapter):
    """Adapter for iCIMS job boards (my.icims.com)."""

    def __init__(self):
        super().__init__()
        self.domain = "my.icims.com"

    def can_handle(self, url: str) -> bool:
        return "my.icims.com" in url or "icims.com" in url

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # iCIMS uses specific data attributes and classes
        # Job description
        desc_elem = soup.find(
            "div", {"data-automation-id": "jobDescription"}
        ) or soup.find("div", class_="job-description")
        if desc_elem:
            details["job_description"] = self._clean_text(desc_elem.get_text())

        # Requirements
        req_elem = soup.find(
            "div", {"data-automation-id": "qualifications"}
        ) or soup.find("div", class_="qualifications")
        if req_elem:
            details["requirements"] = self._clean_text(req_elem.get_text())

        # Location
        location_elem = soup.find(
            "div", {"data-automation-id": "location"}
        ) or soup.find("span", class_="location")
        if location_elem:
            details["location"] = self._clean_text(location_elem.get_text())

        # Company
        company_elem = soup.find("div", class_="company-name") or soup.find(
            "span", class_="company"
        )
        if company_elem:
            details["company"] = self._clean_text(company_elem.get_text())

        # Salary
        salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
        for keyword in salary_keywords:
            element = soup.find(
                text=lambda text: text and keyword.lower() in text.lower()
            )
            if element:
                parent = element.parent
                if parent:
                    text = self._clean_text(parent.get_text())
                    if "$" in text or any(c.isdigit() for c in text):
                        details["salary"] = text
                        break

        return details


class LevelsFYIAdapter(BaseScraperAdapter):
    """Adapter for Levels.fyi job boards."""

    def __init__(self):
        super().__init__()
        self.domain = "levels.fyi"

    def can_handle(self, url: str) -> bool:
        return "levels.fyi" in url

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # Levels.fyi has a specific structure
        # Job description
        desc_elem = soup.find("div", class_="job-description") or soup.find(
            "div", {"data-testid": "job-description"}
        )
        if desc_elem:
            details["job_description"] = self._clean_text(desc_elem.get_text())

        # Requirements
        req_elem = soup.find("div", class_="requirements") or soup.find(
            "div", {"data-testid": "requirements"}
        )
        if req_elem:
            details["requirements"] = self._clean_text(req_elem.get_text())

        # Location
        location_elem = soup.find("div", class_="location") or soup.find(
            "span", class_="location"
        )
        if location_elem:
            details["location"] = self._clean_text(location_elem.get_text())

        # Company
        company_elem = soup.find("div", class_="company-name") or soup.find(
            "span", class_="company"
        )
        if company_elem:
            details["company"] = self._clean_text(company_elem.get_text())

        # Salary - Levels.fyi often has salary information prominently
        salary_elem = soup.find("div", class_="salary") or soup.find(
            "div", {"data-testid": "salary"}
        )
        if salary_elem:
            details["salary"] = self._clean_text(salary_elem.get_text())

        # If no salary element found, try to find it in the text
        if not details["salary"]:
            salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
            for keyword in salary_keywords:
                element = soup.find(
                    text=lambda text: text and keyword.lower() in text.lower()
                )
                if element:
                    parent = element.parent
                    if parent:
                        text = self._clean_text(parent.get_text())
                        if "$" in text or any(c.isdigit() for c in text):
                            details["salary"] = text
                            break

        return details


class GenericAdapter(BaseScraperAdapter):
    """Generic adapter for job boards without specific adapters."""

    def __init__(self):
        super().__init__()
        self.domain = "generic"

    def can_handle(self, url: str) -> bool:
        return True  # Can handle any URL as fallback

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # Try to extract job description from common containers
        desc_selectors = [
            'div[class*="description"]',
            'div[class*="job-description"]',
            'section[class*="description"]',
            'div[data-testid="job-description"]',
            'div[id*="job-description"]',
            'div[class*="posting-description"]',
            "article",
            "main",
        ]

        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                details["job_description"] = self._clean_text(element.get_text())
                break

        # If no specific selector found, try to find the largest text block
        if not details["job_description"]:
            text_blocks = soup.find_all(["div", "section", "article"])
            if text_blocks:
                largest_block = max(text_blocks, key=lambda x: len(x.get_text()))
                details["job_description"] = self._clean_text(largest_block.get_text())

        # Try to extract requirements
        req_keywords = [
            "requirement",
            "qualification",
            "skill",
            "what you need",
            "what we need",
        ]
        for keyword in req_keywords:
            element = soup.find(
                text=lambda text: text and keyword.lower() in text.lower()
            )
            if element:
                parent = element.parent
                if parent:
                    details["requirements"] = self._clean_text(parent.get_text())
                    break

        # Try to extract benefits
        benefit_keywords = [
            "benefit",
            "perk",
            "what we offer",
            "compensation",
            "package",
        ]
        for keyword in benefit_keywords:
            element = soup.find(
                text=lambda text: text and keyword.lower() in text.lower()
            )
            if element:
                parent = element.parent
                if parent:
                    details["benefits"] = self._clean_text(parent.get_text())
                    break

        # Try to extract location
        location_selectors = [
            'div[class*="location"]',
            'span[class*="location"]',
            'div[data-testid="location"]',
            'p[class*="location"]',
        ]
        for selector in location_selectors:
            element = soup.select_one(selector)
            if element:
                details["location"] = self._clean_text(element.get_text())
                break

        # Try to extract company
        company_selectors = [
            'div[class*="company"]',
            'span[class*="company"]',
            'div[data-testid="company-name"]',
            'h1[class*="company"]',
        ]
        for selector in company_selectors:
            element = soup.select_one(selector)
            if element:
                details["company"] = self._clean_text(element.get_text())
                break

        # Try to extract salary
        salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
        for keyword in salary_keywords:
            element = soup.find(
                text=lambda text: text and keyword.lower() in text.lower()
            )
            if element:
                parent = element.parent
                if parent:
                    text = self._clean_text(parent.get_text())
                    if "$" in text or any(c.isdigit() for c in text):
                        details["salary"] = text
                        break

        return details


class AdapterRegistry:
    """Registry for managing scraping adapters."""

    def __init__(self):
        self.adapters = []
        self._register_default_adapters()

    def _register_default_adapters(self):
        """Register default adapters."""
        self.register_adapter(GreenhouseAdapter())
        self.register_adapter(LeverAdapter())
        self.register_adapter(WorkdayAdapter())
        self.register_adapter(SmartRecruitersAdapter())
        self.register_adapter(WorkableAdapter())
        self.register_adapter(JobviteAdapter())
        self.register_adapter(ICIMSAdapter())
        self.register_adapter(LevelsFYIAdapter())
        self.register_adapter(
            GenericAdapter()
        )  # Always register generic last as fallback

    def register_adapter(self, adapter: BaseScraperAdapter):
        """Register a new adapter."""
        self.adapters.append(adapter)

    def get_adapter(self, url: str) -> BaseScraperAdapter:
        """Get the appropriate adapter for a given URL."""
        for adapter in self.adapters:
            if adapter.can_handle(url):
                return adapter
        return self.adapters[-1]  # Return generic adapter as fallback


# Global adapter registry instance
adapter_registry = AdapterRegistry()
