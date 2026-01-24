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

        # Check if this is the new Greenhouse structure
        is_new_structure = soup.find("main", class_=lambda x: x and "job-post" in x)

        if is_new_structure:
            # New Greenhouse structure
            # Extract company name from URL
            company_match = re.search(r"boards\.greenhouse\.io/([^/]+)", url)
            if company_match:
                details["company"] = company_match.group(1).replace("-", " ").title()

            # Extract location from job__location div
            location_elem = soup.find(
                "div", class_=lambda x: x and "job__location" in x
            )
            if location_elem:
                # Location is in a div inside job__location
                location_div = location_elem.find("div")
                if location_div:
                    details["location"] = self._clean_text(location_div.get_text())

            # Extract job description from job__description.body div
            desc_container = soup.find(
                "div", class_=lambda x: x and "job__description" in x and "body" in x
            )
            if desc_container:
                # Get all content divs
                content_divs = desc_container.find_all("div", recursive=False)
                description_parts = []
                requirements_parts = []
                benefits_parts = []
                current_section = "description"

                for content_div in content_divs:
                    # Get all text content from this div
                    text = self._clean_text(content_div.get_text())
                    if not text:
                        continue

                    # Check for section headers (h3 or strong tags)
                    header = content_div.find("h3") or content_div.find("strong")
                    if header:
                        header_text = header.get_text().lower()
                        if "about" in header_text and "role" in header_text:
                            current_section = "description"
                        elif "about" in header_text and "company" in header_text:
                            current_section = "description"
                        elif (
                            "duties" in header_text or "responsibilities" in header_text
                        ):
                            current_section = "description"
                        elif "qualification" in header_text or "skills" in header_text:
                            current_section = "requirements"
                        elif "benefit" in header_text or "what we offer" in header_text:
                            current_section = "benefits"

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

            # Extract salary - look for salary range patterns
            salary_patterns = [
                r"salary range is \$[\d,]+ - \$[\d,]+",
                r"base salary range is \$[\d,]+ - \$[\d,]+",
                r"\$[\d,]+ - \$[\d,]+",
            ]
            page_text = soup.get_text()
            for pattern in salary_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    details["salary"] = match.group(0)
                    break

        else:
            # Legacy Greenhouse structure
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

        # Extract location from posting categories
        location_elem = soup.find("div", class_=lambda x: x and "location" in x)
        if location_elem:
            details["location"] = self._clean_text(location_elem.get_text())

        # Extract job description from data-qa attribute
        desc_section = soup.find("div", {"data-qa": "job-description"})
        if desc_section:
            details["job_description"] = self._clean_text(desc_section.get_text())

        # Extract requirements - look for section with "What We Require" or similar heading
        sections = soup.find_all("div", class_="section")
        for section in sections:
            heading = section.find("h3")
            if heading:
                heading_text = heading.get_text().lower()
                if any(
                    keyword in heading_text
                    for keyword in [
                        "what we require",
                        "requirement",
                        "qualification",
                        "core responsibilities",
                    ]
                ):
                    details["requirements"] = self._clean_text(section.get_text())
                    break

        # Extract benefits - look for section with "Benefits" or "What We Value" heading
        for section in sections:
            heading = section.find("h3")
            if heading:
                heading_text = heading.get_text().lower()
                if any(
                    keyword in heading_text
                    for keyword in ["benefit", "what we value", "perks"]
                ):
                    details["benefits"] = self._clean_text(section.get_text())
                    break

        # Extract salary from closing description section
        closing_section = soup.find("div", {"data-qa": "closing-description"})
        if closing_section:
            # Look for salary information in the closing section
            salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
            for keyword in salary_keywords:
                element = closing_section.find(
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

        # Check if this is the new Workday structure (wd1.myworkdayjobs.com)
        is_new_structure = "wd1.myworkdayjobs.com" in url or soup.find(
            "div", id="root"
        )

        if is_new_structure:
            # New Workday structure with CSS classes
            # Extract description from the main content area
            # Look for div with css-oplht1 class (description container)
            desc_container = soup.find(
                "div", class_=lambda x: x and "css-oplht1" in x
            )
            if desc_container:
                # Get all paragraphs and build the description
                paragraphs = desc_container.find_all("p")
                description_parts = []
                requirements_parts = []
                benefits_parts = []
                current_section = "description"

                for p in paragraphs:
                    text = self._clean_text(p.get_text())
                    if not text:
                        continue

                    # Detect section headers based on bold text
                    bold_text = p.find("b")
                    if bold_text:
                        header_text = bold_text.get_text().lower()
                        if "company overview" in header_text:
                            current_section = "description"
                            continue
                        elif "job description" in header_text or "preferred qualifications" in header_text:
                            current_section = "description"
                            continue
                        elif "key responsibilities" in header_text:
                            current_section = "description"
                            continue
                        elif "qualifications" in header_text or "minimum qualifications" in header_text:
                            current_section = "requirements"
                            continue
                        elif "benefits" in header_text or "total rewards" in header_text:
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

            # Extract location - look for "Primary Location:" pattern
            location_keywords = ["primary location:", "location:", "work location:"]
            for keyword in location_keywords:
                element = soup.find(
                    text=lambda text: text and keyword.lower() in text.lower()
                )
                if element:
                    parent = element.parent
                    if parent:
                        text = self._clean_text(parent.get_text())
                        # Extract location after the keyword
                        if keyword.lower() in text.lower():
                            location = text.lower().split(keyword.lower())[-1].strip()
                            details["location"] = location
                            break

            # Extract company from URL or page
            if not details["company"]:
                company_match = re.search(r"//([^/]+)\.myworkdayjobs\.com", url)
                if company_match:
                    details["company"] = company_match.group(1).replace("-", " ").title()

            # Extract salary - look for "Base Pay Range:" pattern
            salary_keywords = ["base pay range:", "salary range:", "pay range:"]
            for keyword in salary_keywords:
                element = soup.find(
                    text=lambda text: text and keyword.lower() in text.lower()
                )
                if element:
                    parent = element.parent
                    if parent:
                        text = self._clean_text(parent.get_text())
                        # Extract salary after the keyword
                        if keyword.lower() in text.lower():
                            salary = text.lower().split(keyword.lower())[-1].strip()
                            if "$" in salary or any(c.isdigit() for c in salary):
                                details["salary"] = salary
                                break

        else:
            # Legacy Workday structure with data attributes and specific classes
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

        # Check if this is the new SmartRecruiters structure
        is_new_structure = soup.find("main", class_=lambda x: x and "jobad-main" in x)

        if is_new_structure:
            # New SmartRecruiters structure
            # Extract company from meta tags
            company_meta = soup.find("meta", property="og:site_name")
            if company_meta:
                details["company"] = company_meta.get("content", "")

            # Fallback: extract company from hiringOrganization meta
            if not details["company"]:
                company_meta = soup.find("meta", itemprop="name")
                if company_meta:
                    details["company"] = company_meta.get("content", "")

            # Extract location from spl-job-location element
            location_elem = soup.find("spl-job-location")
            if location_elem:
                details["location"] = location_elem.get("formattedaddress", "")

            # Fallback: extract location from jobLocation meta
            if not details["location"]:
                locality = soup.find("meta", itemprop="addressLocality")
                region = soup.find("meta", itemprop="addressRegion")
                if locality and region:
                    details["location"] = (
                        f"{locality.get('content')}, {region.get('content')}"
                    )

            # Extract job description from job sections
            job_sections = soup.find_all("section", class_="job-section")
            for section in job_sections:
                section_id = section.get("id", "")
                title_elem = section.find("h2", class_="title")
                title_text = title_elem.get_text().lower() if title_elem else ""

                # Get the wysiwyg content
                wysiwyg = section.find("div", class_="wysiwyg")
                if wysiwyg:
                    content = self._clean_text(wysiwyg.get_text())

                    # Categorize based on section ID or title
                    if (
                        "jobDescription" in section_id
                        or "job description" in title_text
                    ):
                        details["job_description"] = content
                    elif (
                        "qualifications" in section_id or "qualifications" in title_text
                    ):
                        details["requirements"] = content
                    elif (
                        "additionalInformation" in section_id
                        or "additional information" in title_text
                    ):
                        # Additional information might contain benefits
                        if (
                            "benefit" in content.lower()
                            or "compensation" in content.lower()
                        ):
                            details["benefits"] = content

            # Extract salary - look for pay range patterns
            salary_patterns = [
                r"pay range for this role is \$[\d,]+ to \$[\d,]+",
                r"salary range is \$[\d,]+ to \$[\d,]+",
                r"\$[\d,]+ to \$[\d,]+",
            ]
            page_text = soup.get_text()
            for pattern in salary_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    details["salary"] = match.group(0)
                    break

        else:
            # Legacy SmartRecruiters structure with data attributes
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

        # Extract company name from og:title meta tag
        # Format: "Power Integrations is looking for Staff Field Application Engineer."
        og_title = soup.find("meta", property="og:title")
        if og_title:
            title_text = og_title.get("content", "")
            # Extract company name (everything before " is looking for")
            if " is looking for " in title_text:
                details["company"] = title_text.split(" is looking for ")[0].strip()
            elif " is hiring " in title_text:
                details["company"] = title_text.split(" is hiring ")[0].strip()
            else:
                # Fallback: use the first part of the title
                details["company"] = title_text.split()[0] if title_text else ""

        # Fallback: extract company name from URL
        if not details["company"]:
            company_match = re.search(r"careers\.jobvite\.com/([^/]+)", url)
            if company_match:
                details["company"] = company_match.group(1).replace("-", " ").title()

        # Extract location from job detail meta
        # Format: "Sales<span class="jv-inline-separator"></span>San Jose, California"
        meta_elem = soup.find("p", class_="jv-job-detail-meta")
        if meta_elem:
            # Get all text nodes, excluding the separator
            text_parts = []
            for child in meta_elem.children:
                if child.name is None:  # Text node
                    text_parts.append(child.strip())

            # Location is typically the last text part (after the separator)
            if len(text_parts) >= 2:
                details["location"] = text_parts[-1]
            elif len(text_parts) == 1:
                # If only one part, check if it contains location info
                text = text_parts[0]
                # Try to extract location (usually after a comma or separator)
                if "," in text:
                    details["location"] = text.split(",")[-1].strip()

        # Extract job description from jv-job-detail-description div
        desc_elem = soup.find("div", class_="jv-job-detail-description")
        if desc_elem:
            details["job_description"] = self._clean_text(desc_elem.get_text())

        # Extract requirements - look for "Required Experience & Skills" or similar
        desc_text = details["job_description"].lower()
        if "required experience" in desc_text or "required skills" in desc_text:
            # Try to extract the requirements section
            req_keywords = [
                "required experience",
                "required skills",
                "qualification",
                "what we require",
            ]
            for keyword in req_keywords:
                if keyword in desc_text:
                    # Find the section containing this keyword
                    element = soup.find(
                        text=lambda text: text and keyword.lower() in text.lower()
                    )
                    if element:
                        parent = element.parent
                        if parent:
                            details["requirements"] = self._clean_text(
                                parent.get_text()
                            )
                            break

        # Extract benefits - look for "Compensation" or "Benefits" section
        if "compensation" in desc_text or "benefits" in desc_text:
            benefit_keywords = ["compensation", "benefits", "what we offer"]
            for keyword in benefit_keywords:
                if keyword in desc_text:
                    element = soup.find(
                        text=lambda text: text and keyword.lower() in text.lower()
                    )
                    if element:
                        parent = element.parent
                        if parent:
                            details["benefits"] = self._clean_text(parent.get_text())
                            break

        # Salary - look for salary information in the description
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


# class ICIMSAdapter(BaseScraperAdapter):
#     """Adapter for iCIMS job boards (my.icims.com)."""

#     def __init__(self):
#         super().__init__()
#         self.domain = "my.icims.com"

#     def can_handle(self, url: str) -> bool:
#         return "my.icims.com" in url or "icims.com" in url

#     def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
#         details = {
#             "job_description": "",
#             "requirements": "",
#             "benefits": "",
#             "location": "",
#             "company": "",
#             "salary": "",
#         }

#         # iCIMS uses specific class structure with potentially dynamic suffixes
#         # Main job content container - handle dynamic class names like iCIMS_ff147
#         job_content = (
#             soup.find("div", class_=lambda x: x and "iCIMS_JobContent" in x)
#             or soup.find("div", class_=lambda x: x and "iCIMS_JobContainer" in x)
#             or soup.find("div", class_=lambda x: x and "iCIMS_JobPage" in x)
#         )

#         # Extract company name from job header tags
#         company_elem = None
#         if job_content:
#             # Look for Company field in header tags
#             header_tags = job_content.find_all(
#                 "dt", class_=lambda x: x and "iCIMS_JobHeaderField" in x
#             )
#             for tag in header_tags:
#                 if tag.get_text().strip() == "Company":
#                     company_elem = tag.find_next_sibling(
#                         "dd", class_=lambda x: x and "iCIMS_JobHeaderData" in x
#                     )
#                     if company_elem:
#                         details["company"] = self._clean_text(company_elem.get_text())
#                         break

#         # Fallback: extract company from URL
#         if not details["company"]:
#             company_match = re.search(r"//([^/]+)\.icims\.com", url)
#             if company_match:
#                 details["company"] = company_match.group(1).replace("-", " ").title()

#         # Extract location from header section
#         if job_content:
#             # Look for location in the header left section
#             location_container = job_content.find(
#                 "div", class_=lambda x: x and "header left" in x
#             )
#             if location_container:
#                 # Location is typically in a span after the "Job Location" label
#                 spans = location_container.find_all("span")
#                 for span in spans:
#                     text = self._clean_text(span.get_text())
#                     # Skip the label and get the actual location
#                     if text and "Job Location" not in text:
#                         details["location"] = text
#                         break

#         # Extract job description from iCIMS_InfoMsg sections
#         # Look for sections with specific headers
#         info_sections = (
#             job_content.find_all(
#                 "h2", class_=lambda x: x and "iCIMS_InfoField_Job" in x
#             )
#             if job_content
#             else []
#         )

#         for section_header in info_sections:
#             header_text = self._clean_text(section_header.get_text()).lower()

#             # Get the content div that follows this header
#             content_div = section_header.find_next_sibling(
#                 "div", class_=lambda x: x and "iCIMS_InfoMsg_Job" in x
#             )

#             if content_div:
#                 # Get text from the expandable text container
#                 expandable_text = content_div.find(
#                     "div", class_=lambda x: x and "iCIMS_Expandable_Text" in x
#                 )
#                 if expandable_text:
#                     section_content = self._clean_text(expandable_text.get_text())

#                     # Categorize based on header text
#                     if "job summary" in header_text or "description" in header_text:
#                         details["job_description"] = section_content
#                     elif "requirement" in header_text or "qualification" in header_text:
#                         details["requirements"] = section_content
#                     elif "benefit" in header_text or "what we offer" in header_text:
#                         details["benefits"] = section_content

#         # If we didn't find structured sections, try to extract from all info messages
#         if not details["job_description"]:
#             info_msgs = (
#                 job_content.find_all(
#                     "div", class_=lambda x: x and "iCIMS_InfoMsg_Job" in x
#                 )
#                 if job_content
#                 else []
#             )
#             if info_msgs:
#                 # Use the first info message as job description
#                 first_msg = info_msgs[0]
#                 expandable_text = first_msg.find(
#                     "div", class_=lambda x: x and "iCIMS_Expandable_Text" in x
#                 )
#                 if expandable_text:
#                     details["job_description"] = self._clean_text(
#                         expandable_text.get_text()
#                     )

#         # Salary - look for salary information in the content
#         salary_keywords = ["salary", "compensation", "pay", "hourly", "annual", "$"]
#         for keyword in salary_keywords:
#             element = soup.find(
#                 text=lambda text: text and keyword.lower() in text.lower()
#             )
#             if element:
#                 parent = element.parent
#                 if parent:
#                     text = self._clean_text(parent.get_text())
#                     if "$" in text or any(c.isdigit() for c in text):
#                         details["salary"] = text
#                         break

#         return details


class LevelsFYIAdapter(BaseScraperAdapter):
    """Adapter for Levels.fyi job boards."""

    def __init__(self):
        super().__init__()
        self.domain = "levels.fyi"

    def can_handle(self, url: str) -> bool:
        # Only handle URLs that are exactly /jobs or /jobs with query parameters
        # Exclude /jobs/title, /jobs/company, etc.
        if "levels.fyi" not in url:
            return False

        # Extract the path from the URL
        from urllib.parse import urlparse

        parsed = urlparse(url)
        path = parsed.path

        # Only match /jobs or /jobs/ (with or without query parameters)
        # Exclude paths like /jobs/title, /jobs/company, etc.
        if path == "/jobs" or path == "/jobs/":
            return True

        # Also allow /jobs with query parameters (e.g., /jobs?jobId=123)
        if path.startswith("/jobs") and (path == "/jobs" or "?" in url):
            # Check if there's a path segment after /jobs
            path_parts = path.rstrip("/").split("/")
            if len(path_parts) == 2 and path_parts[1] == "jobs":
                return True

        return False

    def scrape(self, soup: BeautifulSoup, url: str) -> Dict:
        details = {
            "job_description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "company": "",
            "salary": "",
        }

        # Levels.fyi job pages have a specific structure
        # Check if this is a job page (has jobId parameter)
        is_job_page = "jobId" in url

        if is_job_page:
            # Extract company name and location from the details row
            # Format: "Hightouch · 2 months ago · San Francisco, California, United States · Fully Remote"
            details_row = soup.find("p", class_=lambda x: x and "detailsRow" in x)
            if details_row:
                text = self._clean_text(details_row.get_text())
                # Split by "·" to get parts
                parts = [p.strip() for p in text.split("·")]
                if len(parts) >= 3:
                    # First part is company name
                    details["company"] = parts[0]
                    # Third part is location (might include remote info)
                    if len(parts) >= 3:
                        location_part = parts[2]
                        # Remove "Fully Remote" if present
                        details["location"] = location_part.replace(
                            "Fully Remote", ""
                        ).strip()

            # Extract salary from compensation row
            compensation_row = soup.find(
                "div", class_=lambda x: x and "compensationRow" in x
            )
            if compensation_row:
                # Get the text and remove the "(base salary from job description)" part
                salary_text = self._clean_text(compensation_row.get_text())
                # Remove the parenthetical note
                salary_text = re.sub(r"\(.*?\)", "", salary_text).strip()
                if salary_text and (
                    "$" in salary_text or any(c.isdigit() for c in salary_text)
                ):
                    details["salary"] = salary_text

            # Extract job description from the about section
            # Look for the job-details-about section
            about_section = soup.find(
                "section", class_=lambda x: x and "aboutContainer" in x
            )
            if about_section:
                # Get the markdown text container
                markdown_text = about_section.find(
                    "div", class_=lambda x: x and "markdownText" in x
                )
                if markdown_text:
                    details["job_description"] = self._clean_text(
                        markdown_text.get_text()
                    )

            # If no description found in about section, try other selectors
            if not details["job_description"]:
                desc_selectors = [
                    'div[class*="job-description"]',
                    'div[data-testid="job-description"]',
                    'div[class*="description"]',
                    'section[class*="description"]',
                    'div[class*="job-details"]',
                ]

                for selector in desc_selectors:
                    desc_elem = soup.select_one(selector)
                    if desc_elem:
                        details["job_description"] = self._clean_text(
                            desc_elem.get_text()
                        )
                        break

            # If still no description, try to get the largest text block
            if not details["job_description"]:
                text_blocks = soup.find_all(["div", "section", "article"])
                if text_blocks:
                    # Filter out very short blocks and navigation elements
                    valid_blocks = [
                        b
                        for b in text_blocks
                        if len(b.get_text()) > 100
                        and "nav" not in str(b.get("class", ""))
                    ]
                    if valid_blocks:
                        largest_block = max(
                            valid_blocks, key=lambda x: len(x.get_text())
                        )
                        details["job_description"] = self._clean_text(
                            largest_block.get_text()
                        )

            # Extract requirements - look for section with requirements/qualifications
            req_keywords = ["requirement", "qualification", "skill", "what you need"]
            for keyword in req_keywords:
                element = soup.find(
                    text=lambda text: text and keyword.lower() in text.lower()
                )
                if element:
                    parent = element.parent
                    if parent:
                        details["requirements"] = self._clean_text(parent.get_text())
                        break

            # Extract benefits - look for section with benefits/perks
            benefit_keywords = ["benefit", "perk", "what we offer", "compensation"]
            for keyword in benefit_keywords:
                element = soup.find(
                    text=lambda text: text and keyword.lower() in text.lower()
                )
                if element:
                    parent = element.parent
                    if parent:
                        details["benefits"] = self._clean_text(parent.get_text())
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
        # self.register_adapter(ICIMSAdapter())
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
