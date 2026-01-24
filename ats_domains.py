"""
Hardcoded list of supported ATS domains with their configurations.
Each domain has specific scraping and verification requirements.
"""

# Supported ATS domains with their configurations
SUPPORTED_ATS_DOMAINS = {
    # "jobs.ashbyhq.com": {
    #     "name": "Ashby",
    #     "description": "Ashby ATS",
    #     "scraping_adapter": "AshbyAdapter",
    #     "expired_selectors": [
    #         {
    #             "selector": "h1[class*='_title']",
    #             "text": "job not found",
    #             "case_sensitive": False,
    #         },
    #         {
    #             "selector": "h1[class*='_title']",
    #             "text": "position not found",
    #             "case_sensitive": False,
    #         },
    #     ],
    # },
    "boards.greenhouse.io": {
        "name": "Greenhouse",
        "description": "Greenhouse ATS",
        "scraping_adapter": "GreenhouseAdapter",
        "expired_selectors": [
            {
                "selector": "main.main.font-secondary div.index--content h1.page-header.font-primary",
                "text_pattern": "current openings at",
                "case_sensitive": False,
                "pattern_type": "contains",
            },
            {
                "selector": "body div#wrapper div#main p",
                "text": "sorry, but we can't find that page",
                "case_sensitive": False,
            },
        ],
    },
    "jobs.lever.co": {
        "name": "Lever",
        "description": "Lever ATS",
        "scraping_adapter": "LeverAdapter",
        "expired_selectors": [
            {
                "selector": "body.404 div.content-wrapper.posting-page div.content div.section-wrapper.accent-section.page-full-width div.section.narrow-section.page-centered.posting-header div.error-message h2",
                "text": "sorry, we couldn't find anything here",
                "case_sensitive": False,
            },
        ],
    },
    "jobs.smartrecruiters.com": {
        "name": "SmartRecruiters",
        "description": "SmartRecruiters ATS",
        "scraping_adapter": "SmartRecruitersAdapter",
        "expired_selectors": [
            {
                "selector": "body div.jobad.site div.wrapper div.grid div.column.jobad-container.wide-9of16.medium-5of8.print-block.equal-column main.jobad-main.job div.jobad--empty-state h2.font--primary.text--center.margin--bottom--l",
                "text": "this job has expired",
                "case_sensitive": False,
            },
        ],
    },
    "jobs.workable.com": {
        "name": "Workable",
        "description": "Workable ATS",
        "scraping_adapter": "WorkableAdapter",
        "expired_selectors": [
            {
                "selector": "div#app div.jobUnavailable__container h1 strong",
                "text": "This job is not available anymore",
                "case_sensitive": False,
            },
        ],
    },
    "wd1.myworkdayjobs.com": {
        "name": "Workday",
        "description": "Workday ATS",
        "scraping_adapter": "WorkdayAdapter",
        "expired_selectors": [
            {
                "selector": "body div#root div div div div#mainContent div div div span[data-automation-id='errorMessage']",
                "case_sensitive": False,
            },
        ],
    },
    "jobs.jobvite.com": {
        "name": "Jobvite",
        "description": "Jobvite ATS",
        "scraping_adapter": "JobviteAdapter",
        "expired_selectors": [
            {
                "selector": "html.js.supports.cssanimations.csstransforms.csstransforms3d body.jv-desktop.jv-page-jobs.ng-scope div.jv-page-container div.jv-page div.jv-page-content div.jv-page-error p.jv-page-error-header",
                "text": "the job listing no longer exists",
                "case_sensitive": False,
            },
        ],
    },
    "careers.icims.com": {
        "name": "iCIMS",
        "description": "iCIMS ATS",
        "scraping_adapter": "IcimsAdapter",
        "expired_selectors": [
            {
                "selector": "html#ng-app.no-js.rms-node.overthrow-enabled body.careers-home.external div#all-content.snap-content div.jibe-container.not-found-container div.not-found-content h2",
                "text": "the page you are looking for no longer exists",
                "case_sensitive": False,
            },
        ],
    },
    "www.levels.fyi": {
        "name": "Levels.fyi",
        "description": "Levels.fyi Job Board",
        "scraping_adapter": "LevelsFYIAdapter",
        "expired_selectors": [
            {
                "selector": "body div#__next div div div div section section div h1",
                "text": "oops! this job has expired",
                "case_sensitive": False,
            },
        ],
    },
}


def get_supported_domains():
    """Get list of all supported ATS domains."""
    return list(SUPPORTED_ATS_DOMAINS.keys())


def get_domain_config(domain: str):
    """Get configuration for a specific ATS domain."""
    return SUPPORTED_ATS_DOMAINS.get(domain)


def get_expired_indicators(domain: str):
    """Get expired job indicators for a specific ATS domain."""
    config = get_domain_config(domain)
    if config:
        return config.get("expired_indicators", [])
    return []


def get_scraping_adapter_name(domain: str):
    """Get the scraping adapter class name for a specific ATS domain."""
    config = get_domain_config(domain)
    if config:
        return config.get("scraping_adapter")
    return None
