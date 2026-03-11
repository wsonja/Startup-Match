#!/usr/bin/env python3
"""
Scrape startup websites for careers/job pages, collect posting text,
extract skills from a predefined vocabulary, and save aggregated_skills.

Usage:
    python scrape_jobs.py --input init.json --output enriched_init.json

Expected input format:
{
  "startups": [
    {
      "company_id": 1,
      "company_name": "Example",
      "website": "https://example.com",
      ...
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# -------------------------
# Config
# -------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT = 12
SLEEP_BETWEEN_COMPANIES = 1.0
MAX_INTERNAL_LINKS_TO_SCAN = 40
MAX_JOB_LINKS_TO_SCAN = 20
MIN_TEXT_LENGTH = 120

CAREERS_KEYWORDS = [
    "careers", "career", "jobs", "job", "join-us", "join_us", "joinus",
    "hiring", "work-with-us", "work_with_us", "open-roles", "open_roles",
    "openings", "positions", "opportunities", "team"
]

JOB_POSTING_KEYWORDS = [
    "software engineer", "frontend", "backend", "full stack", "machine learning",
    "data scientist", "product manager", "designer", "intern", "analyst",
    "research engineer", "applied scientist", "devops", "infrastructure",
    "security engineer", "ai engineer"
]

# Predefined skill vocabulary for extraction.
# Expand this over time.
SKILL_VOCAB = {
    "python": [r"\bpython\b"],
    "java": [r"\bjava\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b"],
    "typescript": [r"\btypescript\b", r"\bts\b"],
    "react": [r"\breact\b", r"\breact\.js\b"],
    "node.js": [r"\bnode\.?js\b"],
    "next.js": [r"\bnext\.?js\b"],
    "html": [r"\bhtml\b"],
    "css": [r"\bcss\b"],
    "sql": [r"\bsql\b"],
    "postgresql": [r"\bpostgres(?:ql)?\b"],
    "mysql": [r"\bmysql\b"],
    "mongodb": [r"\bmongodb\b"],
    "redis": [r"\bredis\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle cloud\b"],
    "azure": [r"\bazure\b"],
    "docker": [r"\bdocker\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "git": [r"\bgit\b"],
    "linux": [r"\blinux\b"],
    "bash": [r"\bbash\b"],
    "c": [r"(?<!\+\+)\bc\b(?!\+\+)"],
    "c++": [r"\bc\+\+\b"],
    "go": [r"\bgo\b", r"\bgolang\b"],
    "rust": [r"\brust\b"],
    "pytorch": [r"\bpytorch\b"],
    "tensorflow": [r"\btensorflow\b"],
    "scikit-learn": [r"\bscikit[- ]learn\b", r"\bsklearn\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep learning": [r"\bdeep learning\b"],
    "nlp": [r"\bnlp\b", r"\bnatural language processing\b"],
    "llms": [r"\bllms?\b", r"\blarge language models?\b"],
    "computer vision": [r"\bcomputer vision\b"],
    "fastapi": [r"\bfastapi\b"],
    "flask": [r"\bflask\b"],
    "django": [r"\bdjango\b"],
    "apis": [r"\bapi\b", r"\bapis\b"],
    "data analysis": [r"\bdata analysis\b", r"\banalytics\b"],
    "tableau": [r"\btableau\b"],
    "looker": [r"\blooker\b"],
    "spark": [r"\bspark\b", r"\bpyspark\b"],
    "airflow": [r"\bairflow\b"],
    "graphql": [r"\bgraphql\b"],
}

# -------------------------
# Data classes
# -------------------------

@dataclass
class ScrapeResult:
    careers_urls: List[str]
    job_posting_urls: List[str]
    job_posting_text: str
    aggregated_skills: List[str]
    skill_counts: Dict[str, int]


# -------------------------
# Helpers
# -------------------------

def fetch_url(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def same_domain(base_url: str, candidate_url: str) -> bool:
    try:
        return urlparse(base_url).netloc.replace("www.", "") == urlparse(candidate_url).netloc.replace("www.", "")
    except Exception:
        return False


def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text

def is_valid_href(href: str) -> bool:
    if not href:
        return False

    href = href.strip()
    lowered = href.lower()

    # skip obvious junk
    if lowered in {"#", "/", ""}:
        return False

    if lowered.startswith(("mailto:", "tel:", "javascript:")):
        return False

    # skip placeholder/template links
    if "[" in href or "]" in href or "{" in href or "}" in href:
        return False

    if "godaddy_link" in lowered:
        return False

    return True

def get_internal_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if not is_valid_href(href):
            continue

        try:
            full = urljoin(base_url, href)
        except Exception:
            continue
        if same_domain(base_url, full):
            links.add(full.split("#")[0])

    return list(links)


def looks_like_careers_link(url: str, anchor_text: str = "") -> bool:
    text = f"{url} {anchor_text}".lower()
    return any(keyword in text for keyword in CAREERS_KEYWORDS)


def looks_like_job_posting(url: str, text: str = "") -> bool:
    haystack = f"{url} {text}".lower()
    return (
        any(keyword in haystack for keyword in JOB_POSTING_KEYWORDS)
        or "/jobs/" in haystack
        or "/careers/" in haystack
        or "greenhouse.io" in haystack
        or "lever.co" in haystack
        or "ashbyhq.com" in haystack
        or "workable.com" in haystack
    )


def discover_career_pages(base_url: str) -> List[str]:
    candidate_urls: Set[str] = set()

    try:
        html = fetch_url(base_url)
    except Exception:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Direct anchor scan on homepage
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        anchor_text = a.get_text(" ", strip=True)

        if not is_valid_href(href):
            continue

        try:
            full = urljoin(base_url, href)
        except Exception:
            continue

        if looks_like_careers_link(full, anchor_text):
            candidate_urls.add(full.split("#")[0])

    # Common fallback paths
    common_paths = [
        "/careers", "/career", "/jobs", "/hiring", "/join-us",
        "/about/careers", "/company/careers", "/careers/"
    ]
    for path in common_paths:
        candidate_urls.add(urljoin(base_url, path))

    # Keep only same-domain or known ATS pages discovered from site links
    cleaned = []
    for url in candidate_urls:
        if same_domain(base_url, url) or any(host in url for host in ["greenhouse.io", "lever.co", "ashbyhq.com", "workable.com"]):
            cleaned.append(url)

    return sorted(set(cleaned))


def extract_job_links_from_page(page_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True)

        if not is_valid_href(href):
            continue

        try:
            full = urljoin(page_url, href)
        except Exception:
            continue

        if looks_like_job_posting(full, text):
            links.add(full.split("#")[0])

    return sorted(links)


def collect_job_posting_text(job_urls: List[str]) -> Tuple[List[str], str]:
    collected_urls: List[str] = []
    text_chunks: List[str] = []

    for url in job_urls[:MAX_JOB_LINKS_TO_SCAN]:
        try:
            html = fetch_url(url)
            text = extract_visible_text(html)
            if len(text) >= MIN_TEXT_LENGTH:
                collected_urls.append(url)
                text_chunks.append(text)
        except Exception:
            continue

    return collected_urls, "\n\n".join(text_chunks)


def extract_skills(text: str, skill_vocab: Dict[str, List[str]]) -> Dict[str, int]:
    counts: Counter = Counter()
    lowered = text.lower()

    for canonical_skill, patterns in skill_vocab.items():
        for pattern in patterns:
            matches = re.findall(pattern, lowered)
            if matches:
                counts[canonical_skill] += len(matches)

    return dict(counts)


def scrape_company_jobs(website: str) -> ScrapeResult:
    website = normalize_url(website)
    if not website:
        return ScrapeResult([], [], "", [], {})

    careers_urls = discover_career_pages(website)
    discovered_job_urls: Set[str] = set()
    job_text_chunks: List[str] = []

    for careers_url in careers_urls[:10]:
        try:
            html = fetch_url(careers_url)
            links = extract_job_links_from_page(careers_url, html)

            # If careers page itself has useful text, keep it too
            careers_text = extract_visible_text(html)
            if len(careers_text) >= MIN_TEXT_LENGTH:
                job_text_chunks.append(careers_text)

            for link in links:
                discovered_job_urls.add(link)

            # Also scan a few internal links if careers page is generic
            internal_links = get_internal_links(careers_url, html)[:MAX_INTERNAL_LINKS_TO_SCAN]
            for link in internal_links:
                if looks_like_job_posting(link, link):
                    discovered_job_urls.add(link)

        except Exception:
            continue

    final_job_urls, posting_text = collect_job_posting_text(sorted(discovered_job_urls))
    all_text = "\n\n".join(job_text_chunks + [posting_text]).strip()

    skill_counts = extract_skills(all_text, SKILL_VOCAB)
    aggregated_skills = sorted(skill_counts.keys())

    return ScrapeResult(
        careers_urls=careers_urls,
        job_posting_urls=final_job_urls,
        job_posting_text=all_text,
        aggregated_skills=aggregated_skills,
        skill_counts=skill_counts,
    )


def build_company_document(company: dict) -> str:
    """
    Combined text field for TF-IDF retrieval.
    """
    fields = [
        company.get("company_name", ""),
        company.get("short_description", ""),
        company.get("long_description", ""),
        " ".join(company.get("tags", []) or []),
        company.get("location", ""),
        company.get("country", ""),
        " ".join(company.get("aggregated_skills", []) or []),
    ]
    return " ".join(str(x) for x in fields if x).strip()


def enrich_companies(data: dict) -> dict:
    startups = data.get("startups") or data.get("companies") or []
    enriched = []

    for idx, company in enumerate(startups, start=1):
        name = company.get("company_name", f"company_{idx}")
        website = company.get("website", "")

        print(f"[{idx}/{len(startups)}] Scraping {name} -> {website}")

        result = scrape_company_jobs(website)

        company_copy = dict(company)
        company_copy["careers_urls"] = result.careers_urls
        company_copy["job_posting_urls"] = result.job_posting_urls
        company_copy["job_posting_text"] = result.job_posting_text
        company_copy["aggregated_skills"] = result.aggregated_skills
        company_copy["skill_counts"] = result.skill_counts
        company_copy["retrieval_document"] = build_company_document(company_copy)

        enriched.append(company_copy)
        time.sleep(SLEEP_BETWEEN_COMPANIES)

    if "startups" in data:
        return {"startups": enriched}
    return {"companies": enriched}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched_data = enrich_companies(data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    print(f"\nWrote enriched data to {args.output}")


if __name__ == "__main__":
    main()