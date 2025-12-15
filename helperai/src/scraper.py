"""Simple web scraper for public pages."""
from __future__ import annotations

import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "HelperAI/0.1 (+https://example.com)"}


def scrape_web(url: str, timeout: int = 10) -> str:
    """Fetch and extract visible paragraph text from a public web page.

    Args:
        url: Public URL to scrape.
        timeout: Request timeout in seconds.

    Returns:
        Concatenated text content from paragraph tags.
    """

    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)

    logger.info("Scraped %d paragraphs from %s", len(paragraphs), url)
    return text


def save_raw_text(text: str, path: str) -> None:
    """Save scraped text to a file for inspection."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.debug("Saved raw text to %s", path)


__all__ = ["scrape_web", "save_raw_text"]
