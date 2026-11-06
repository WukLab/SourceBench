"""URL helpers shared by the source-collection scripts.

`canonicalize_url` and `get_domain_category` match the helpers used to build
`data/collected-sources/`, so new runs deduplicate URLs and assign domain
categories the same way.
"""

from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import parse_qsl, urlparse, urlunparse


def canonicalize_url(url: str) -> str:
    """Return a normalized URL string used for deduplication.

    Lowercases the host, drops the fragment and any trailing slash, and sorts
    query parameters.
    """
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        if not path.startswith("/"):
            path = f"/{path}"
        query_parts = sorted(parse_qsl(parsed.query, keep_blank_values=True))
        query = "&".join(f"{key}={value}" for key, value in query_parts)
        return urlunparse((scheme, netloc, path.rstrip("/"), "", query, ""))
    except Exception:  # pylint: disable=broad-except
        return url.strip().lower()


def get_domain_category(url: str) -> str:
    """Return the coarse domain category stored as `domain_category`.

    The scoring rubric reads this field (e.g. `.gov` and `.edu` for authority).
    """
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        social_domains = [
            "facebook.com",
            "twitter.com",
            "x.com",
            "instagram.com",
            "linkedin.com",
            "youtube.com",
            "tiktok.com",
            "reddit.com",
        ]
        if any(social in domain for social in social_domains):
            return "social_media"
        if domain.endswith(".edu"):
            return ".edu"
        if domain.endswith(".gov") or domain.endswith(".gov.uk"):
            return ".gov"
        if domain.endswith(".org"):
            return ".org"
        if domain.endswith(".com"):
            return ".com"
        news_keywords = ["news", "media", "bbc", "cnn", "nytimes", "reuters"]
        if any(keyword in domain for keyword in news_keywords):
            return "news"
        return "other"
    except Exception:  # pylint: disable=broad-except
        return "unknown"


def format_http_date(value: Optional[str]) -> Optional[str]:
    """Convert an HTTP date header (RFC 7231) to YYYY-MM-DD, or None."""
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError, IndexError):
        return None
