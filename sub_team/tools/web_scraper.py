"""
Web Scraper Tool — Firecrawl-backed web scraping for agents.

Extracts clean markdown content from any URL. Falls back to a simple
requests + markdownify approach if Firecrawl is unavailable.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from typing import Optional
from urllib.parse import urlparse

from crewai.tools import BaseTool

_log = logging.getLogger(__name__)


def _is_safe_url(url: str) -> bool:
    """Block URLs targeting private/internal/metadata IP ranges (SSRF protection)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False

        # Block common cloud metadata endpoints
        if hostname in (
            "169.254.169.254",
            "metadata.google.internal",
            "metadata.internal",
        ):
            return False

        # Resolve hostname and check for private IPs
        import socket

        try:
            addr_infos = socket.getaddrinfo(hostname, None)
            for _family, _type, _proto, _canonname, sockaddr in addr_infos:
                ip = ipaddress.ip_address(sockaddr[0])
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                ):
                    return False
        except socket.gaierror:
            return False  # Can't resolve — block it

        return True
    except Exception:
        return False


class WebScraperTool(BaseTool):
    """Scrape and extract clean content from a web page URL."""

    name: str = "web_scraper"
    description: str = (
        "Extract the main content from a web page as clean markdown. "
        "Input should be a full URL (https://...). Use this when you need to "
        "read the actual content of a specific web page, documentation, article, "
        "or GitHub README."
    )

    def _run(self, url: str) -> str:
        """Scrape a URL and return markdown content."""
        if not url or not url.strip():
            return "Error: No URL provided."

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        # SSRF protection — block private/internal/metadata URLs
        if not _is_safe_url(url):
            return "Error: URL targets a private, internal, or metadata endpoint. Access denied."

        # Try Firecrawl first (higher quality)
        firecrawl_key = os.environ.get("FIRECRAWL_API_KEY")
        if firecrawl_key:
            try:
                return self._scrape_firecrawl(url, firecrawl_key)
            except Exception as e:
                _log.debug("Firecrawl failed, falling back: %s", e)

        # Fallback: requests + markdownify
        return self._scrape_simple(url)

    def _scrape_firecrawl(self, url: str, api_key: str) -> str:
        """Scrape using Firecrawl API for high-quality extraction."""
        from firecrawl import FirecrawlApp

        app = FirecrawlApp(api_key=api_key)
        result = app.scrape_url(url, params={"formats": ["markdown"]})

        if isinstance(result, dict):
            md = result.get("markdown", "")
            if md:
                return md[:15000]  # Truncate to avoid token overflow
            return f"Firecrawl returned no markdown for {url}"

        return str(result)[:15000]

    def _scrape_simple(self, url: str) -> str:
        """Fallback scraper using requests + markdownify."""
        try:
            import requests
            from markdownify import markdownify as md
        except ImportError:
            return "Error: requests and markdownify packages required for fallback scraping."

        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "SubTeam-Agent/1.0"},
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"Non-HTML content type: {content_type}. Cannot extract text."

            markdown = md(resp.text, strip=["script", "style", "nav", "footer"])
            # Clean up excessive whitespace
            lines = [line.strip() for line in markdown.splitlines()]
            cleaned = "\n".join(line for line in lines if line)
            return cleaned[:15000]

        except requests.RequestException as e:
            return f"Failed to fetch {url}: {e}"
