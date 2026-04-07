"""
Web Search Tool — DuckDuckGo-backed search for agents.

Uses the duckduckgo-search library (no API key required) to give agents
the ability to search the web for current information.
"""

from __future__ import annotations

import logging
from typing import Optional

from crewai.tools import BaseTool
from pydantic import Field

_log = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo. Returns top results with titles, URLs, and snippets."""

    name: str = "web_search"
    description: str = (
        "Search the web for current information on any topic. "
        "Input should be a search query string. Returns titles, URLs, and snippets "
        "from the top results. Use this when you need up-to-date information, "
        "to research a topic, or to find specific resources online."
    )
    max_results: int = Field(default=8, description="Maximum number of results to return")

    def _run(self, query: str) -> str:
        """Execute a web search and return formatted results."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"

        if not query or not query.strip():
            return "Error: Empty search query provided."

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            if not results:
                return f"No results found for: {query}"

            lines = [f"## Search Results for: {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                href = r.get("href", r.get("link", ""))
                body = r.get("body", r.get("snippet", ""))
                lines.append(f"### {i}. {title}")
                if href:
                    lines.append(f"**URL:** {href}")
                if body:
                    lines.append(f"{body}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            _log.warning("Web search failed for query=%s: %s", query, e)
            return f"Search error: {e}"
