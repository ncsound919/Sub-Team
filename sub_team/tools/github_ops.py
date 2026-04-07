"""
GitHub Operations Tools — Repository search and info via GitHub API.

Uses the GitHub REST API (with optional GITHUB_TOKEN for higher rate limits)
to search repos, read READMEs, and get repository metadata.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests
from crewai.tools import BaseTool
from pydantic import Field

_log = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"


def _github_headers() -> dict:
    """Build GitHub API headers with optional auth token."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class GitHubSearchTool(BaseTool):
    """Search GitHub repositories by topic, language, or keyword."""

    name: str = "github_search"
    description: str = (
        "Search GitHub for repositories matching a query. "
        "Input should be a search query (e.g., 'multi-agent framework python', "
        "'react component library stars:>1000'). Returns repo names, descriptions, "
        "star counts, and URLs. Use GitHub search qualifiers for precision: "
        "language:python, stars:>1000, topic:ai-agents, etc."
    )
    max_results: int = Field(default=10, description="Max repositories to return")

    def _run(self, query: str) -> str:
        if not query or not query.strip():
            return "Error: No search query provided."

        try:
            resp = requests.get(
                f"{_GITHUB_API}/search/repositories",
                params={
                    "q": query.strip(),
                    "sort": "stars",
                    "order": "desc",
                    "per_page": self.max_results,
                },
                headers=_github_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("items", [])
            if not items:
                return f"No repositories found for: {query}"

            lines = [f"## GitHub Search: {query} ({data.get('total_count', 0):,} total)\n"]
            for i, repo in enumerate(items, 1):
                name = repo.get("full_name", "")
                desc = repo.get("description", "No description") or "No description"
                stars = repo.get("stargazers_count", 0)
                lang = repo.get("language", "Unknown")
                url = repo.get("html_url", "")
                updated = repo.get("updated_at", "")[:10]
                topics = ", ".join(repo.get("topics", [])[:5])

                lines.append(f"### {i}. {name} ({stars:,} stars)")
                lines.append(f"**Language:** {lang} | **Updated:** {updated}")
                if topics:
                    lines.append(f"**Topics:** {topics}")
                lines.append(f"**URL:** {url}")
                lines.append(f"{desc}")
                lines.append("")

            return "\n".join(lines)

        except requests.RequestException as e:
            return f"GitHub API error: {e}"


class GitHubRepoInfoTool(BaseTool):
    """Get detailed information about a specific GitHub repository."""

    name: str = "github_repo_info"
    description: str = (
        "Get detailed info about a GitHub repository including README, "
        "stats, and structure. Input should be 'owner/repo' format "
        "(e.g., 'crewAIInc/crewAI'). Returns star count, description, "
        "language breakdown, recent activity, and README content."
    )

    def _run(self, repo: str) -> str:
        if not repo or "/" not in repo.strip():
            return "Error: Input must be 'owner/repo' format."

        repo = repo.strip()
        headers = _github_headers()

        try:
            # Get repo metadata
            resp = requests.get(
                f"{_GITHUB_API}/repos/{repo}",
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            info = resp.json()

            lines = [
                f"## {info.get('full_name', repo)}",
                f"**Description:** {info.get('description', 'None')}",
                f"**Stars:** {info.get('stargazers_count', 0):,} | "
                f"**Forks:** {info.get('forks_count', 0):,} | "
                f"**Open Issues:** {info.get('open_issues_count', 0):,}",
                f"**Language:** {info.get('language', 'Unknown')}",
                f"**License:** {info.get('license', {}).get('name', 'None') if info.get('license') else 'None'}",
                f"**Created:** {info.get('created_at', '')[:10]} | "
                f"**Updated:** {info.get('updated_at', '')[:10]}",
                f"**URL:** {info.get('html_url', '')}",
                "",
            ]

            # Try to get README
            try:
                readme_resp = requests.get(
                    f"{_GITHUB_API}/repos/{repo}/readme",
                    headers={**headers, "Accept": "application/vnd.github.raw+json"},
                    timeout=15,
                )
                if readme_resp.status_code == 200:
                    readme_text = readme_resp.text[:8000]
                    lines.append("### README (truncated)")
                    lines.append(readme_text)
            except Exception:
                lines.append("*README not available*")

            return "\n".join(lines)

        except requests.RequestException as e:
            return f"GitHub API error for {repo}: {e}"
