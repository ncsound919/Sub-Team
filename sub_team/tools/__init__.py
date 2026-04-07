"""
Sub-Team Tools — MCP-integrated tool adapters for the agentic workforce.

Provides CrewAI-compatible tools that wrap external capabilities:
  - Web search (DuckDuckGo)
  - Web scraping (Firecrawl)
  - Code execution (E2B sandbox)
  - File operations (local filesystem)
  - Data analysis (pandas)
  - GitHub operations (gh CLI)
"""

from .web_search import WebSearchTool
from .web_scraper import WebScraperTool
from .code_executor import CodeExecutorTool
from .file_ops import FileReadTool, FileWriteTool, DirectoryListTool
from .data_analysis import DataAnalysisTool
from .github_ops import GitHubSearchTool, GitHubRepoInfoTool
from .shell_exec import ShellExecTool

__all__ = [
    "WebSearchTool",
    "WebScraperTool",
    "CodeExecutorTool",
    "FileReadTool",
    "FileWriteTool",
    "DirectoryListTool",
    "DataAnalysisTool",
    "GitHubSearchTool",
    "GitHubRepoInfoTool",
    "ShellExecTool",
]
