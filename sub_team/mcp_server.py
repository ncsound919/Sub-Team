"""
Sub-Team MCP Server — Model Context Protocol adapter for Sub-Team tools.

Exposes Sub-Team's CrewAI tools and workforce capabilities as an MCP server
so that Draymond Orchestrator (or any MCP client) can call them via the
standardized MCP protocol.

Supported transports:
  - stdio  (default, for subprocess-based MCP clients)
  - sse    (for HTTP-based MCP clients)

Usage::

    # Start MCP server (stdio transport — for Draymond subprocess integration)
    python -m sub_team.mcp_server

    # Start MCP server (SSE transport — for HTTP-based clients)
    python -m sub_team.mcp_server --transport sse --port 8051

Tools exposed:
  - web_search          : Search the web via DuckDuckGo
  - web_scrape          : Scrape and extract content from a URL
  - execute_code        : Run Python/JS code in sandboxed environment
  - read_file           : Read file contents
  - write_file          : Write content to a file
  - list_directory      : List directory contents
  - analyze_data        : Run pandas-based data analysis
  - github_search       : Search GitHub repositories and code
  - shell_exec          : Execute shell commands
  - run_task            : Execute a full agentic task via the workforce
  - list_capabilities   : List all available agents and task types

Resources exposed:
  - sub-team://capabilities  : Agent and task type catalog
  - sub-team://memory/{id}   : Agent memory namespace
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_log = logging.getLogger(__name__)


def create_mcp_server():
    """Create and configure the MCP server with all Sub-Team tools."""
    from mcp.server import Server
    from mcp.types import (
        TextContent,
        Tool,
        Resource,
    )

    server = Server("sub-team")

    # ────────────────────────────────────────────────────────────────
    # Tool Definitions
    # ────────────────────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="web_search",
                description="Search the web for current information using DuckDuckGo. Returns titles, URLs, and snippets.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 8)",
                            "default": 8,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="web_scrape",
                description="Scrape and extract content from a URL. Returns markdown-formatted content.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to scrape",
                        },
                    },
                    "required": ["url"],
                },
            ),
            Tool(
                name="execute_code",
                description="Execute Python or JavaScript code in a sandboxed environment and return the output.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute. Prefix with ```python or ```javascript to specify language.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max execution time in seconds (default: 30)",
                            "default": 30,
                        },
                    },
                    "required": ["code"],
                },
            ),
            Tool(
                name="read_file",
                description="Read the contents of a file at the given path.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="write_file",
                description="Write content to a file at the given path.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to write to",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
            ),
            Tool(
                name="list_directory",
                description="List the contents of a directory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list",
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="analyze_data",
                description="Run pandas-based data analysis. Provide a file path or inline data with an analysis query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to analyze (e.g., 'summarize', 'correlations', 'top 10 by revenue')",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to a CSV/JSON/Excel file to analyze",
                        },
                        "data": {
                            "type": "string",
                            "description": "Inline CSV data to analyze (alternative to file_path)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="github_search",
                description="Search GitHub for repositories, code, or issues.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for GitHub",
                        },
                        "search_type": {
                            "type": "string",
                            "description": "Type of search: 'repositories', 'code', or 'issues'",
                            "enum": ["repositories", "code", "issues"],
                            "default": "repositories",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="shell_exec",
                description="Execute a shell command and return the output. Use with caution.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max execution time in seconds (default: 30)",
                            "default": 30,
                        },
                    },
                    "required": ["command"],
                },
            ),
            Tool(
                name="run_task",
                description=(
                    "Execute a full agentic task using the Sub-Team workforce. "
                    "Routes the objective to the best-fit agents automatically. "
                    "Supports 30+ task types including research, coding, data science, "
                    "business strategy, security, and cross-disciplinary work."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "objective": {
                            "type": "string",
                            "description": "What the agents should accomplish",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Explicit task type (auto-classified if omitted)",
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Write output to this file",
                        },
                    },
                    "required": ["objective"],
                },
            ),
            Tool(
                name="list_capabilities",
                description="List all Sub-Team agents, task types, and supported capabilities.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    # ────────────────────────────────────────────────────────────────
    # Tool Handlers
    # ────────────────────────────────────────────────────────────────

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Route tool calls to the appropriate handler."""

        try:
            result = _handle_tool(name, arguments)
            return [TextContent(type="text", text=result)]
        except Exception as e:
            _log.error("Tool %s failed: %s", name, e, exc_info=True)
            return [
                TextContent(type="text", text=f"Error in {name}: {type(e).__name__}")
            ]

    # ────────────────────────────────────────────────────────────────
    # Resources
    # ────────────────────────────────────────────────────────────────

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="sub-team://capabilities",
                name="Sub-Team Capabilities",
                description="Full catalog of agents, task types, and capabilities",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if uri == "sub-team://capabilities":
            from sub_team.crews import AgentRole, TaskType

            return json.dumps(
                {
                    "agents": [r.value for r in AgentRole],
                    "task_types": [t.value for t in TaskType],
                    "total_agents": len(AgentRole),
                    "total_task_types": len(TaskType),
                },
                indent=2,
            )

        # Memory resource: sub-team://memory/{agent_id}
        if uri.startswith("sub-team://memory/"):
            agent_id = uri.split("sub-team://memory/")[1]
            from sub_team.memory import get_memory

            memory = get_memory()
            results = memory.get_all(agent_id=agent_id)
            return json.dumps({"agent_id": agent_id, "memories": results}, indent=2)

        return json.dumps({"error": f"Unknown resource: {uri}"})

    return server


# ────────────────────────────────────────────────────────────────────────
# Tool dispatch (sync wrappers)
# ────────────────────────────────────────────────────────────────────────


def _handle_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call to the appropriate CrewAI tool."""

    if name == "web_search":
        from sub_team.tools import WebSearchTool

        tool = WebSearchTool(max_results=arguments.get("max_results", 8))
        return tool._run(arguments["query"])

    elif name == "web_scrape":
        from sub_team.tools import WebScraperTool

        tool = WebScraperTool()
        return tool._run(arguments["url"])

    elif name == "execute_code":
        from sub_team.tools import CodeExecutorTool

        tool = CodeExecutorTool(timeout_seconds=min(arguments.get("timeout", 30), 300))
        return tool._run(arguments["code"])

    elif name == "read_file":
        from sub_team.tools import FileReadTool

        tool = FileReadTool()
        return tool._run(arguments["path"])

    elif name == "write_file":
        from sub_team.tools import FileWriteTool

        tool = FileWriteTool()
        return tool._run(f"{arguments['path']}|||{arguments['content']}")

    elif name == "list_directory":
        from sub_team.tools import DirectoryListTool

        tool = DirectoryListTool()
        return tool._run(arguments["path"])

    elif name == "analyze_data":
        from sub_team.tools import DataAnalysisTool

        tool = DataAnalysisTool()
        input_str = arguments.get("query", "")
        if arguments.get("file_path"):
            input_str = f"File: {arguments['file_path']}\nQuery: {input_str}"
        elif arguments.get("data"):
            input_str = f"Data:\n{arguments['data']}\nQuery: {input_str}"
        return tool._run(input_str)

    elif name == "github_search":
        from sub_team.tools import GitHubSearchTool

        tool = GitHubSearchTool()
        query = arguments["query"]
        search_type = arguments.get("search_type", "repositories")
        return tool._run(f"{search_type}:{query}")

    elif name == "shell_exec":
        from sub_team.tools import ShellExecTool

        tool = ShellExecTool(timeout_seconds=min(arguments.get("timeout", 30), 300))
        return tool._run(arguments["command"])

    elif name == "run_task":
        from sub_team.entry_points import run_task

        result = run_task(
            objective=arguments["objective"],
            task_type=arguments.get("task_type"),
            output_file=arguments.get("output_file"),
            verbose=False,
        )
        return json.dumps(result, indent=2, default=str)

    elif name == "list_capabilities":
        from sub_team.crews import AgentRole, TaskType

        caps = {
            "agents": [{"role": r.value, "name": r.name} for r in AgentRole],
            "task_types": [t.value for t in TaskType],
            "total_agents": len(AgentRole),
            "total_task_types": len(TaskType),
        }
        return json.dumps(caps, indent=2)

    else:
        raise ValueError(f"Unknown tool: {name}")


# ────────────────────────────────────────────────────────────────────────
# Main — run MCP server
# ────────────────────────────────────────────────────────────────────────


def main():
    """Start the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Sub-Team MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8051,
        help="Port for SSE transport (default: 8051)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=getattr(
            logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO
        ),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    server = create_mcp_server()

    if args.transport == "stdio":
        from mcp.server.stdio import stdio_server
        import asyncio

        async def _run():
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream, server.create_initialization_options()
                )

        asyncio.run(_run())

    elif args.transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn

        sse = SseServerTransport("/messages")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )

        async def handle_messages(request):
            await sse.handle_post_message(request.scope, request.receive, request._send)

        app = Starlette(
            routes=[
                Route("/sse", handle_sse),
                Route("/messages", handle_messages, methods=["POST"]),
            ],
        )
        uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
