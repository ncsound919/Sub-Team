"""
File Operations Tools — Local filesystem access for agents.

Provides read, write, and directory listing capabilities with safety
guards against path traversal and dangerous operations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from crewai.tools import BaseTool
from pydantic import Field

_log = logging.getLogger(__name__)

# Safety: restrict file operations to workspace directory only
_ALLOWED_ROOTS = [
    os.environ.get("SUB_TEAM_WORKSPACE", os.getcwd()),
]


def _get_allowed_roots() -> list[str]:
    """Get allowed roots evaluated at call time (not stale import-time values)."""
    return [
        os.environ.get("SUB_TEAM_WORKSPACE", os.getcwd()),
    ]


def _is_safe_path(path: str) -> bool:
    """Check if a path is within allowed roots (prevent traversal attacks)."""
    resolved = os.path.realpath(os.path.abspath(path))
    return any(
        resolved.startswith(os.path.realpath(os.path.abspath(root)))
        for root in _get_allowed_roots()
        if root
    )


class FileReadTool(BaseTool):
    """Read the contents of a file from the local filesystem."""

    name: str = "file_read"
    description: str = (
        "Read the contents of a file. Input should be the file path. "
        "Returns the file contents as text. Use this to read source code, "
        "configuration files, documentation, data files, or any text file."
    )
    max_size_bytes: int = Field(default=500_000, description="Max file size to read")

    def _run(self, file_path: str) -> str:
        if not file_path or not file_path.strip():
            return "Error: No file path provided."

        path = Path(file_path.strip()).resolve()

        if not _is_safe_path(str(path)):
            return f"Error: Access denied — path outside allowed workspace."

        if not path.exists():
            return f"Error: File not found: {path}"

        if not path.is_file():
            return f"Error: Not a file: {path}"

        size = path.stat().st_size
        if size > self.max_size_bytes:
            return (
                f"Error: File too large ({size:,} bytes, max {self.max_size_bytes:,})."
            )

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            return content
        except Exception as e:
            return f"Error reading {path}: {e}"


class FileWriteTool(BaseTool):
    """Write content to a file on the local filesystem."""

    name: str = "file_write"
    description: str = (
        "Write content to a file. Input should be formatted as: "
        "'FILE_PATH|||CONTENT' where ||| separates the path from the content. "
        "Creates parent directories if needed. Use this to save code, reports, "
        "data exports, or any generated content to disk."
    )

    def _run(self, input_str: str) -> str:
        if not input_str or "|||" not in input_str:
            return "Error: Input must be 'FILE_PATH|||CONTENT'."

        file_path, _, content = input_str.partition("|||")
        file_path = file_path.strip()

        if not file_path:
            return "Error: No file path provided."

        path = Path(file_path).resolve()

        if not _is_safe_path(str(path)):
            return f"Error: Access denied — path outside allowed workspace."

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content):,} chars to {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"


class DirectoryListTool(BaseTool):
    """List files and directories in a given path."""

    name: str = "directory_list"
    description: str = (
        "List files and subdirectories in a directory. Input should be the "
        "directory path. Returns a structured listing with file sizes and types. "
        "Use this to explore project structures, find files, or understand "
        "a codebase layout."
    )
    max_entries: int = Field(default=200, description="Max entries to return")

    def _run(self, dir_path: str) -> str:
        if not dir_path or not dir_path.strip():
            dir_path = "."

        path = Path(dir_path.strip()).resolve()

        if not _is_safe_path(str(path)):
            return f"Error: Access denied — path outside allowed workspace."

        if not path.exists():
            return f"Error: Directory not found: {path}"

        if not path.is_dir():
            return f"Error: Not a directory: {path}"

        try:
            entries = sorted(
                path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
            lines = [f"## Directory: {path}\n"]

            for i, entry in enumerate(entries):
                if i >= self.max_entries:
                    lines.append(
                        f"\n... and {len(list(path.iterdir())) - self.max_entries} more entries"
                    )
                    break

                if entry.is_dir():
                    lines.append(f"  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    lines.append(f"  {entry.name}  ({size_str})")

            return "\n".join(lines)
        except PermissionError:
            return f"Error: Permission denied for {path}"
        except Exception as e:
            return f"Error listing {path}: {e}"
