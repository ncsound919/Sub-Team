"""
Shell Execution Tool — Controlled shell command execution for agents.

Provides agents the ability to run shell commands with safety guards:
  - Executable allowlist (only permitted binaries)
  - No shell=True (prevents shell injection)
  - Secrets stripped from subprocess environment
  - Timeout enforcement
  - Output truncation to prevent token overflow
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
from typing import Optional

from crewai.tools import BaseTool
from pydantic import Field

_log = logging.getLogger(__name__)

# Only these executables are permitted
_ALLOWED_EXECUTABLES = frozenset(
    {
        # Version control
        "git",
        # Python tooling
        "python",
        "python3",
        "pip",
        "pip3",
        "pytest",
        "mypy",
        "ruff",
        "black",
        "isort",
        # Node.js tooling
        "node",
        "npm",
        "npx",
        "yarn",
        "pnpm",
        "tsc",
        # System utilities (read-only / safe)
        "ls",
        "dir",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "wc",
        "sort",
        "uniq",
        "diff",
        "file",
        "which",
        "where",
        "echo",
        "pwd",
        "date",
        "whoami",
        # Build tools
        "make",
        "cmake",
        "cargo",
        "go",
        "javac",
        "java",
        "dotnet",
        # Network (read-only)
        "curl",
        "wget",
        "ping",
        # Docker (common operations)
        "docker",
        "docker-compose",
    }
)

# Environment variable name patterns that indicate secrets
_SECRET_ENV_PATTERNS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
    "PRIVATE",
    "SIGNING",
    "ENCRYPTION",
)


def _safe_env() -> dict[str, str]:
    """Build a sanitized copy of the environment with secrets stripped."""
    safe = {}
    for k, v in os.environ.items():
        k_upper = k.upper()
        if any(pat in k_upper for pat in _SECRET_ENV_PATTERNS):
            continue
        safe[k] = v
    # Ensure basic PATH is always present
    if "PATH" not in safe:
        safe["PATH"] = os.environ.get("PATH", "")
    return safe


def _resolve_executable(name: str) -> Optional[str]:
    """Resolve an executable name to its full path, or None if not allowed."""
    base = os.path.basename(name).lower()
    # Strip .exe/.cmd/.bat on Windows
    for suffix in (".exe", ".cmd", ".bat"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if base not in _ALLOWED_EXECUTABLES:
        return None
    return shutil.which(name) or name


class ShellExecTool(BaseTool):
    """Execute shell commands with an executable allowlist and no shell injection."""

    name: str = "shell_exec"
    description: str = (
        "Execute a shell command and return its output. "
        "Input should be the command to run. Supports standard commands "
        "like ls, cat, grep, git, pip, npm, curl, etc. "
        "Uses an executable allowlist — only permitted binaries can run. "
        "Secrets are stripped from the subprocess environment."
    )
    timeout_seconds: int = Field(default=60, description="Max execution time")
    max_output_chars: int = Field(default=10000, description="Max output characters")

    def _run(self, command: str) -> str:
        if not command or not command.strip():
            return "Error: No command provided."

        command = command.strip()

        # Parse the command into tokens (no shell=True)
        try:
            tokens = shlex.split(command)
        except ValueError as e:
            return f"Error: Could not parse command: {e}"

        if not tokens:
            return "Error: Empty command after parsing."

        executable = tokens[0]

        # Resolve and validate executable against allowlist
        resolved = _resolve_executable(executable)
        if resolved is None:
            return (
                f"Error: Executable '{executable}' is not in the allowlist. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXECUTABLES))}"
            )

        tokens[0] = resolved

        try:
            result = subprocess.run(
                tokens,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=os.environ.get("SUB_TEAM_WORKSPACE", os.getcwd()),
                env=_safe_env(),
            )

            parts = []
            if result.stdout:
                stdout = result.stdout[: self.max_output_chars]
                parts.append(f"STDOUT:\n{stdout}")
                if len(result.stdout) > self.max_output_chars:
                    parts.append(f"... (truncated, {len(result.stdout):,} total chars)")
            if result.stderr:
                stderr = result.stderr[: self.max_output_chars]
                parts.append(f"STDERR:\n{stderr}")
            if result.returncode != 0:
                parts.append(f"Exit code: {result.returncode}")

            return (
                "\n\n".join(parts)
                if parts
                else "Command executed successfully (no output)."
            )

        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.timeout_seconds}s."
        except FileNotFoundError:
            return f"Error: Executable '{executable}' not found on this system."
        except Exception as e:
            return f"Command execution error: {type(e).__name__}"
