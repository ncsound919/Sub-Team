"""
Code Executor Tool — Sandboxed code execution via E2B.

Gives agents the ability to write and run Python/JavaScript code in a
secure cloud sandbox. Falls back to a restricted local subprocess if
E2B is unavailable, with secrets stripped from the subprocess environment.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional

from crewai.tools import BaseTool
from pydantic import Field

_log = logging.getLogger(__name__)

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
    """Build a minimal environment for code execution with secrets stripped."""
    safe = {}
    for k, v in os.environ.items():
        k_upper = k.upper()
        if any(pat in k_upper for pat in _SECRET_ENV_PATTERNS):
            continue
        safe[k] = v
    # Always set these
    safe["PYTHONDONTWRITEBYTECODE"] = "1"
    if "PATH" not in safe:
        safe["PATH"] = os.environ.get("PATH", "")
    return safe


class CodeExecutorTool(BaseTool):
    """Execute Python or JavaScript code in a sandboxed environment."""

    name: str = "code_executor"
    description: str = (
        "Execute Python or JavaScript code and return the output. "
        "Input should be the code to execute as a string. Prefix with "
        "'```python' or '```javascript' to specify the language, otherwise "
        "Python is assumed. Use this for data processing, calculations, "
        "generating files, testing code snippets, or running analysis scripts."
    )
    timeout_seconds: int = Field(
        default=30, description="Max execution time in seconds"
    )

    def _run(self, code: str) -> str:
        """Execute code and return output."""
        if not code or not code.strip():
            return "Error: No code provided."

        # Parse language from markdown code fence
        language = "python"
        clean_code = code.strip()

        if clean_code.startswith("```"):
            first_line, _, rest = clean_code.partition("\n")
            lang_hint = first_line.replace("```", "").strip().lower()
            if lang_hint in ("python", "py"):
                language = "python"
            elif lang_hint in ("javascript", "js", "node"):
                language = "javascript"
            clean_code = rest
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()

        # Try E2B sandbox first
        e2b_key = os.environ.get("E2B_API_KEY")
        if e2b_key:
            try:
                return self._exec_e2b(clean_code, language, e2b_key)
            except Exception as e:
                _log.debug("E2B execution failed, falling back to local: %s", e)

        # Fallback: restricted local execution (Python only)
        if language == "python":
            return self._exec_local_python(clean_code)
        else:
            return f"Local execution of {language} not supported. Set E2B_API_KEY for full sandbox support."

    def _exec_e2b(self, code: str, language: str, api_key: str) -> str:
        """Execute code in E2B cloud sandbox."""
        from e2b_code_interpreter import Sandbox

        sandbox = Sandbox(api_key=api_key, timeout=self.timeout_seconds)
        try:
            execution = sandbox.run_code(code, language=language)

            parts = []
            if execution.logs.stdout:
                parts.append(f"STDOUT:\n{''.join(execution.logs.stdout)}")
            if execution.logs.stderr:
                parts.append(f"STDERR:\n{''.join(execution.logs.stderr)}")
            if execution.error:
                parts.append(f"ERROR:\n{execution.error.name}: {execution.error.value}")
                if execution.error.traceback:
                    parts.append(execution.error.traceback)
            if execution.results:
                for r in execution.results:
                    if hasattr(r, "text") and r.text:
                        parts.append(f"RESULT:\n{r.text}")

            return (
                "\n\n".join(parts)
                if parts
                else "Code executed successfully (no output)."
            )
        finally:
            sandbox.kill()

    def _exec_local_python(self, code: str) -> str:
        """Execute Python code in a restricted local subprocess with secrets stripped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=tempfile.gettempdir(),
                env=_safe_env(),
            )

            parts = []
            if result.stdout:
                parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                parts.append(f"STDERR:\n{result.stderr}")
            if result.returncode != 0:
                parts.append(f"Exit code: {result.returncode}")

            return (
                "\n\n".join(parts)
                if parts
                else "Code executed successfully (no output)."
            )

        except subprocess.TimeoutExpired:
            return f"Execution timed out after {self.timeout_seconds}s."
        except Exception as e:
            return f"Execution error: {type(e).__name__}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
