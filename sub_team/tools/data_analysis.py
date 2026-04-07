"""
Data Analysis Tool — Pandas-powered data processing for agents.

Allows agents to load, transform, and analyze structured data (CSV, JSON,
Excel) using pandas. Returns summary statistics, filtered views, and
computed metrics.

Uses df.query() for safe user expressions instead of eval().
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

from crewai.tools import BaseTool
from pydantic import Field

_log = logging.getLogger(__name__)

# Restrict data file access to workspace directory
_WORKSPACE = os.environ.get("SUB_TEAM_WORKSPACE", os.getcwd())


def _is_safe_data_path(path: str) -> bool:
    """Validate that a data file path is within the workspace."""
    resolved = os.path.realpath(os.path.abspath(path))
    workspace = os.path.realpath(os.path.abspath(_WORKSPACE))
    return resolved.startswith(workspace)


class DataAnalysisTool(BaseTool):
    """Analyze structured data using pandas. Supports CSV, JSON, and Excel."""

    name: str = "data_analysis"
    description: str = (
        "Analyze structured data from a file or inline CSV/JSON. "
        "Input should be either a file path to a CSV/JSON/Excel file, "
        "or inline CSV data. Returns summary statistics, shape, columns, "
        "data types, null counts, and a preview. "
        "For filtering, append a pandas query expression after '|||' separator. "
        "Example: 'data.csv|||category == \"electronics\" and price > 100' "
        "(uses df.query() syntax — column conditions, not arbitrary Python)"
    )

    def _run(self, input_str: str) -> str:
        try:
            import pandas as pd
        except ImportError:
            return "Error: pandas not installed. Run: pip install pandas"

        if not input_str or not input_str.strip():
            return "Error: No data source provided."

        # Split optional query
        query = None
        if "|||" in input_str:
            source, _, query = input_str.partition("|||")
            source = source.strip()
            query = query.strip()
        else:
            source = input_str.strip()

        # Load data
        df = self._load_data(pd, source)
        if isinstance(df, str):
            return df  # Error message

        # Build summary
        parts = [
            f"## Data Summary",
            f"**Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns",
            f"**Columns:** {', '.join(df.columns.tolist())}",
            "",
            "### Data Types",
            df.dtypes.to_string(),
            "",
            "### Null Counts",
            df.isnull().sum().to_string(),
            "",
            "### Statistics",
            df.describe(include="all").to_string(),
            "",
            "### First 5 Rows",
            df.head().to_string(),
        ]

        # Execute optional query using safe df.query()
        if query:
            try:
                result = df.query(query)
                parts.append("")
                parts.append(f"### Query Result: `{query}`")
                parts.append(f"**Matching rows:** {len(result):,}")
                if isinstance(result, pd.DataFrame):
                    # Limit output to first 50 rows
                    parts.append(result.head(50).to_string())
                    if len(result) > 50:
                        parts.append(f"... ({len(result) - 50} more rows)")
                else:
                    parts.append(str(result))
            except Exception as e:
                parts.append(f"\n### Query Error: {type(e).__name__}: {e}")

        return "\n".join(parts)

    def _load_data(self, pd, source: str):
        """Try to load data from file path or inline CSV/JSON."""
        # Try as file path first
        if os.path.isfile(source):
            # Validate path is within workspace
            if not _is_safe_data_path(source):
                return f"Error: Access denied — file path must be within the workspace directory."

            ext = os.path.splitext(source)[1].lower()
            try:
                if ext == ".csv":
                    return pd.read_csv(source)
                elif ext in (".json", ".jsonl"):
                    return pd.read_json(source, lines=ext == ".jsonl")
                elif ext in (".xlsx", ".xls"):
                    return pd.read_excel(source)
                elif ext == ".parquet":
                    return pd.read_parquet(source)
                else:
                    # Try CSV as default
                    return pd.read_csv(source)
            except Exception as e:
                return f"Error reading file {source}: {type(e).__name__}: {e}"

        # Try as inline CSV
        try:
            return pd.read_csv(io.StringIO(source))
        except Exception:
            pass

        # Try as inline JSON
        try:
            return pd.read_json(io.StringIO(source))
        except Exception:
            pass

        return f"Error: Could not load data from '{source[:100]}...'. Provide a file path or inline CSV/JSON."
