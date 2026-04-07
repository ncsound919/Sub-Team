"""
main.py — Sub-Team orchestrator.

Supports three modes:

1. **Legacy pipeline** (default) — Runs the 4 deterministic CPU RTL agents.
2. **Agentic task execution** — Route a natural-language task to the CrewAI
   workforce (8 specialized agents).
3. **HTTP server** — Start the FastAPI server so Draymond can call Sub-Team
   like any other agent in the fleet.

Usage::

    # Legacy CPU pipeline (default)
    python main.py

    # Agentic task
    python main.py --task "Research the current state of RISC-V adoption"

    # Start the HTTP server (for Draymond integration)
    python main.py --serve
    python main.py --serve --port 8050 --host 0.0.0.0

    # Cross-disciplinary analysis (legacy)
    python main.py --analyze --domains logistics fintech --name "Supply chain optimization"

    # Business analysis (legacy)
    python main.py --business --domains finance sales --name "Revenue analysis"

    # List capabilities
    python main.py --capabilities
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
_log = logging.getLogger("sub-team")


# ============================================================================
# Legacy CPU Pipeline & Agentic Task Execution
# ============================================================================
# Canonical implementations live in sub_team.entry_points.
# Re-export here for backward compatibility with CLI callers.

from sub_team.entry_points import run_pipeline, run_task


# ============================================================================
# Cross-Disciplinary & Business Analysis (Legacy CLIs)
# ============================================================================


def run_analysis(name: str, domains: list[str], use_llm: bool = False) -> dict:
    """Run the cross-disciplinary analysis agent."""
    from sub_team import CrossDisciplinaryAgent, DomainProblem

    problem = DomainProblem(name=name, domains=domains, parameters={})
    agent = CrossDisciplinaryAgent()
    analysis = agent.run(problem, use_llm=use_llm)

    return {
        "name": analysis.problem_name,
        "domains_analyzed": analysis.domains_analyzed,
        "insights_count": len(analysis.insights),
        "overall_risk_score": analysis.overall_risk_score,
        "recommendations": analysis.recommendations,
    }


def run_business(name: str, domains: list[str], use_llm: bool = False) -> dict:
    """Run the business intelligence agent."""
    from sub_team import BusinessAgent, BusinessProblem

    problem = BusinessProblem(name=name, domains=domains, parameters={})
    agent = BusinessAgent()
    analysis = agent.run(problem, use_llm=use_llm)

    return {
        "name": analysis.problem_name,
        "domains_analyzed": analysis.domains_analyzed,
        "insights_count": len(analysis.insights),
        "overall_risk_score": analysis.overall_risk_score,
        "recommendations": analysis.recommendations,
    }


# ============================================================================
# HTTP Server
# ============================================================================


def serve(host: str = "0.0.0.0", port: int = 8050):
    """Start the FastAPI HTTP server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    print("=" * 60)
    print("Sub-Team: Starting HTTP Server (Agentic Workforce)")
    print(f"  Host:  {host}")
    print(f"  Port:  {port}")
    print(f"  Docs:  http://{host}:{port}/docs")
    print("=" * 60)

    uvicorn.run(
        "sub_team.server:app",
        host=host,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        reload=os.environ.get("SUB_TEAM_DEV", "").lower() in ("1", "true"),
    )


# ============================================================================
# Capabilities
# ============================================================================


def print_capabilities():
    """Print all Sub-Team capabilities to stdout."""
    from sub_team.crews import AgentRole, TaskType

    print("=" * 60)
    print("Sub-Team: Full Capabilities")
    print("=" * 60)
    print()

    print("Agents (CrewAI Workforce):")
    for role in AgentRole:
        print(f"  - {role.value}")
    print()

    print("Task Types (auto-routed):")
    for tt in TaskType:
        print(f"  - {tt.value}")
    print()

    print("Legacy Pipeline Agents:")
    print("  - SpecificationAgent")
    print("  - MicroarchitectureAgent")
    print("  - ImplementationAgent")
    print("  - VerificationAgent")
    print()

    print("Legacy Domain Agents:")
    print(
        "  - CrossDisciplinaryAgent (logistics, biotech, fintech, probability, legal)"
    )
    print("  - BusinessAgent (finance, sales + Stripe/HubSpot connectors)")
    print()

    print("Modes:")
    print("  - python main.py                  # Legacy CPU pipeline")
    print("  - python main.py --serve           # HTTP server for Draymond")
    print("  - python main.py --mcp             # MCP server (stdio transport)")
    print("  - python main.py --mcp --transport sse  # MCP server (SSE transport)")
    print("  - python main.py --task '...'      # Agentic task execution")
    print("  - python main.py --analyze ...     # Cross-disciplinary analysis")
    print("  - python main.py --business ...    # Business analysis")


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sub-team",
        description="Sub-Team: Full-spectrum agentic workforce + deterministic CPU pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default mode (no subcommand) = legacy pipeline
    # We handle this in main()

    # --serve
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI HTTP server for Draymond integration",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Server bind port (default: 8050)",
    )

    # --mcp
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Start the MCP (Model Context Protocol) server",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport mode (default: stdio)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8051,
        help="Port for MCP SSE transport (default: 8051)",
    )

    # --task
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Execute an agentic task (natural-language objective)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="Explicit task type for --task (auto-classified if omitted)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write task output to this file",
    )

    # --analyze
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run cross-disciplinary analysis",
    )
    parser.add_argument(
        "--business",
        action="store_true",
        help="Run business analysis",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Analysis",
        help="Problem name for analysis modes",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domains to analyze (e.g., logistics fintech)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM augmentation for analysis modes",
    )

    # --capabilities
    parser.add_argument(
        "--capabilities",
        action="store_true",
        help="Print all available capabilities and exit",
    )

    # --quiet
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Capabilities ────────────────────────────────────────────────
    if args.capabilities:
        print_capabilities()
        return

    # ── HTTP server ─────────────────────────────────────────────────
    if args.serve:
        serve(host=args.host, port=args.port)
        return

    # ── MCP server ──────────────────────────────────────────────────
    if args.mcp:
        # Delegate to the MCP server module with the right args
        sys.argv = ["mcp_server", "--transport", args.transport]
        if args.transport == "sse":
            sys.argv.extend(["--port", str(args.mcp_port)])
        from sub_team.mcp_server import main as mcp_main

        mcp_main()
        return

    # ── Agentic task ────────────────────────────────────────────────
    if args.task:
        result = run_task(
            objective=args.task,
            task_type=args.task_type,
            output_file=args.output,
            verbose=not args.quiet,
        )
        print()
        print("=" * 60)
        print("Result:")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0 if result.get("success") else 1)

    # ── Cross-disciplinary analysis ─────────────────────────────────
    if args.analyze:
        domains = args.domains or ["logistics", "fintech"]
        result = run_analysis(name=args.name, domains=domains, use_llm=args.use_llm)
        print(json.dumps(result, indent=2, default=str))
        return

    # ── Business analysis ───────────────────────────────────────────
    if args.business:
        domains = args.domains or ["finance", "sales"]
        result = run_business(name=args.name, domains=domains, use_llm=args.use_llm)
        print(json.dumps(result, indent=2, default=str))
        return

    # ── Default: Legacy CPU pipeline ────────────────────────────────
    success = run_pipeline()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
