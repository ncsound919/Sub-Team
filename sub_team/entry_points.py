"""
Entry points for Sub-Team operations.

Provides ``run_pipeline`` (legacy CPU autocoding) and ``run_task`` (agentic
workforce) as importable functions from within the ``sub_team`` package.

Previously these lived only in the top-level ``main.py``, which made
``from main import run_pipeline`` imports fragile — they broke when the
working directory or ``sys.path`` changed.  Now every consumer imports from
``sub_team.entry_points`` instead.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


# ============================================================================
# Legacy CPU Pipeline
# ============================================================================


def run_pipeline(cpu=None, rtl_output_dir: str = "rtl_out") -> bool:
    """
    Execute the full sub-agent pipeline for *cpu*.

    Returns True if verification passes, False otherwise.
    """
    from sub_team import (
        CPU,
        ISA,
        PipelineTemplate,
        SpecificationAgent,
        MicroarchitectureAgent,
        ImplementationAgent,
        VerificationAgent,
    )
    from sub_team.cpu import gshare

    if cpu is None:
        cpu = CPU(
            isa=ISA.RV32IM,
            pipeline=PipelineTemplate.FIVE_STAGE,
            forwarding=True,
            branch_predictor=gshare(bits=8),
        )

    print("=" * 60)
    print("Sub-Team: Deterministic CPU Autocoding Pipeline")
    print("=" * 60)
    print()
    print(cpu.summary())
    print()

    # Agent 1 — Specification
    print("─" * 60)
    print("Agent 1: SpecificationAgent  (Constraint Extraction)")
    print("─" * 60)
    spec_agent = SpecificationAgent()
    spec = spec_agent.run(cpu)
    print(spec.summary())
    print()

    # Agent 2 — Microarchitecture
    print("─" * 60)
    print("Agent 2: MicroarchitectureAgent  (Structure Synthesis)")
    print("─" * 60)
    uarch_agent = MicroarchitectureAgent()
    plan = uarch_agent.run(spec)
    print(plan.summary())
    print()

    # Agent 3 — Implementation
    print("─" * 60)
    print("Agent 3: ImplementationAgent  (Code Generation)")
    print("─" * 60)
    impl_agent = ImplementationAgent()
    rtl = impl_agent.run(spec, plan)
    print(rtl.summary())
    paths = rtl.write_to_dir(rtl_output_dir)
    print(f"  Written to: {paths}")
    print()

    # Agent 4 — Verification
    print("─" * 60)
    print("Agent 4: VerificationAgent  (Correctness Proof)")
    print("─" * 60)
    verif_agent = VerificationAgent()
    report = verif_agent.run(spec, rtl)
    print(report.summary())
    print()

    return report.all_passed


# ============================================================================
# Agentic Task Execution
# ============================================================================


def run_task(
    objective: str,
    task_type: str | None = None,
    output_file: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Execute a task using the CrewAI agentic workforce.

    Parameters
    ----------
    objective : str
        Natural-language description of what to accomplish.
    task_type : str, optional
        Explicit task type (e.g., 'research', 'code_generation').
        Auto-classified if omitted.
    output_file : str, optional
        Write output to this file.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict
        Result dictionary with output, agents_used, duration_ms, etc.
    """
    from sub_team.crews import SubTeamWorkforce, TaskType

    workforce = SubTeamWorkforce(verbose=verbose)

    if task_type:
        try:
            tt = TaskType(task_type)
        except ValueError:
            print(f"Unknown task type: {task_type}")
            print(f"Available: {[t.value for t in TaskType]}")
            return {"success": False, "error": f"Unknown task type: {task_type}"}

        result = workforce.execute(
            task_type=tt,
            objective=objective,
            output_file=output_file,
        )
    else:
        result = workforce.classify_and_execute(
            objective=objective,
            output_file=output_file,
        )

    return result.to_dict()
