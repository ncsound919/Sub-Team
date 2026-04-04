"""
main.py — Sub-Team orchestrator.

Runs the four sub-agents in sequence to produce verified RTL from a
low-code CPU specification.

Usage::

    python main.py

To target a different ISA or pipeline, edit the CPU(...) call at the
bottom of this file or import the sub_team package directly.
"""

from sub_team import (
    CPU,
    ISA,
    PipelineTemplate,
    BranchPredictor,
    SpecificationAgent,
    MicroarchitectureAgent,
    ImplementationAgent,
    VerificationAgent,
)
from sub_team.cpu import gshare


def run_pipeline(cpu: CPU, rtl_output_dir: str = "rtl_out") -> bool:
    """
    Execute the full sub-agent pipeline for *cpu*.

    Returns True if verification passes, False otherwise.
    """
    print("=" * 60)
    print("Sub-Team: Deterministic CPU Autocoding Pipeline")
    print("=" * 60)
    print()
    print(cpu.summary())
    print()

    # ------------------------------------------------------------------
    # Agent 1 — Specification
    # ------------------------------------------------------------------
    print("─" * 60)
    print("Agent 1: SpecificationAgent  (Constraint Extraction)")
    print("─" * 60)
    spec_agent = SpecificationAgent()
    spec = spec_agent.run(cpu)
    print(spec.summary())
    print()

    # ------------------------------------------------------------------
    # Agent 2 — Microarchitecture
    # ------------------------------------------------------------------
    print("─" * 60)
    print("Agent 2: MicroarchitectureAgent  (Structure Synthesis)")
    print("─" * 60)
    uarch_agent = MicroarchitectureAgent()
    plan = uarch_agent.run(spec)
    print(plan.summary())
    print()

    # ------------------------------------------------------------------
    # Agent 3 — Implementation
    # ------------------------------------------------------------------
    print("─" * 60)
    print("Agent 3: ImplementationAgent  (Code Generation)")
    print("─" * 60)
    impl_agent = ImplementationAgent()
    rtl = impl_agent.run(spec, plan)
    print(rtl.summary())
    paths = rtl.write_to_dir(rtl_output_dir)
    print(f"  Written to: {paths}")
    print()

    # ------------------------------------------------------------------
    # Agent 4 — Verification
    # ------------------------------------------------------------------
    print("─" * 60)
    print("Agent 4: VerificationAgent  (Correctness Proof)")
    print("─" * 60)
    verif_agent = VerificationAgent()
    report = verif_agent.run(spec, rtl)
    print(report.summary())
    print()

    return report.all_passed


if __name__ == "__main__":
    cpu = CPU(
        isa=ISA.RV32IM,
        pipeline=PipelineTemplate.FIVE_STAGE,
        forwarding=True,
        branch_predictor=gshare(bits=8),
    )
    success = run_pipeline(cpu)
    raise SystemExit(0 if success else 1)
