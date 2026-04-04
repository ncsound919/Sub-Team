"""
Verification Agent — Structural Consistency Checks.

Responsibility
--------------
Input  : FormalSpec + RTLOutput.
Output : A VerificationReport summarising the outcome (PASS / FAIL /
         UNRESOLVED) for each instruction's checks.

Method
------
The agent performs lightweight, structural checks over the generated
RTL/Verilog text. Typical checks include:

  * Interface and register-map consistency between the FormalSpec and
    the emitted RTL (e.g. expected port / signal names are present).
  * Basic encoding- and naming-level sanity checks for instructions.
  * Simple invariants that can be expressed as pattern or substring
    matches over the Verilog source.

This module does not generate SMT-LIB 2 assertions, does not run a
symbolic interpreter, and does not provide a formal “correct-by-
construction” proof of correctness. Its guarantees are limited to
the structural/textual properties that are explicitly checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

from .specification_agent import FormalSpec
from .implementation_agent import RTLOutput


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

class CheckStatus(Enum):
    PASS       = auto()
    FAIL       = auto()
    UNRESOLVED = auto()


@dataclass
class CheckResult:
    check_name: str
    status: CheckStatus
    detail: str = ""


@dataclass
class VerificationReport:
    """Complete report produced by VerificationAgent."""
    results: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.FAIL)

    @property
    def unresolved(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.UNRESOLVED)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.unresolved == 0

    def summary(self) -> str:
        lines = [
            "VerificationReport",
            f"  PASS       : {self.passed}",
            f"  FAIL       : {self.failed}",
            f"  UNRESOLVED : {self.unresolved}",
        ]
        for r in self.results:
            if r.status != CheckStatus.PASS:
                lines.append(f"  [{r.status.name}] {r.check_name}: {r.detail}")
        verdict = "✓ ALL CHECKS PASSED" if self.all_passed else "✗ VERIFICATION FAILED"
        lines.append(f"  {verdict}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structural / invariant checkers
# ---------------------------------------------------------------------------

def _check_module_present(rtl: RTLOutput, module_name: str) -> CheckResult:
    found = any(m.name == module_name for m in rtl.modules)
    if found:
        return CheckResult(
            check_name=f"module_present:{module_name}",
            status=CheckStatus.PASS,
        )
    return CheckResult(
        check_name=f"module_present:{module_name}",
        status=CheckStatus.FAIL,
        detail=f"Module '{module_name}' not found in RTLOutput",
    )


def _check_x0_hardwired_zero(rtl: RTLOutput) -> CheckResult:
    """
    Verify that the register-file module contains the x0 hardwire logic.
    """
    rf = next((m for m in rtl.modules if m.name == "regfile"), None)
    if rf is None:
        return CheckResult(
            check_name="x0_hardwired_zero",
            status=CheckStatus.UNRESOLVED,
            detail="regfile module not found",
        )
    # Deterministic structural check: look for the zero-assignment pattern
    if "rs1_addr == 0" in rf.source and "rs2_addr == 0" in rf.source:
        return CheckResult(
            check_name="x0_hardwired_zero",
            status=CheckStatus.PASS,
        )
    return CheckResult(
        check_name="x0_hardwired_zero",
        status=CheckStatus.FAIL,
        detail="Register file does not hard-wire x0 to zero",
    )


def _check_alu_completeness(
    rtl: RTLOutput, mnemonics: List[str]
) -> List[CheckResult]:
    """
    Check that the ALU source covers the core arithmetic operations.
    """
    alu = next((m for m in rtl.modules if m.name == "alu"), None)
    if alu is None:
        return [CheckResult(
            check_name="alu_completeness",
            status=CheckStatus.UNRESOLVED,
            detail="alu module not found",
        )]

    results: List[CheckResult] = []
    required_ops = ["ALU_ADD", "ALU_SUB", "ALU_AND", "ALU_OR", "ALU_XOR"]
    for op in required_ops:
        if op in alu.source:
            results.append(CheckResult(
                check_name=f"alu_op:{op}",
                status=CheckStatus.PASS,
            ))
        else:
            results.append(CheckResult(
                check_name=f"alu_op:{op}",
                status=CheckStatus.FAIL,
                detail=f"ALU operation {op} not found in alu module source",
            ))

    mul_needed = any(m in mnemonics for m in ("MUL", "DIV", "REM"))
    if mul_needed:
        for op in ("ALU_MUL", "ALU_DIV", "ALU_REM"):
            if op in alu.source:
                results.append(CheckResult(
                    check_name=f"alu_op:{op}",
                    status=CheckStatus.PASS,
                ))
            else:
                results.append(CheckResult(
                    check_name=f"alu_op:{op}",
                    status=CheckStatus.FAIL,
                    detail=f"M-extension op {op} missing from ALU",
                ))
    return results


def _check_pc_increment(rtl: RTLOutput, isa_name: str) -> CheckResult:
    """
    Verify that the top-level CPU module updates the PC.
    """
    top = next(
        (m for m in rtl.modules if m.name == f"cpu_{isa_name.lower()}"), None
    )
    if top is None:
        return CheckResult(
            check_name="pc_increment",
            status=CheckStatus.UNRESOLVED,
            detail=f"cpu_{isa_name.lower()} module not found",
        )
    if "pc_next" in top.source and "pc + " in top.source:
        return CheckResult(check_name="pc_increment", status=CheckStatus.PASS)
    return CheckResult(
        check_name="pc_increment",
        status=CheckStatus.FAIL,
        detail="PC increment logic not detected in top-level module",
    )


def _check_formula_postconditions(spec: FormalSpec) -> List[CheckResult]:
    """
    Structural check: every instruction must have at least one postcondition.
    """
    results: List[CheckResult] = []
    for formula in spec.formulas:
        if formula.postconditions:
            results.append(CheckResult(
                check_name=f"formula:{formula.instruction}",
                status=CheckStatus.PASS,
            ))
        else:
            results.append(CheckResult(
                check_name=f"formula:{formula.instruction}",
                status=CheckStatus.FAIL,
                detail="No postconditions defined for this instruction",
            ))
    return results


def _check_hazard_unit_present(rtl: RTLOutput) -> CheckResult:
    return _check_module_present(rtl, "hazard_unit")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VerificationAgent:
    """
    Runs formal structural checks on the generated RTL against the FormalSpec.

    Usage::

        agent  = VerificationAgent()
        report = agent.run(formal_spec, rtl_output)
        print(report.summary())
    """

    def run(self, spec: FormalSpec, rtl: RTLOutput) -> VerificationReport:
        """
        Execute all verification checks and return a VerificationReport.
        """
        report = VerificationReport()
        mnemonics = [enc.mnemonic for enc in spec.encodings]

        # 1. Required module presence
        for mod_name in ("alu", "regfile", "hazard_unit"):
            report.results.append(_check_module_present(rtl, mod_name))

        # 2. x0 hardwired-to-zero invariant
        report.results.append(_check_x0_hardwired_zero(rtl))

        # 3. ALU operation completeness
        report.results.extend(_check_alu_completeness(rtl, mnemonics))

        # 4. PC increment logic
        report.results.append(_check_pc_increment(rtl, spec.isa_name))

        # 5. Formula completeness (every instruction has postconditions)
        report.results.extend(_check_formula_postconditions(spec))

        return report
