"""
CPU low-code specification interface.

Provides a declarative, parameter-driven way to describe a CPU design.
The CPU object is consumed by the sub-agent pipeline to produce formally
verified RTL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ISA(Enum):
    """Supported Instruction Set Architectures."""
    RV32I  = auto()   # RISC-V base integer (32-bit)
    RV32IM = auto()   # RV32I + multiply/divide
    RV32IMA = auto()  # RV32IM + atomic
    RV64I  = auto()   # RISC-V base integer (64-bit)
    RV64IM = auto()   # RV64I + multiply/divide
    MIPS32 = auto()   # MIPS 32-bit
    ARM_M0 = auto()   # ARM Cortex-M0 (Thumb-1)


class PipelineTemplate(Enum):
    """Predefined pipeline depth templates."""
    SINGLE_CYCLE = auto()   # No pipeline; simplest correct design
    MULTI_CYCLE  = auto()   # Shared datapath, multi-cycle execution
    FIVE_STAGE   = auto()   # Classic IF/ID/EX/MEM/WB pipeline
    OUT_OF_ORDER = auto()   # Tomasulo-style out-of-order execution


@dataclass
class BranchPredictor:
    """Parameterised branch predictor specification."""
    scheme: str          # e.g. "gshare", "bimodal", "always_not_taken"
    bits: int = 8        # History / PHT size in bits

    def __str__(self) -> str:
        return f"{self.scheme}(bits={self.bits})"


def gshare(bits: int = 8) -> BranchPredictor:
    """Convenience constructor for a GShare predictor."""
    return BranchPredictor(scheme="gshare", bits=bits)


def bimodal(bits: int = 8) -> BranchPredictor:
    """Convenience constructor for a Bimodal predictor."""
    return BranchPredictor(scheme="bimodal", bits=bits)


@dataclass
class CPU:
    """
    Low-code CPU specification.

    Example::

        cpu = CPU(
            isa=ISA.RV32IM,
            pipeline=PipelineTemplate.FIVE_STAGE,
            forwarding=True,
            branch_predictor=gshare(bits=8),
        )
    """

    isa: ISA
    pipeline: PipelineTemplate = PipelineTemplate.FIVE_STAGE
    forwarding: bool = True
    branch_predictor: Optional[BranchPredictor] = None
    # Power / area / timing constraints (normalised, 0.0 – 1.0)
    max_power: float = 1.0
    max_area: float = 1.0
    target_freq_mhz: Optional[float] = None
    # Extra user-defined constraints passed through to the agents
    extra_constraints: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "CPU Specification",
            f"  ISA              : {self.isa.name}",
            f"  Pipeline         : {self.pipeline.name}",
            f"  Data forwarding  : {self.forwarding}",
            f"  Branch predictor : {self.branch_predictor or 'none'}",
            f"  Max power        : {self.max_power}",
            f"  Max area         : {self.max_area}",
            f"  Target freq (MHz): {self.target_freq_mhz or 'unconstrained'}",
        ]
        if self.extra_constraints:
            lines.append(f"  Extra constraints: {self.extra_constraints}")
        return "\n".join(lines)
