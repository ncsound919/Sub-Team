"""
Sub-Team: Deterministic autocoding sub-agent system for low-code CPU design.

Four collaborative agents:
  - SpecificationAgent       : Extracts formal constraints from ISA specs
  - MicroarchitectureAgent   : Synthesizes structural RTL skeletons
  - ImplementationAgent      : Generates synthesizable Verilog/VHDL via formal grammars
  - VerificationAgent        : Produces formal correctness proofs or counterexamples

Cross-disciplinary analysis agent:
  - CrossDisciplinaryAgent   : Rule-based analysis across logistics, biotech,
                               fintech and probability domains
"""

from .specification_agent import SpecificationAgent
from .microarchitecture_agent import MicroarchitectureAgent
from .implementation_agent import ImplementationAgent
from .verification_agent import VerificationAgent
from .cpu import CPU, ISA, PipelineTemplate, BranchPredictor
from .cross_disciplinary_agent import (
    CrossDisciplinaryAgent,
    DomainProblem,
    CrossDisciplinaryAnalysis,
    DomainInsight,
    CrossDomainLink,
    SUPPORTED_DOMAINS,
)

__all__ = [
    "SpecificationAgent",
    "MicroarchitectureAgent",
    "ImplementationAgent",
    "VerificationAgent",
    "CPU",
    "ISA",
    "PipelineTemplate",
    "BranchPredictor",
    "CrossDisciplinaryAgent",
    "DomainProblem",
    "CrossDisciplinaryAnalysis",
    "DomainInsight",
    "CrossDomainLink",
    "SUPPORTED_DOMAINS",
]
