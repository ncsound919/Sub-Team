"""
Microarchitecture Agent — Structure Synthesis.

Responsibility
--------------
Input  : FormalSpec produced by SpecificationAgent + CPU constraints.
Output : A MicroarchPlan describing:
           - Pipeline stage breakdown (names, latencies)
           - Hazard detection / forwarding paths
           - Memory hierarchy description
           - Branch prediction configuration

Method
------
Constraint-based template selection (CEGIS-inspired).  Templates for common
CPU types are stored as parameterised skeletons; the agent selects and
instantiates the matching template deterministically from the constraints.
Same constraints → same architecture every time.

LLM augmentation (optional)
----------------------------
When ``use_llm=True`` is passed to ``MicroarchitectureAgent.run()`` **and** an
API key is available, the LLM is asked to:

  * Justify the chosen pipeline template against the constraints.
  * Identify potential micro-architectural risks or trade-offs.
  * Suggest one or two alternative configurations the designer might consider.

Results are stored in ``MicroarchPlan.llm_rationale`` (list of strings) and
never alter the deterministic stage/hazard/memory configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable

from .cpu import PipelineTemplate
from .specification_agent import FormalSpec


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    name: str
    latency_cycles: int
    description: str


@dataclass
class HazardUnit:
    forwarding_enabled: bool
    stall_on_load: bool       # load-use hazard requires a bubble
    branch_resolution_stage: str  # stage where branch is resolved


@dataclass
class MemoryConfig:
    data_width_bits: int
    addr_width_bits: int
    icache: Optional[str] = None   # e.g. "direct-mapped 4KB"
    dcache: Optional[str] = None


@dataclass
class MicroarchPlan:
    """Structural plan emitted by the MicroarchitectureAgent."""
    pipeline_name: str
    stages: List[PipelineStage] = field(default_factory=list)
    hazard_unit: Optional[HazardUnit] = None
    memory: Optional[MemoryConfig] = None
    branch_predictor: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    # LLM-generated rationale / alternative suggestions (empty when not used)
    llm_rationale: List[str] = field(default_factory=list)

    def summary(self) -> str:
        stage_names = " → ".join(s.name for s in self.stages)
        lines = [
            f"MicroarchPlan [{self.pipeline_name}]",
            f"  Stages          : {stage_names}",
        ]
        if self.hazard_unit:
            lines.append(
                f"  Forwarding      : {self.hazard_unit.forwarding_enabled}"
            )
            lines.append(
                f"  Load-use stall  : {self.hazard_unit.stall_on_load}"
            )
            lines.append(
                f"  Branch resolves : {self.hazard_unit.branch_resolution_stage}"
            )
        if self.branch_predictor:
            lines.append(f"  Branch predictor: {self.branch_predictor}")
        if self.memory:
            lines.append(
                f"  Data width      : {self.memory.data_width_bits}-bit"
            )
        for note in self.notes:
            lines.append(f"  Note: {note}")
        if self.llm_rationale:
            lines.append(f"  LLM rationale   : {len(self.llm_rationale)} items")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline templates
# ---------------------------------------------------------------------------

def _single_cycle_plan(spec: FormalSpec) -> MicroarchPlan:
    data_width = 64 if "64" in spec.isa_name else 32
    plan = MicroarchPlan(pipeline_name="SINGLE_CYCLE")
    plan.stages = [
        PipelineStage("FETCH",   1, "Fetch and decode instruction in one cycle"),
        PipelineStage("EXECUTE", 1, "All ALU, memory and branch operations"),
        PipelineStage("WRITEBACK", 1, "Write result to register file"),
    ]
    plan.hazard_unit = HazardUnit(
        forwarding_enabled=False,
        stall_on_load=False,
        branch_resolution_stage="EXECUTE",
    )
    plan.memory = MemoryConfig(
        data_width_bits=data_width,
        addr_width_bits=data_width,
    )
    plan.notes.append("No pipeline hazards; new instruction starts each cycle.")
    return plan


def _multi_cycle_plan(spec: FormalSpec) -> MicroarchPlan:
    data_width = 64 if "64" in spec.isa_name else 32
    plan = MicroarchPlan(pipeline_name="MULTI_CYCLE")
    plan.stages = [
        PipelineStage("IF",  1, "Instruction Fetch"),
        PipelineStage("ID",  1, "Instruction Decode & Register Read"),
        PipelineStage("EX",  1, "Execute / Address Calculation"),
        PipelineStage("MEM", 1, "Memory Access"),
        PipelineStage("WB",  1, "Write Back"),
    ]
    plan.hazard_unit = HazardUnit(
        forwarding_enabled=False,
        stall_on_load=True,
        branch_resolution_stage="EX",
    )
    plan.memory = MemoryConfig(
        data_width_bits=data_width,
        addr_width_bits=data_width,
        icache="direct-mapped 4KB",
        dcache="direct-mapped 4KB",
    )
    plan.notes.append("Shared datapath; operations take multiple clock cycles.")
    return plan


def _five_stage_plan(spec: FormalSpec, forwarding: bool) -> MicroarchPlan:
    data_width = 64 if "64" in spec.isa_name else 32
    plan = MicroarchPlan(pipeline_name="FIVE_STAGE")
    plan.stages = [
        PipelineStage("IF",  1, "Instruction Fetch"),
        PipelineStage("ID",  1, "Instruction Decode & Register Read"),
        PipelineStage("EX",  1, "Execute / Address Calculation"),
        PipelineStage("MEM", 1, "Memory Access"),
        PipelineStage("WB",  1, "Write Back"),
    ]
    plan.hazard_unit = HazardUnit(
        forwarding_enabled=forwarding,
        stall_on_load=True,   # load-use hazard still needs a bubble
        branch_resolution_stage="MEM",
    )
    plan.memory = MemoryConfig(
        data_width_bits=data_width,
        addr_width_bits=data_width,
        icache="direct-mapped 16KB",
        dcache="direct-mapped 16KB",
    )
    bp = spec.constraints.get("branch_predictor")
    plan.branch_predictor = str(bp) if bp else "static-not-taken"
    if forwarding:
        plan.notes.append("EX-EX and MEM-EX forwarding paths active.")
    else:
        plan.notes.append("No forwarding; stall on all data hazards.")
    return plan


def _out_of_order_plan(spec: FormalSpec) -> MicroarchPlan:
    data_width = 64 if "64" in spec.isa_name else 32
    plan = MicroarchPlan(pipeline_name="OUT_OF_ORDER")
    plan.stages = [
        PipelineStage("FETCH",    1, "Superscalar fetch (2-wide)"),
        PipelineStage("DECODE",   1, "Decode & rename registers (ROB)"),
        PipelineStage("DISPATCH", 1, "Issue to reservation stations"),
        PipelineStage("EXECUTE",  1, "Multi-functional-unit execution"),
        PipelineStage("COMPLETE", 1, "Writeback to ROB"),
        PipelineStage("COMMIT",   1, "In-order retirement"),
    ]
    plan.hazard_unit = HazardUnit(
        forwarding_enabled=True,
        stall_on_load=False,   # OOO can fill load latency with other instructions
        branch_resolution_stage="EXECUTE",
    )
    plan.memory = MemoryConfig(
        data_width_bits=data_width,
        addr_width_bits=data_width,
        icache="set-associative 32KB",
        dcache="set-associative 32KB",
    )
    bp = spec.constraints.get("branch_predictor")
    plan.branch_predictor = str(bp) if bp else "tournament"
    plan.notes.append("Tomasulo-style OOO execution with reorder buffer.")
    return plan


_TEMPLATE_MAP: Dict[str, Callable] = {
    "SINGLE_CYCLE": _single_cycle_plan,
    "MULTI_CYCLE":  _multi_cycle_plan,
    "FIVE_STAGE":   _five_stage_plan,
    "OUT_OF_ORDER": _out_of_order_plan,
}


# ---------------------------------------------------------------------------
# LLM augmentation helper
# ---------------------------------------------------------------------------

def _llm_augment_plan(spec: FormalSpec, plan: "MicroarchPlan") -> List[str]:
    """
    Ask the LLM to justify the chosen pipeline and suggest alternatives.
    Returns a list of rationale strings, or empty list if unavailable.
    """
    from .llm_client import llm_complete

    system = (
        "You are a senior CPU microarchitect. Given an ISA formal spec and the "
        "pipeline template that was deterministically selected, provide:\n"
        "1. A brief justification for why this pipeline template was appropriate.\n"
        "2. Any micro-architectural risks or bottlenecks to watch for.\n"
        "3. One or two alternative pipeline configurations worth considering.\n"
        "Be concise — respond as a numbered list (max 5 items)."
    )

    stage_names = " → ".join(s.name for s in plan.stages)
    forwarding = plan.hazard_unit.forwarding_enabled if plan.hazard_unit else False
    user = (
        f"ISA: {spec.isa_name}\n"
        f"Selected pipeline: {plan.pipeline_name}\n"
        f"Stages: {stage_names}\n"
        f"Forwarding enabled: {forwarding}\n"
        f"Branch predictor: {plan.branch_predictor or 'none'}\n"
        f"Constraints: {spec.constraints}\n\n"
        "Provide your microarchitecture rationale."
    )

    raw = llm_complete(system, user, max_tokens=512, temperature=0.2)
    if not raw:
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MicroarchitectureAgent:
    """
    Deterministically selects and instantiates a microarchitecture plan.

    Usage::

        agent = MicroarchitectureAgent()
        plan  = agent.run(formal_spec)
        print(plan.summary())

    LLM augmentation::

        plan = agent.run(formal_spec, use_llm=True)
        # plan.llm_rationale contains justification and alternatives from LLM
    """

    def run(self, spec: FormalSpec, *, use_llm: bool = False) -> MicroarchPlan:
        """
        Select the appropriate pipeline template based on the constraints
        embedded in *spec* and return the instantiated MicroarchPlan.

        Parameters
        ----------
        spec : FormalSpec
            Formal specification produced by SpecificationAgent.
        use_llm : bool
            When True and an API key is available, populate
            ``MicroarchPlan.llm_rationale`` with LLM-generated analysis.
        """
        pipeline_name: str = spec.constraints.get("pipeline", "FIVE_STAGE")
        forwarding: bool = bool(spec.constraints.get("forwarding", True))

        builder = _TEMPLATE_MAP.get(pipeline_name)
        if builder is None:
            raise ValueError(
                f"Unknown pipeline template '{pipeline_name}'. "
                f"Available: {list(_TEMPLATE_MAP)}"
            )

        # FIVE_STAGE template accepts forwarding flag; others do not
        if pipeline_name == "FIVE_STAGE":
            plan = builder(spec, forwarding)
        else:
            plan = builder(spec)

        # Optional LLM augmentation
        if use_llm:
            plan.llm_rationale = _llm_augment_plan(spec, plan)

        return plan
