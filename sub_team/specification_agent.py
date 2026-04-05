"""
Specification Agent — Constraint Extraction.

Responsibility
--------------
Input  : A CPU object (natural-language or parametric ISA specification).
Output : A FormalSpec containing:
           - Logic formulas (pre/post-conditions) for each instruction
           - Register map (name → bit-width)
           - Instruction encodings (mnemonic → opcode bit-pattern)

Method
------
Formal grammar-based parsing instead of neural token generation.  All
production rules are deterministic: the same CPU description always yields
the same FormalSpec.

LLM augmentation (optional)
----------------------------
When ``use_llm=True`` is passed to ``SpecificationAgent.run()`` **and** an
OpenRouter API key is present in the environment, the agent additionally asks
the LLM to:

  * Summarise the ISA constraints in plain English.
  * Flag any edge-cases or gaps in the deterministic register/encoding tables.
  * Provide richer pre/post-condition descriptions for complex instructions.

The LLM output is stored in ``FormalSpec.llm_notes`` (a list of strings) and
does **not** modify the deterministic register map, encodings, or formulas.
If the LLM is unavailable the field is simply empty and everything else is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cpu import CPU, ISA


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class RegisterMap:
    """Maps register names to their bit-widths."""
    registers: Dict[str, int] = field(default_factory=dict)

    def add(self, name: str, width: int) -> None:
        self.registers[name] = width

    def __repr__(self) -> str:
        return f"RegisterMap({self.registers})"


@dataclass
class InstructionEncoding:
    """Describes the bit-encoding of a single instruction."""
    mnemonic: str
    opcode: str          # binary string, e.g. "0110011"
    funct3: str = ""     # 3-bit function code (if applicable)
    funct7: str = ""     # 7-bit function code (if applicable)
    format: str = ""     # e.g. "R", "I", "S", "B", "U", "J"

    def __repr__(self) -> str:
        return (
            f"InstructionEncoding({self.mnemonic}, opcode={self.opcode}, "
            f"fmt={self.format})"
        )


@dataclass
class LogicFormula:
    """A single pre/post-condition in a simple conjunctive form."""
    instruction: str
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class FormalSpec:
    """Complete formal specification produced by the SpecificationAgent."""
    isa_name: str
    register_map: RegisterMap = field(default_factory=RegisterMap)
    encodings: List[InstructionEncoding] = field(default_factory=list)
    formulas: List[LogicFormula] = field(default_factory=list)
    constraints: Dict[str, object] = field(default_factory=dict)
    # LLM-generated supplementary notes (empty when LLM is not used / unavailable)
    llm_notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"FormalSpec for {self.isa_name}",
            f"  Registers  : {len(self.register_map.registers)}",
            f"  Encodings  : {len(self.encodings)} instructions",
            f"  Formulas   : {len(self.formulas)} logic assertions",
            f"  Constraints: {list(self.constraints.keys())}",
        ]
        if self.llm_notes:
            lines.append(f"  LLM notes  : {len(self.llm_notes)} items")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ISA grammar tables (deterministic, no neural inference)
# ---------------------------------------------------------------------------

_RV32I_REGISTERS: List[Tuple[str, int]] = [
    ("x0", 32), ("ra", 32), ("sp", 32), ("gp", 32), ("tp", 32),
    ("t0", 32), ("t1", 32), ("t2", 32), ("s0", 32), ("s1", 32),
    ("a0", 32), ("a1", 32), ("a2", 32), ("a3", 32), ("a4", 32),
    ("a5", 32), ("a6", 32), ("a7", 32),
    ("s2", 32), ("s3", 32), ("s4", 32), ("s5", 32), ("s6", 32),
    ("s7", 32), ("s8", 32), ("s9", 32), ("s10", 32), ("s11", 32),
    ("t3", 32), ("t4", 32), ("t5", 32), ("t6", 32),
    ("pc", 32),
]

_RV64I_REGISTERS: List[Tuple[str, int]] = [
    (name, 64) for name, _ in _RV32I_REGISTERS
]

# Subset of RV32I encodings (opcode, funct3, funct7, format)
_RV32I_ENCODINGS: List[Tuple[str, str, str, str, str]] = [
    # mnemonic  opcode      funct3  funct7    fmt
    ("ADD",  "0110011", "000", "0000000", "R"),
    ("SUB",  "0110011", "000", "0100000", "R"),
    ("XOR",  "0110011", "100", "0000000", "R"),
    ("OR",   "0110011", "110", "0000000", "R"),
    ("AND",  "0110011", "111", "0000000", "R"),
    ("SLL",  "0110011", "001", "0000000", "R"),
    ("SRL",  "0110011", "101", "0000000", "R"),
    ("SRA",  "0110011", "101", "0100000", "R"),
    ("ADDI", "0010011", "000", "",        "I"),
    ("XORI", "0010011", "100", "",        "I"),
    ("ORI",  "0010011", "110", "",        "I"),
    ("ANDI", "0010011", "111", "",        "I"),
    ("LW",   "0000011", "010", "",        "I"),
    ("SW",   "0100011", "010", "",        "S"),
    ("BEQ",  "1100011", "000", "",        "B"),
    ("BNE",  "1100011", "001", "",        "B"),
    ("JAL",  "1101111", "",   "",         "J"),
    ("JALR", "1100111", "000", "",        "I"),
    ("LUI",  "0110111", "",   "",         "U"),
    ("AUIPC","0010111", "",   "",         "U"),
]

# Extra encodings for the M extension (multiply/divide)
_RV_M_ENCODINGS: List[Tuple[str, str, str, str, str]] = [
    ("MUL",    "0110011", "000", "0000001", "R"),
    ("MULH",   "0110011", "001", "0000001", "R"),
    ("MULHSU", "0110011", "010", "0000001", "R"),
    ("MULHU",  "0110011", "011", "0000001", "R"),
    ("DIV",    "0110011", "100", "0000001", "R"),
    ("DIVU",   "0110011", "101", "0000001", "R"),
    ("REM",    "0110011", "110", "0000001", "R"),
    ("REMU",   "0110011", "111", "0000001", "R"),
]

_ISA_REGISTER_TABLE: Dict[ISA, List[Tuple[str, int]]] = {
    ISA.RV32I:   _RV32I_REGISTERS,
    ISA.RV32IM:  _RV32I_REGISTERS,
    ISA.RV32IMA: _RV32I_REGISTERS,
    ISA.RV64I:   _RV64I_REGISTERS,
    ISA.RV64IM:  _RV64I_REGISTERS,
}

_ISA_ENCODING_TABLE: Dict[ISA, List[Tuple[str, str, str, str, str]]] = {
    ISA.RV32I:   _RV32I_ENCODINGS,
    ISA.RV32IM:  _RV32I_ENCODINGS + _RV_M_ENCODINGS,
    ISA.RV32IMA: _RV32I_ENCODINGS + _RV_M_ENCODINGS,
    ISA.RV64I:   _RV32I_ENCODINGS,          # instruction encodings identical to RV32I; register widths differ
    ISA.RV64IM:  _RV32I_ENCODINGS + _RV_M_ENCODINGS,
}


def _make_formula(mnemonic: str, fmt: str) -> LogicFormula:
    """Generate deterministic pre/post-conditions from encoding metadata."""
    pre: List[str] = ["valid_instruction(inst)"]
    post: List[str] = []

    if fmt == "R":
        post += [
            f"rd := alu_op_{mnemonic.lower()}(rs1, rs2)",
            "pc := pc + 4",
        ]
    elif fmt == "I" and mnemonic.startswith("L"):
        post += [
            "rd := mem[rs1 + sign_ext(imm12)]",
            "pc := pc + 4",
        ]
    elif fmt == "I":
        post += [
            f"rd := alu_op_{mnemonic.lower()}(rs1, sign_ext(imm12))",
            "pc := pc + 4",
        ]
    elif fmt == "S":
        post += [
            "mem[rs1 + sign_ext(imm12)] := rs2",
            "pc := pc + 4",
        ]
    elif fmt == "B":
        post += [
            f"pc := cond_branch_{mnemonic.lower()}(rs1, rs2) ? pc + sign_ext(imm13) : pc + 4"
        ]
    elif fmt in ("U", "J"):
        post += [
            f"rd := upper_or_jal_{mnemonic.lower()}(pc, imm)",
            "pc := updated_pc",
        ]
    else:
        post += [f"exec_{mnemonic.lower()}()"]

    return LogicFormula(
        instruction=mnemonic,
        preconditions=pre,
        postconditions=post,
    )


# ---------------------------------------------------------------------------
# LLM augmentation helper
# ---------------------------------------------------------------------------

def _llm_augment_spec(spec: "FormalSpec", cpu: "CPU") -> List[str]:
    """
    Ask the LLM for supplementary analysis of the generated FormalSpec.
    Returns a list of note strings, or an empty list if LLM is unavailable.
    """
    try:
        from .llm_client import llm_complete  # local import keeps dependency soft
    except ImportError:
        return []

    system = (
        "You are an expert CPU architect and hardware verification engineer. "
        "Analyse the following ISA formal specification and provide concise "
        "observations about correctness, completeness, edge-cases, and any "
        "improvements or risks. Respond as a numbered list (1–5 items max)."
    )

    mnemonic_list = ", ".join(e.mnemonic for e in spec.encodings)
    user = (
        f"ISA: {spec.isa_name}\n"
        f"Pipeline: {spec.constraints.get('pipeline', 'unknown')}\n"
        f"Forwarding: {spec.constraints.get('forwarding', 'unknown')}\n"
        f"Register count: {len(spec.register_map.registers)}\n"
        f"Instructions ({len(spec.encodings)}): {mnemonic_list}\n"
        f"Constraints: {spec.constraints}\n\n"
        "Please provide your analysis."
    )

    raw = llm_complete(system, user, max_tokens=512, temperature=0.2)
    if not raw:
        return []

    # Split numbered list into individual notes; strip blank lines
    notes = [line.strip() for line in raw.splitlines() if line.strip()]
    return notes


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SpecificationAgent:
    """
    Deterministically extracts a FormalSpec from a CPU description.

    Usage::

        agent = SpecificationAgent()
        spec  = agent.run(cpu)
        print(spec.summary())

    LLM augmentation::

        spec = agent.run(cpu, use_llm=True)
        # spec.llm_notes contains plain-English ISA analysis from the LLM
    """

    def run(self, cpu: CPU, *, use_llm: bool = False) -> FormalSpec:
        """
        Main entry point.  Parses the CPU specification and returns a
        FormalSpec.  Raises ValueError for unsupported ISAs.

        Parameters
        ----------
        cpu : CPU
            The CPU specification to parse.
        use_llm : bool
            When True and an API key is available, augment the spec with
            LLM-generated notes stored in ``FormalSpec.llm_notes``.
            The deterministic register map, encodings, and formulas are
            never modified by the LLM path.
        """
        isa = cpu.isa
        if isa not in _ISA_REGISTER_TABLE:
            raise ValueError(
                f"ISA {isa.name} is not yet supported by SpecificationAgent. "
                f"Supported: {[i.name for i in _ISA_REGISTER_TABLE]}"
            )

        spec = FormalSpec(isa_name=isa.name)

        # 1. Populate register map from grammar table
        for name, width in _ISA_REGISTER_TABLE[isa]:
            spec.register_map.add(name, width)

        # 2. Populate instruction encodings
        for row in _ISA_ENCODING_TABLE[isa]:
            mnemonic, opcode, funct3, funct7, fmt = row
            spec.encodings.append(
                InstructionEncoding(
                    mnemonic=mnemonic,
                    opcode=opcode,
                    funct3=funct3,
                    funct7=funct7,
                    format=fmt,
                )
            )

        # 3. Generate logic formulas for each instruction
        for enc in spec.encodings:
            spec.formulas.append(_make_formula(enc.mnemonic, enc.format))

        # 4. Propagate CPU-level constraints
        spec.constraints["forwarding"] = cpu.forwarding
        spec.constraints["pipeline"] = cpu.pipeline.name
        spec.constraints["max_power"] = cpu.max_power
        spec.constraints["max_area"] = cpu.max_area
        if cpu.target_freq_mhz is not None:
            spec.constraints["target_freq_mhz"] = cpu.target_freq_mhz
        if cpu.branch_predictor is not None:
            spec.constraints["branch_predictor"] = str(cpu.branch_predictor)
        spec.constraints.update(cpu.extra_constraints)

        # 5. Optional LLM augmentation
        if use_llm:
            spec.llm_notes = _llm_augment_spec(spec, cpu)

        return spec
