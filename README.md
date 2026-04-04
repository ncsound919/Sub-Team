# Sub-Team

Deterministic autocoding sub-agent system for **low-code CPU design**.

## Overview

Sub-Team implements the four-agent pipeline described in the `Sub agents`
file.  Instead of probabilistic token generation, every agent uses
formal grammars or constraint-based templates so that the **same CPU
specification always produces the same verified RTL**.

```
CPU spec  →  SpecificationAgent  →  MicroarchitectureAgent
                                          ↓
         VerificationAgent  ←  ImplementationAgent
```

### Agents

| Agent | Role |
|---|---|
| `SpecificationAgent` | Parses an ISA spec into formal register maps, instruction encodings and logic formulas |
| `MicroarchitectureAgent` | Selects and instantiates a pipeline template (CEGIS-inspired constraint solving) |
| `ImplementationAgent` | Generates synthesisable Verilog via grammar-based templates |
| `VerificationAgent` | Runs structural/formal checks and produces a pass/fail report |

## Quick Start

```python
from sub_team import CPU, ISA, PipelineTemplate, SpecificationAgent
from sub_team import MicroarchitectureAgent, ImplementationAgent, VerificationAgent
from sub_team.cpu import gshare

cpu = CPU(
    isa=ISA.RV32IM,
    pipeline=PipelineTemplate.FIVE_STAGE,
    forwarding=True,
    branch_predictor=gshare(bits=8),
)

spec   = SpecificationAgent().run(cpu)
plan   = MicroarchitectureAgent().run(spec)
rtl    = ImplementationAgent().run(spec, plan)
report = VerificationAgent().run(spec, rtl)

rtl.write_to_dir("rtl_out/")
print(report.summary())
```

Or run the full pipeline directly:

```bash
python main.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Supported ISAs

`RV32I`, `RV32IM`, `RV32IMA`, `RV64I`, `RV64IM`

## Supported Pipeline Templates

`SINGLE_CYCLE`, `MULTI_CYCLE`, `FIVE_STAGE`, `OUT_OF_ORDER`
