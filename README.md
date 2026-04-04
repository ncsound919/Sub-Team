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

---

## Uplift Agent Integration

Sub-Team is wrapped by 5 tools in the Uplift Agent, allowing the full CPU design pipeline to be invoked directly from Uplift Agent conversations:

| Uplift Agent Tool | Maps To |
|-------------------|---------|
| `sub_team_run` | Full four-agent pipeline end-to-end |
| `sub_team_spec` | `SpecificationAgent` only |
| `sub_team_microarch` | `MicroarchitectureAgent` only |
| `sub_team_implement` | `ImplementationAgent` only |
| `sub_team_verify` | `VerificationAgent` only |

This means users can invoke CPU RTL generation from natural language within the Uplift Agent, running individual pipeline stages or the full chain.

---

## Draymond Orchestrator Ready

Sub-Team is registered as a **Draymond Orchestrator** entity.

**Draymond** is the central AI agent management dashboard for the Uplift Ecosystem. It orchestrates all agents from a single Marvel-style character roster dashboard — each agent gets a character bio card with capabilities, status, and invocation controls.

**Entity registration:**
- **Slug:** `sub-team`
- **Kind:** `agent`
- **Registered in:** `seed.ts` and `business-chains.ts`

**When connected to Draymond, Sub-Team gains:**
- Centralized health monitoring from the Draymond dashboard
- Multi-agent chain workflows — Sub-Team is the core of the **CPU RTL Generation Pipeline** chain template, which can be triggered from the Draymond dashboard or scheduled to run automatically
- Scheduled job execution managed by Draymond's job scheduler

**Connection is optional** — Sub-Team works identically as a standalone tool. Run `python main.py` or import the agents directly in Python. Draymond integration simply adds orchestration visibility and cross-agent workflow capabilities.

**The Uplift Ecosystem includes:** Uplift Agent, Sports Steve, Sub Team, Megacode, OmniResearch Pro, Social Media Dashboard, Indy Music Platform, TradingAgents, Overlay Chain.
