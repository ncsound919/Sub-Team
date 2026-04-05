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

### LLM Augmentation (optional)

Every pipeline agent accepts a `use_llm=True` keyword.  When set, the agent
calls an OpenRouter-backed LLM to append advisory notes to its output.  The
deterministic fields are **never modified** — LLM output lands in separate
fields only.

```python
import os
from dotenv import load_dotenv
load_dotenv()          # loads OPENROUTER_API_KEY from .env

spec   = SpecificationAgent().run(cpu, use_llm=True)
plan   = MicroarchitectureAgent().run(spec, use_llm=True)
rtl    = ImplementationAgent().run(spec, plan, use_llm=True)
report = VerificationAgent().run(spec, rtl, use_llm=True)

print(spec.llm_notes)        # List[str] — advisory notes on the spec
print(plan.llm_rationale)    # List[str] — rationale for pipeline choices
print(rtl.llm_review)        # List[str] — code-review observations
print(report.llm_analysis)   # List[str] — deeper analysis of the report
```

**Requirements:** `pip install -r requirements.txt` and set
`OPENROUTER_API_KEY` in your `.env` file (or as an environment variable).
If the key is absent or the `openai` package is not installed, `use_llm=True`
is silently ignored and the deterministic output is returned unchanged.

Check availability at runtime:

```python
from sub_team import llm_available
print(llm_available())   # True / False
```

Override the default model (`openai/gpt-4o-mini`) via:

```bash
SUB_TEAM_LLM_MODEL=anthropic/claude-3-haiku python main.py
```

## Running Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Supported ISAs

`RV32I`, `RV32IM`, `RV32IMA`, `RV64I`, `RV64IM`

## Supported Pipeline Templates

`SINGLE_CYCLE`, `MULTI_CYCLE`, `FIVE_STAGE`, `OUT_OF_ORDER`

---

## Cross-Disciplinary Agent

`CrossDisciplinaryAgent` analyses problems that span multiple knowledge
domains and returns insights, risk scores, and cross-domain dependency links.

### Supported Domains

`logistics`, `biotech`, `fintech`, `probability`, **`legal`** (new)

### Legal Domain

The `legal` domain maps regulatory obligations, scores contract risk, and
identifies compliance gaps.

**Parameters** (pass inside `DomainProblem.params`):

| Parameter | Type | Description |
|---|---|---|
| `data_types` | `list[str]` | Data categories handled (e.g. `["personal_data", "health_records"]`) |
| `jurisdictions` | `list[str]` | Operating jurisdictions (e.g. `["EU", "US", "California"]`) |
| `contract_clause_count` | `int` | Number of clauses in the primary contract |
| `liability_cap_usd` | `float` | Liability cap in USD (`0` = uncapped) |
| `industry` | `str` | Sector: `healthcare`, `finance`, `retail`, `technology`, `logistics`, or `general` |

**Regulatory rules checked:** GDPR, HIPAA, PCI-DSS, SOC2, CCPA

**Example:**

```python
from sub_team import CrossDisciplinaryAgent, DomainProblem

problem = DomainProblem(
    domains=["legal", "fintech"],
    params={
        "data_types": ["personal_data", "payment_data"],
        "jurisdictions": ["EU", "US"],
        "contract_clause_count": 85,
        "liability_cap_usd": 500_000,
        "industry": "finance",
    },
)

result = CrossDisciplinaryAgent().run(problem)
print(result.insights)          # regulatory + contract observations
print(result.risk_score)        # 0.0 – 1.0 combined risk
print(result.cross_domain_links)  # e.g. legal ↔ fintech dependency
```

**Cross-domain links involving `legal`:**

| Link | Type | Rationale |
|---|---|---|
| `legal ↔ fintech` | dependency | Regulatory capital requirements affect fintech product design |
| `legal ↔ biotech` | dependency | FDA/EMA clinical-trial regulations govern biotech timelines |
| `legal ↔ logistics` | shared_risk | Customs and trade-sanctions exposure |
| `legal ↔ probability` | synergy | Probabilistic methods quantify compliance risk |

---

## Business Agent

`BusinessAgent` analyses business problems across **finance** and **sales** domains,
producing risk-scored insights, cross-domain dependency links, and ranked
recommendations.  Like all Sub-Team agents, it is **deterministic** by default.

### Supported Domains

| Domain | Metrics Analysed |
|--------|-----------------|
| `finance` | MRR growth, gross margin, runway, ARR consistency, COGS efficiency |
| `sales` | Pipeline coverage, win rate, sales cycle, churn, LTV:CAC, NDR, expansion mix, logo vs revenue churn |

### Quick Start

```python
from sub_team import BusinessAgent, BusinessProblem

problem = BusinessProblem(
    name="q1-review",
    domains=["finance", "sales"],
    parameters={
        "mrr_usd": 50000,
        "mrr_growth_pct": 8.0,
        "gross_margin_pct": 72.0,
        "burn_rate_usd": 30000,
        "cash_balance_usd": 500000,
        "pipeline_value_usd": 600000,
        "quota_usd": 150000,
        "win_rate_pct": 30.0,
        "churn_rate_pct": 3.0,
        "ltv_usd": 30000,
        "cac_usd": 8000,
        "ndr_pct": 105.0,
    },
)

analysis = BusinessAgent().run(problem)
print(analysis.summary())
print(analysis.overall_risk_score)       # 0.0 - 1.0
print(analysis.insights_for("finance"))  # Finance-specific insights
print(analysis.links_involving("sales")) # Cross-domain links for sales
```

### Connectors (Live Data Ingestion)

Two connectors fetch real-time data from external APIs and return parameter
dictionaries that plug directly into `BusinessProblem.parameters`:

| Connector | API | Env Variable | Returns |
|-----------|-----|-------------|---------|
| `StripeConnector` | Stripe Billing & Balance | `STRIPE_API_KEY` | `mrr_usd`, `arr_usd`, `cash_balance_usd` |
| `HubSpotConnector` | HubSpot CRM Deals | `HUBSPOT_API_KEY` | `pipeline_value_usd`, `win_rate_pct`, `avg_deal_size_usd`, `avg_sales_cycle_days` |

Both connectors degrade gracefully: missing API keys or network errors return
`None` without raising exceptions.

```python
from sub_team.connectors import StripeConnector, HubSpotConnector

stripe_data = StripeConnector().fetch()   # or StripeConnector(api_key="sk_...")
hubspot_data = HubSpotConnector().fetch()

params = {}
if stripe_data:
    params.update(stripe_data)
if hubspot_data:
    params.update(hubspot_data)

problem = BusinessProblem(name="live-review", parameters=params)
analysis = BusinessAgent().run(problem)
```

### LLM Augmentation

Like all agents, `BusinessAgent` supports optional LLM commentary:

```python
analysis = BusinessAgent().run(problem, use_llm=True)
print(analysis.llm_commentary)  # List[str] — strategic observations
```

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
