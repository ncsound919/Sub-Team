# Design: BusinessAgent — Financial Performance & Sales Intelligence Skills

**Date:** 2026-04-05
**Status:** Approved
**Scope:** Sub-Team repository (`ncsound919/Sub-Team`)

---

## 1. Motivation

The existing Sub-Team agents handle CPU design automation and cross-disciplinary technical
analysis. The agents are now being extended to handle daily and business operational tasks.
The first skills target **business intelligence** — specifically financial performance and
sales/customer metrics — because these are the highest-value daily decision inputs for a
growing business.

---

## 2. Architecture

### 2.1 New class: `BusinessAgent`

Lives in `sub_team/business_agent.py`. Parallel to `CrossDisciplinaryAgent`; does NOT
extend or modify it.

Follows the identical pattern:

```
BusinessProblem  →  BusinessAgent.run()  →  BusinessAnalysis
```

- `BusinessProblem` — input dataclass with `name`, `domains`, `parameters`
- `BusinessInsight` — single finding: `domain`, `finding`, `metric_name`, `metric_value`,
  `confidence`, `risk_level`, `related_domains`
- `BusinessAnalysis` — output report: `insights`, `cross_domain_links`, `overall_risk_score`,
  `recommendations`, plus convenience helpers `insights_for(domain)` and
  `links_involving(domain)`

`BusinessAgent.run(problem, use_llm=False)` accepts an optional `use_llm` flag that calls
`llm_client.llm_complete()` for supplementary LLM commentary — same pattern as all
existing pipeline agents.

### 2.2 Supported domains (initial)

| Domain    | Key question answered                                  |
|-----------|--------------------------------------------------------|
| `finance` | Is the business financially healthy and growing?       |
| `sales`   | Is the revenue pipeline healthy and converting well?   |

Both domains are added to `BUSINESS_DOMAINS` (analogous to `SUPPORTED_DOMAINS`).

### 2.3 Data ingestion: connector layer

Lives in `sub_team/connectors/`. Two thin connector classes:

| Class               | Source   | Env var required     | Returns                        |
|---------------------|----------|----------------------|--------------------------------|
| `StripeConnector`   | Stripe   | `STRIPE_API_KEY`     | Normalised `finance` param dict |
| `HubSpotConnector`  | HubSpot  | `HUBSPOT_API_KEY`    | Normalised `sales` param dict  |

Both connectors:
- Are **optional** — absent key or API failure → `None` returned, no exception raised
- Use the `requests` library (already standard in Python envs; added to `requirements.txt`)
- Provide a `.fetch()` method returning `dict | None`
- Are invoked by the caller, not automatically by `BusinessAgent` — the agent only consumes
  the normalised parameter dict, keeping it testable without live API access

`BusinessAgent.run()` signature:

```python
def run(
    self,
    problem: BusinessProblem,
    *,
    use_llm: bool = False,
) -> BusinessAnalysis:
```

The caller is responsible for merging connector output into `BusinessProblem.parameters`
before calling `.run()`.

---

## 3. Finance Domain

### 3.1 Parameters

| Parameter          | Type    | Default   | Description                                           |
|--------------------|---------|-----------|-------------------------------------------------------|
| `mrr_usd`          | `float` | `0.0`     | Monthly Recurring Revenue in USD                      |
| `mrr_growth_pct`   | `float` | `0.0`     | MoM MRR growth rate (%)                               |
| `arr_usd`          | `float` | `0.0`     | Annual Recurring Revenue (cross-check: should ≈ MRR×12)|
| `gross_margin_pct` | `float` | `0.0`     | Gross margin (%)                                      |
| `cogs_pct`         | `float` | `0.0`     | COGS as % of revenue                                  |
| `burn_rate_usd`    | `float` | `0.0`     | Monthly net cash burn in USD                          |
| `cash_balance_usd` | `float` | `0.0`     | Current cash / cash equivalents in USD                |

All values must be `>= 0`; percentages must be in `[0.0, 100.0]`. Violations raise
`ValueError`.

### 3.2 Rules

| Rule                   | Condition              | Risk    |
|------------------------|------------------------|---------|
| MRR growth             | `< 0%`                 | high    |
|                        | `0–5%`                 | medium  |
|                        | `> 5%`                 | low     |
| Gross margin           | `< 40%`                | high    |
|                        | `40–60%`               | medium  |
|                        | `> 60%`                | low     |
| Runway                 | `< 6 months`           | high    |
| (`cash / burn_rate`)   | `6–12 months`          | medium  |
|                        | `> 12 months`          | low     |
|                        | `burn_rate == 0`       | low (profitable / no burn) |
| ARR consistency        | `abs(arr - mrr*12) > 10%` | medium (flag mismatch) |
| COGS efficiency        | `cogs_pct > 60%`       | high    |
|                        | `40–60%`               | medium  |
|                        | `< 40%`                | low     |

### 3.3 StripeConnector field mapping

| Stripe API field                    | Maps to parameter     |
|-------------------------------------|-----------------------|
| `subscriptions.data[*].plan.amount` | `mrr_usd`             |
| MoM delta of MRR                    | `mrr_growth_pct`      |
| `mrr_usd * 12`                      | `arr_usd`             |
| `balance.available[*].amount`       | `cash_balance_usd`    |

Stripe does not expose gross margin or burn rate — caller must supply those manually or via
a second data source.

---

## 4. Sales Domain

### 4.1 Parameters

| Parameter                | Type    | Default  | Description                                           |
|--------------------------|---------|----------|-------------------------------------------------------|
| `pipeline_value_usd`     | `float` | `0.0`    | Total open pipeline value in USD                      |
| `quota_usd`              | `float` | `0.0`    | Period revenue quota in USD                           |
| `win_rate_pct`           | `float` | `0.0`    | Historical closed-won rate (%)                        |
| `avg_deal_size_usd`      | `float` | `0.0`    | Average deal size in USD                              |
| `avg_sales_cycle_days`   | `float` | `30.0`   | Average days from first touch to close                |
| `churn_rate_pct`         | `float` | `0.0`    | Monthly revenue churn rate (%)                        |
| `ltv_usd`                | `float` | `0.0`    | Customer Lifetime Value in USD                        |
| `cac_usd`                | `float` | `0.0`    | Customer Acquisition Cost in USD                      |
| `ndr_pct`                | `float` | `100.0`  | Net Dollar Retention (%)                              |
| `expansion_mrr_usd`      | `float` | `0.0`    | MRR from expansion (upsell/cross-sell) in USD         |
| `contraction_mrr_usd`    | `float` | `0.0`    | MRR lost to downgrades in USD                         |
| `logo_churn_pct`         | `float` | `0.0`    | Monthly customer count churn rate (%)                 |
| `total_mrr_usd`          | `float` | `0.0`    | Total MRR for this period (used for expansion mix ratio; if 0, expansion mix rule is skipped) |

All values must be `>= 0`; percentages must be in `[0.0, 200.0]` (NDR can exceed 100%).

### 4.2 Rules

| Rule                    | Condition                    | Risk    |
|-------------------------|------------------------------|---------|
| Pipeline coverage       | `pipeline / quota < 3×`      | high    |
|                         | `3–5×`                       | medium  |
|                         | `> 5×`                       | low     |
| Win rate                | `< 20%`                      | high    |
|                         | `20–35%`                     | medium  |
|                         | `> 35%`                      | low     |
| Sales cycle             | `> 90 days`                  | high    |
|                         | `30–90 days`                 | medium  |
|                         | `< 30 days`                  | low     |
| Revenue churn           | `> 5% / month`               | high    |
|                         | `2–5%`                       | medium  |
|                         | `< 2%`                       | low     |
| LTV:CAC ratio           | `< 3×`                       | high    |
| (if both > 0)           | `3–5×`                       | medium  |
|                         | `> 5×`                       | low     |
| Net Dollar Retention    | `< 90%`                      | high    |
|                         | `90–110%`                    | medium  |
|                         | `> 110%`                     | low     |
| Expansion mix           | `expansion_mrr / total_mrr < 10%` (only if `total_mrr > 0`) | medium (flag low expansion) |
| Logo vs revenue churn   | `logo_churn > churn_rate * 1.5` | medium (larger customers churning) |

### 4.3 HubSpotConnector field mapping

| HubSpot API field                          | Maps to parameter        |
|--------------------------------------------|--------------------------|
| `deals?stage=open&sum(amount)`             | `pipeline_value_usd`     |
| `deals?stage=closedwon / total deals`      | `win_rate_pct`           |
| `deals?stage=closedwon&avg(amount)`        | `avg_deal_size_usd`      |
| `deals?avg(close_date - create_date)`      | `avg_sales_cycle_days`   |
| `contacts?count`                           | logo count (used for `logo_churn_pct`) |

HubSpot does not expose LTV, CAC, NDR, or MRR — caller must supply those manually or via
the Stripe connector.

---

## 5. Cross-Domain Links

| Domains               | Link type     | Description                                                                   |
|-----------------------|---------------|-------------------------------------------------------------------------------|
| `finance` ↔ `sales`   | `dependency`  | Pipeline coverage and win rate determine whether revenue targets are achievable given current gross margin and burn runway. |
| `finance` ↔ `fintech` | `synergy`     | Financial risk models and VaR assessments are directly informed by MRR volatility and churn metrics. |
| `sales` ↔ `legal`     | `shared_risk` | Contract terms (liability caps, governing law) directly affect churn behaviour and LTV calculation. |

---

## 6. Error Handling

- Missing required connectors: the agent accepts parameters directly; no connector is
  required. If a connector's `.fetch()` returns `None`, the caller logs a warning and
  can either skip the domain or use defaults.
- Invalid parameter values (negative numbers, out-of-range percentages): `ValueError`
  raised in the domain analyser, same as `cross_disciplinary_agent.py`.
- `use_llm=True` with no API key: gracefully returns empty `llm_commentary` list, no
  exception.

---

## 7. Testing

All tests added to `tests/test_sub_team.py` following existing patterns:

- `TestBusinessAgent` — structure, defaults, determinism
- `TestFinanceDomain` — each rule boundary (high/medium/low)
- `TestSalesDomain` — each rule boundary including NDR, LTV:CAC, logo churn
- `TestStripeConnector` — mocked `requests.get`, tests normalisation + graceful degradation
- `TestHubSpotConnector` — mocked `requests.get`, same
- `TestBusinessLLMAugmentation` — monkeypatched `llm_complete`, same pattern as existing

Target: all existing 108 tests continue to pass; new tests bring total to ~165+.

---

## 8. Files Changed / Created

| File                                      | Action   |
|-------------------------------------------|----------|
| `sub_team/business_agent.py`              | Create   |
| `sub_team/connectors/__init__.py`         | Create   |
| `sub_team/connectors/stripe_connector.py` | Create   |
| `sub_team/connectors/hubspot_connector.py`| Create   |
| `sub_team/__init__.py`                    | Edit (add exports) |
| `requirements.txt`                        | Edit (add `requests>=2.28`) |
| `tests/test_sub_team.py`                  | Edit (add new test classes) |

No existing files are deleted or structurally refactored.

---

## 9. Out of Scope (this iteration)

- Operations & engineering health domain (uptime, error rates, latency)
- People & HR analytics domain
- SaaS efficiency ratios (magic number, Quick Ratio, payback period)
- Sales productivity metrics (rep quota attainment, headcount per ARR)
- Direct database connectors (Postgres, BigQuery, Snowflake)
- Scheduling / task management domain
- Communication / email drafting domain
