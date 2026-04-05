"""
Business Intelligence Agent.

Responsibility
--------------
Input  : A BusinessProblem describing a financial / sales scenario.
Output : A BusinessAnalysis report containing:
           - Per-domain insights (finance, sales)
           - Cross-domain links (finance <-> sales, finance <-> fintech, sales <-> legal)
           - Overall risk score
           - Ranked recommendations

Method
------
Rule-based analysis using business knowledge tables.  Deterministic:
the same BusinessProblem always produces the same BusinessAnalysis.

Supported domains
-----------------
- finance  : MRR, ARR, gross margin, burn rate, runway, COGS efficiency
- sales    : pipeline coverage, win rate, sales cycle, churn, LTV:CAC, NDR,
             expansion mix, logo vs revenue churn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Supported domain names
# ---------------------------------------------------------------------------

BUSINESS_DOMAINS: List[str] = ["finance", "sales"]

# ---------------------------------------------------------------------------
# Input data structure
# ---------------------------------------------------------------------------


@dataclass
class BusinessProblem:
    """
    Describes a business problem to be analysed across one or more domains.

    Parameters
    ----------
    name : str
        Human-readable problem identifier.
    domains : list[str]
        Subset of BUSINESS_DOMAINS to include in the analysis.
        Duplicate entries are silently de-duplicated; order is preserved.
    parameters : dict
        Domain-specific key/value pairs consumed by the per-domain
        analysers.  Unknown keys are accepted and may be ignored.
    """

    name: str
    domains: List[str] = field(default_factory=lambda: list(BUSINESS_DOMAINS))
    parameters: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate types
        if not isinstance(self.parameters, dict):
            raise TypeError(
                f"parameters must be a dict, got {type(self.parameters).__name__}."
            )
        for d in self.domains:
            if not isinstance(d, str):
                raise TypeError(
                    f"Each domain must be a string, got {type(d).__name__}: {d!r}."
                )
        # De-duplicate while preserving order
        seen: List[str] = []
        for d in self.domains:
            if d not in seen:
                seen.append(d)
        self.domains = seen


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass
class BusinessInsight:
    """A single finding produced for one business domain."""

    domain: str
    finding: str
    metric_name: str
    metric_value: float
    confidence: float              # 0.0 (low) - 1.0 (high)
    risk_level: str                # "low", "medium", "high"
    related_domains: List[str] = field(default_factory=list)

    _VALID_RISK_LEVELS = {"low", "medium", "high"}

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}."
            )
        if self.risk_level not in self._VALID_RISK_LEVELS:
            raise ValueError(
                f"risk_level must be one of {sorted(self._VALID_RISK_LEVELS)}, "
                f"got {self.risk_level!r}."
            )


@dataclass
class CrossDomainLink:
    """Describes a synergy, dependency, or shared risk between two domains."""

    domains: List[str]             # exactly two domain names
    link_type: str                 # "synergy", "shared_risk", "dependency"
    description: str


@dataclass
class BusinessAnalysis:
    """Complete analysis report produced by BusinessAgent."""

    problem_name: str
    domains_analysed: List[str]
    insights: List[BusinessInsight] = field(default_factory=list)
    cross_domain_links: List[CrossDomainLink] = field(default_factory=list)
    overall_risk_score: float = 0.0   # 0.0 (no risk) - 1.0 (critical)
    recommendations: List[str] = field(default_factory=list)
    llm_commentary: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    def insights_for(self, domain: str) -> List[BusinessInsight]:
        """Return insights belonging to a specific domain."""
        return [i for i in self.insights if i.domain == domain]

    def links_involving(self, domain: str) -> List[CrossDomainLink]:
        """Return cross-domain links that involve a specific domain."""
        return [lnk for lnk in self.cross_domain_links if domain in lnk.domains]

    # ------------------------------------------------------------------ #
    # Human-readable output
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines = [
            f"BusinessAnalysis - '{self.problem_name}'",
            f"  Domains analysed : {', '.join(self.domains_analysed)}",
            f"  Overall risk     : {self.overall_risk_score:.2f}  "
            f"({'low' if self.overall_risk_score < 0.4 else 'medium' if self.overall_risk_score < 0.7 else 'high'})",
            "",
            "  Insights:",
        ]
        for ins in self.insights:
            lines.append(
                f"    [{ins.domain:8s}] ({ins.risk_level:6s} | conf={ins.confidence:.2f})  "
                f"{ins.metric_name}={ins.metric_value:.2f} - {ins.finding}"
            )
        if self.cross_domain_links:
            lines.append("")
            lines.append("  Cross-domain links:")
            for lnk in self.cross_domain_links:
                pair = " <-> ".join(lnk.domains)
                lines.append(f"    [{lnk.link_type:12s}] {pair}: {lnk.description}")
        if self.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
        if self.llm_commentary:
            lines.append("")
            lines.append("  LLM Commentary:")
            for comment in self.llm_commentary:
                lines.append(f"    - {comment}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _p(params: Dict, key: str, default: object) -> object:
    """Retrieve a parameter with a typed default."""
    return params.get(key, default)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_non_negative(name: str, value: float) -> None:
    """Raise ValueError if value is negative."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def _validate_percentage(name: str, value: float, max_val: float = 100.0) -> None:
    """Raise ValueError if value is outside [0, max_val]."""
    if value < 0 or value > max_val:
        raise ValueError(
            f"{name} must be in [0.0, {max_val}], got {value}."
        )


# ---------------------------------------------------------------------------
# Finance domain analysis
# ---------------------------------------------------------------------------

def _analyse_finance(params: Dict) -> List[BusinessInsight]:
    mrr: float = float(_p(params, "mrr_usd", 0.0))
    mrr_growth: float = float(_p(params, "mrr_growth_pct", 0.0))
    arr: float = float(_p(params, "arr_usd", 0.0))
    gross_margin: float = float(_p(params, "gross_margin_pct", 0.0))
    cogs: float = float(_p(params, "cogs_pct", 0.0))
    burn_rate: float = float(_p(params, "burn_rate_usd", 0.0))
    cash_balance: float = float(_p(params, "cash_balance_usd", 0.0))

    # Validate
    _validate_non_negative("mrr_usd", mrr)
    _validate_non_negative("arr_usd", arr)
    _validate_non_negative("burn_rate_usd", burn_rate)
    _validate_non_negative("cash_balance_usd", cash_balance)
    _validate_percentage("gross_margin_pct", gross_margin)
    _validate_percentage("cogs_pct", cogs)
    # mrr_growth_pct can be negative (contraction), but we validate range
    if mrr_growth < -100.0 or mrr_growth > 1000.0:
        raise ValueError(
            f"mrr_growth_pct must be in [-100.0, 1000.0], got {mrr_growth}."
        )

    insights: List[BusinessInsight] = []

    # ── MRR growth ──────────────────────────────────────────────────────────
    if mrr_growth < 0:
        risk = "high"
        finding = (
            f"MRR is contracting at {mrr_growth:.1f}% MoM. "
            "Immediate investigation into churn drivers and pricing is critical."
        )
    elif mrr_growth <= 5.0:
        risk = "medium"
        finding = (
            f"MRR growth of {mrr_growth:.1f}% MoM is below the 5% benchmark. "
            "Consider growth levers: expansion revenue, pricing adjustments, "
            "or new customer acquisition campaigns."
        )
    else:
        risk = "low"
        finding = (
            f"Healthy MRR growth of {mrr_growth:.1f}% MoM. "
            "Continue current growth trajectory and monitor for sustainability."
        )
    insights.append(BusinessInsight(
        domain="finance",
        finding=finding,
        metric_name="mrr_growth_pct",
        metric_value=mrr_growth,
        confidence=0.90,
        risk_level=risk,
        related_domains=["sales"],
    ))

    # ── Gross margin ────────────────────────────────────────────────────────
    if gross_margin < 40.0:
        risk = "high"
        finding = (
            f"Gross margin of {gross_margin:.1f}% is below 40%. "
            "Cost structure needs urgent review; SaaS benchmarks target >60%."
        )
    elif gross_margin <= 60.0:
        risk = "medium"
        finding = (
            f"Gross margin of {gross_margin:.1f}% is moderate. "
            "Opportunities to improve through infrastructure optimisation "
            "and vendor renegotiation."
        )
    else:
        risk = "low"
        finding = (
            f"Strong gross margin of {gross_margin:.1f}%. "
            "Margin is within or above SaaS benchmarks."
        )
    insights.append(BusinessInsight(
        domain="finance",
        finding=finding,
        metric_name="gross_margin_pct",
        metric_value=gross_margin,
        confidence=0.88,
        risk_level=risk,
        related_domains=["sales"],
    ))

    # ── Runway ──────────────────────────────────────────────────────────────
    if burn_rate == 0:
        runway_months = float("inf")
        risk = "low"
        finding = (
            "Zero burn rate indicates profitability or no cash consumption. "
            "No runway risk."
        )
        runway_value = 999.0  # sentinel for metric_value
    else:
        runway_months = cash_balance / burn_rate
        if runway_months < 6:
            risk = "high"
            finding = (
                f"Cash runway of {runway_months:.1f} months is critically short. "
                "Fundraising or expense reduction must begin immediately."
            )
        elif runway_months <= 12:
            risk = "medium"
            finding = (
                f"Cash runway of {runway_months:.1f} months. "
                "Begin fundraising planning within the next quarter."
            )
        else:
            risk = "low"
            finding = (
                f"Cash runway of {runway_months:.1f} months provides adequate buffer. "
                "Continue monitoring burn rate trends."
            )
        runway_value = runway_months
    insights.append(BusinessInsight(
        domain="finance",
        finding=finding,
        metric_name="runway_months",
        metric_value=runway_value,
        confidence=0.92,
        risk_level=risk,
        related_domains=["sales"],
    ))

    # ── ARR consistency check ───────────────────────────────────────────────
    expected_arr = mrr * 12
    if expected_arr > 0 and arr > 0:
        arr_diff_pct = abs(arr - expected_arr) / expected_arr * 100
        if arr_diff_pct > 10:
            insights.append(BusinessInsight(
                domain="finance",
                finding=(
                    f"ARR (${arr:,.0f}) deviates from MRR*12 (${expected_arr:,.0f}) "
                    f"by {arr_diff_pct:.1f}%. Reconcile data sources."
                ),
                metric_name="arr_consistency_gap_pct",
                metric_value=arr_diff_pct,
                confidence=0.85,
                risk_level="medium",
                related_domains=[],
            ))

    # ── COGS efficiency ─────────────────────────────────────────────────────
    if cogs > 60.0:
        risk = "high"
        finding = (
            f"COGS at {cogs:.1f}% of revenue is unsustainably high. "
            "Review vendor contracts, infrastructure costs, and service delivery."
        )
    elif cogs >= 40.0:
        risk = "medium"
        finding = (
            f"COGS at {cogs:.1f}% of revenue is moderate. "
            "Incremental efficiencies may improve margin."
        )
    else:
        risk = "low"
        finding = (
            f"COGS at {cogs:.1f}% of revenue is efficient. "
            "Cost structure is healthy."
        )
    insights.append(BusinessInsight(
        domain="finance",
        finding=finding,
        metric_name="cogs_pct",
        metric_value=cogs,
        confidence=0.85,
        risk_level=risk,
        related_domains=[],
    ))

    return insights


# ---------------------------------------------------------------------------
# Sales domain analysis
# ---------------------------------------------------------------------------

def _analyse_sales(params: Dict) -> List[BusinessInsight]:
    pipeline: float = float(_p(params, "pipeline_value_usd", 0.0))
    quota: float = float(_p(params, "quota_usd", 0.0))
    win_rate: float = float(_p(params, "win_rate_pct", 0.0))
    avg_deal: float = float(_p(params, "avg_deal_size_usd", 0.0))
    cycle_days: float = float(_p(params, "avg_sales_cycle_days", 30.0))
    churn: float = float(_p(params, "churn_rate_pct", 0.0))
    ltv: float = float(_p(params, "ltv_usd", 0.0))
    cac: float = float(_p(params, "cac_usd", 0.0))
    ndr: float = float(_p(params, "ndr_pct", 100.0))
    expansion_mrr: float = float(_p(params, "expansion_mrr_usd", 0.0))
    contraction_mrr: float = float(_p(params, "contraction_mrr_usd", 0.0))
    logo_churn: float = float(_p(params, "logo_churn_pct", 0.0))
    total_mrr: float = float(_p(params, "total_mrr_usd", 0.0))

    # Validate
    _validate_non_negative("pipeline_value_usd", pipeline)
    _validate_non_negative("quota_usd", quota)
    _validate_non_negative("avg_deal_size_usd", avg_deal)
    _validate_non_negative("avg_sales_cycle_days", cycle_days)
    _validate_non_negative("ltv_usd", ltv)
    _validate_non_negative("cac_usd", cac)
    _validate_non_negative("expansion_mrr_usd", expansion_mrr)
    _validate_non_negative("contraction_mrr_usd", contraction_mrr)
    _validate_non_negative("total_mrr_usd", total_mrr)
    _validate_percentage("win_rate_pct", win_rate)
    _validate_percentage("churn_rate_pct", churn)
    _validate_percentage("ndr_pct", ndr, max_val=200.0)
    _validate_percentage("logo_churn_pct", logo_churn)

    insights: List[BusinessInsight] = []

    # ── Pipeline coverage ───────────────────────────────────────────────────
    if quota > 0:
        coverage = pipeline / quota
        if coverage < 3.0:
            risk = "high"
            finding = (
                f"Pipeline coverage of {coverage:.1f}x is below the 3x minimum. "
                "Urgent pipeline generation needed to meet quota."
            )
        elif coverage <= 5.0:
            risk = "medium"
            finding = (
                f"Pipeline coverage of {coverage:.1f}x is adequate but below best-in-class. "
                "Continue pipeline-building activities."
            )
        else:
            risk = "low"
            finding = (
                f"Strong pipeline coverage of {coverage:.1f}x. "
                "Pipeline is healthy relative to quota."
            )
        insights.append(BusinessInsight(
            domain="sales",
            finding=finding,
            metric_name="pipeline_coverage_ratio",
            metric_value=coverage,
            confidence=0.88,
            risk_level=risk,
            related_domains=["finance"],
        ))

    # ── Win rate ────────────────────────────────────────────────────────────
    if win_rate < 20.0:
        risk = "high"
        finding = (
            f"Win rate of {win_rate:.1f}% is critically low. "
            "Review qualification criteria, sales process, and competitive positioning."
        )
    elif win_rate <= 35.0:
        risk = "medium"
        finding = (
            f"Win rate of {win_rate:.1f}% is moderate. "
            "Sales enablement and deal coaching may improve conversion."
        )
    else:
        risk = "low"
        finding = (
            f"Win rate of {win_rate:.1f}% is strong. "
            "Continue refining and documenting winning playbooks."
        )
    insights.append(BusinessInsight(
        domain="sales",
        finding=finding,
        metric_name="win_rate_pct",
        metric_value=win_rate,
        confidence=0.85,
        risk_level=risk,
        related_domains=["finance"],
    ))

    # ── Sales cycle ─────────────────────────────────────────────────────────
    if cycle_days > 90:
        risk = "high"
        finding = (
            f"Average sales cycle of {cycle_days:.0f} days is long. "
            "Evaluate deal complexity, decision-maker access, and proposal turnaround."
        )
    elif cycle_days >= 30:
        risk = "medium"
        finding = (
            f"Average sales cycle of {cycle_days:.0f} days is typical. "
            "Monitor for elongation trends."
        )
    else:
        risk = "low"
        finding = (
            f"Short sales cycle of {cycle_days:.0f} days. "
            "Fast conversion indicates product-market fit or efficient process."
        )
    insights.append(BusinessInsight(
        domain="sales",
        finding=finding,
        metric_name="avg_sales_cycle_days",
        metric_value=cycle_days,
        confidence=0.82,
        risk_level=risk,
        related_domains=["finance"],
    ))

    # ── Revenue churn ───────────────────────────────────────────────────────
    if churn > 5.0:
        risk = "high"
        finding = (
            f"Monthly revenue churn of {churn:.1f}% is critical. "
            "Prioritise retention: customer success, product improvements, "
            "and churn root-cause analysis."
        )
    elif churn >= 2.0:
        risk = "medium"
        finding = (
            f"Monthly revenue churn of {churn:.1f}% is above ideal. "
            "Targeted retention programmes recommended."
        )
    else:
        risk = "low"
        finding = (
            f"Monthly revenue churn of {churn:.1f}% is healthy. "
            "Continue monitoring cohort-level trends."
        )
    insights.append(BusinessInsight(
        domain="sales",
        finding=finding,
        metric_name="churn_rate_pct",
        metric_value=churn,
        confidence=0.88,
        risk_level=risk,
        related_domains=["finance"],
    ))

    # ── LTV:CAC ratio ───────────────────────────────────────────────────────
    if ltv > 0 and cac > 0:
        ltv_cac = ltv / cac
        if ltv_cac < 3.0:
            risk = "high"
            finding = (
                f"LTV:CAC ratio of {ltv_cac:.1f}x is below 3x. "
                "Unit economics are unsustainable; reduce CAC or increase LTV."
            )
        elif ltv_cac <= 5.0:
            risk = "medium"
            finding = (
                f"LTV:CAC ratio of {ltv_cac:.1f}x is acceptable. "
                "Look for ways to improve retention or reduce acquisition cost."
            )
        else:
            risk = "low"
            finding = (
                f"LTV:CAC ratio of {ltv_cac:.1f}x is excellent. "
                "Strong unit economics support growth investment."
            )
        insights.append(BusinessInsight(
            domain="sales",
            finding=finding,
            metric_name="ltv_cac_ratio",
            metric_value=ltv_cac,
            confidence=0.87,
            risk_level=risk,
            related_domains=["finance"],
        ))

    # ── Net Dollar Retention ────────────────────────────────────────────────
    if ndr < 90.0:
        risk = "high"
        finding = (
            f"NDR of {ndr:.1f}% indicates net revenue shrinkage from existing customers. "
            "Critical: expansion revenue must outpace contraction and churn."
        )
    elif ndr <= 110.0:
        risk = "medium"
        finding = (
            f"NDR of {ndr:.1f}% is stable but not expanding. "
            "Invest in upsell/cross-sell motions to drive NDR above 110%."
        )
    else:
        risk = "low"
        finding = (
            f"NDR of {ndr:.1f}% shows strong expansion revenue. "
            "Existing customer base is growing without new logos."
        )
    insights.append(BusinessInsight(
        domain="sales",
        finding=finding,
        metric_name="ndr_pct",
        metric_value=ndr,
        confidence=0.88,
        risk_level=risk,
        related_domains=["finance"],
    ))

    # ── Expansion mix ───────────────────────────────────────────────────────
    if total_mrr > 0:
        expansion_ratio = expansion_mrr / total_mrr
        if expansion_ratio < 0.10:
            insights.append(BusinessInsight(
                domain="sales",
                finding=(
                    f"Expansion MRR is only {expansion_ratio:.1%} of total MRR. "
                    "Low expansion mix suggests limited upsell/cross-sell; "
                    "consider product-led growth or tiered pricing."
                ),
                metric_name="expansion_mix_ratio",
                metric_value=expansion_ratio,
                confidence=0.80,
                risk_level="medium",
                related_domains=["finance"],
            ))

    # ── Logo vs revenue churn ───────────────────────────────────────────────
    if logo_churn > churn * 1.5 and churn > 0:
        insights.append(BusinessInsight(
            domain="sales",
            finding=(
                f"Logo churn ({logo_churn:.1f}%) significantly exceeds revenue churn "
                f"({churn:.1f}%): smaller customers are churning disproportionately. "
                "Consider segment-specific retention strategies or SMB pricing review."
            ),
            metric_name="logo_vs_revenue_churn_ratio",
            metric_value=logo_churn / churn,
            confidence=0.78,
            risk_level="medium",
            related_domains=["finance"],
        ))

    return insights


# ---------------------------------------------------------------------------
# Cross-domain link rules
# ---------------------------------------------------------------------------

_CROSS_DOMAIN_RULES: List[Dict] = [
    {
        "required": {"finance", "sales"},
        "link_type": "dependency",
        "description": (
            "Pipeline coverage and win rate determine whether revenue targets "
            "are achievable given current gross margin and burn runway."
        ),
    },
    {
        "required": {"finance", "fintech"},
        "link_type": "synergy",
        "description": (
            "Financial risk models and VaR assessments are directly informed "
            "by MRR volatility and churn metrics."
        ),
    },
    {
        "required": {"sales", "legal"},
        "link_type": "shared_risk",
        "description": (
            "Contract terms (liability caps, governing law) directly affect "
            "churn behaviour and LTV calculation."
        ),
    },
]


def _build_cross_domain_links(
    active_domains: List[str],
) -> List[CrossDomainLink]:
    active_set = set(active_domains)
    links: List[CrossDomainLink] = []
    for rule in _CROSS_DOMAIN_RULES:
        required: set = rule["required"]
        if required.issubset(active_set):
            links.append(CrossDomainLink(
                domains=sorted(required),
                link_type=rule["link_type"],
                description=rule["description"],
            ))
    return links


# ---------------------------------------------------------------------------
# Recommendation synthesis
# ---------------------------------------------------------------------------

def _synthesise_recommendations(
    insights: List[BusinessInsight],
    links: List[CrossDomainLink],
) -> List[str]:
    recs: List[str] = []

    high_risk_domains = {i.domain for i in insights if i.risk_level == "high"}
    for domain in sorted(high_risk_domains):
        recs.append(
            f"[{domain}] Address high-risk findings before proceeding."
        )

    synergy_links = [lnk for lnk in links if lnk.link_type == "synergy"]
    if synergy_links:
        pair = " & ".join(synergy_links[0].domains).title()
        recs.append(
            f"Exploit {pair} synergy: {synergy_links[0].description.split(';')[0]}."
        )

    dependency_links = [lnk for lnk in links if lnk.link_type == "dependency"]
    if dependency_links:
        pair = " & ".join(dependency_links[0].domains).title()
        recs.append(
            f"Manage {pair} dependency risk proactively: "
            f"{dependency_links[0].description.split(';')[0]}."
        )

    if not recs:
        recs.append("No critical actions required; continue monitoring all domains.")

    return recs


# ---------------------------------------------------------------------------
# Domain analysers registry
# ---------------------------------------------------------------------------

_DOMAIN_ANALYSERS = {
    "finance": _analyse_finance,
    "sales":   _analyse_sales,
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class BusinessAgent:
    """
    Deterministically analyses business problems by applying rule-based
    knowledge from finance and sales domains.

    Usage::

        agent   = BusinessAgent()
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
            },
        )
        analysis = agent.run(problem)
        print(analysis.summary())
    """

    def run(
        self,
        problem: BusinessProblem,
        *,
        use_llm: bool = False,
    ) -> BusinessAnalysis:
        """
        Run business analysis for *problem*.

        Parameters
        ----------
        problem : BusinessProblem
            The business problem to analyse.
        use_llm : bool, optional
            If True, augment the deterministic analysis with LLM commentary.
            Defaults to False.

        Raises
        ------
        ValueError
            If no recognised domain is present in ``problem.domains``.
        """
        active = [d for d in problem.domains if d in BUSINESS_DOMAINS]
        unknown = [d for d in problem.domains if d not in BUSINESS_DOMAINS]

        if not active:
            raise ValueError(
                f"None of the requested domains {problem.domains} are supported. "
                f"Supported: {BUSINESS_DOMAINS}"
            )

        params = problem.parameters
        all_insights: List[BusinessInsight] = []

        for domain in active:
            all_insights.extend(_DOMAIN_ANALYSERS[domain](params))

        links = _build_cross_domain_links(active)

        # Overall risk score: weighted average of per-insight risk levels
        _risk_weight = {"low": 0.1, "medium": 0.5, "high": 0.9}
        if all_insights:
            overall_risk = sum(
                _risk_weight.get(i.risk_level, 0.5) * i.confidence
                for i in all_insights
            ) / sum(i.confidence for i in all_insights)
        else:
            overall_risk = 0.0

        recommendations = _synthesise_recommendations(all_insights, links)

        analysis = BusinessAnalysis(
            problem_name=problem.name,
            domains_analysed=active,
            insights=all_insights,
            cross_domain_links=links,
            overall_risk_score=round(overall_risk, 4),
            recommendations=recommendations,
        )

        if unknown:
            analysis.recommendations.append(
                f"Note: unrecognised domain(s) {unknown} were skipped. "
                f"Supported domains: {BUSINESS_DOMAINS}"
            )

        # ── Optional LLM augmentation ──────────────────────────────────────
        if use_llm:
            analysis.llm_commentary = self._augment_with_llm(analysis)

        return analysis

    def _augment_with_llm(self, analysis: BusinessAnalysis) -> List[str]:
        """Call LLM for supplementary commentary; returns empty list on failure."""
        try:
            from .llm_client import llm_complete
        except ImportError:
            return []

        system_prompt = (
            "You are a business analyst. Given a set of business insights "
            "and metrics, provide concise strategic commentary and action items. "
            "Focus on the most impactful findings."
        )

        # Build user prompt from insights
        insight_lines = []
        for ins in analysis.insights:
            insight_lines.append(
                f"- [{ins.domain}] {ins.metric_name}={ins.metric_value:.2f} "
                f"(risk={ins.risk_level}): {ins.finding}"
            )
        user_prompt = (
            f"Business: {analysis.problem_name}\n"
            f"Domains: {', '.join(analysis.domains_analysed)}\n"
            f"Overall risk: {analysis.overall_risk_score:.2f}\n\n"
            f"Insights:\n" + "\n".join(insight_lines) + "\n\n"
            f"Provide 2-3 strategic observations and action items."
        )

        result = llm_complete(system_prompt, user_prompt)
        if result is None:
            return []

        # Split into individual commentary items
        lines = [line.strip() for line in result.split("\n") if line.strip()]
        return lines if lines else []
