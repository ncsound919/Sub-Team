"""
Cross-Disciplinary Analysis Agent.

Responsibility
--------------
Input  : A DomainProblem describing a multi-domain scenario.
Output : A CrossDisciplinaryAnalysis report containing:
           - Per-domain insights (logistics, biotech, fintech, probability)
           - Cross-domain synergies identified between disciplines
           - Probability-grounded risk scores
           - Ranked recommendations

Method
------
Rule-based analysis using domain knowledge tables.  Deterministic:
the same DomainProblem always produces the same CrossDisciplinaryAnalysis.

Supported domains
-----------------
- logistics    : supply-chain routing, inventory, demand variability
- biotech      : drug-pipeline analysis, trial-phase risk, compound screening
- fintech      : transaction-risk modelling, liquidity, regulatory scoring
- probability  : distribution fitting, confidence intervals, statistical power
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Supported domain names
# ---------------------------------------------------------------------------

SUPPORTED_DOMAINS: List[str] = ["logistics", "biotech", "fintech", "probability"]

# ---------------------------------------------------------------------------
# Input data structure
# ---------------------------------------------------------------------------


@dataclass
class DomainProblem:
    """
    Describes a problem to be analysed across one or more disciplines.

    Parameters
    ----------
    name : str
        Human-readable problem identifier.
    domains : list[str]
        Subset of SUPPORTED_DOMAINS to include in the analysis.
        Duplicate entries are silently de-duplicated; order is preserved.
    parameters : dict
        Domain-specific key/value pairs consumed by the per-domain
        analysers.  Unknown keys are accepted and may be ignored by the
        analysers; they do not raise an error.

    Logistics parameters
    --------------------
    units : int                   Number of units in the supply chain (default 1000)
    routes : int                  Number of distinct routing paths (default 10)
    demand_variability : float    Coefficient of variation in demand, 0.0–1.0 (default 0.2)
    lead_time_days : int          Average replenishment lead time (default 14)

    Biotech parameters
    ------------------
    compound_count : int          Compounds under evaluation (default 5)
    trial_phase : int             Current clinical trial phase, 1–3 (default 1)
    target_indication : str       Therapeutic target area (default "oncology")

    Fintech parameters
    ------------------
    transaction_volume : int      Daily transaction count (default 10_000)
    risk_tolerance : float        Operator-defined risk tolerance, 0.0–1.0 (default 0.5)
    market_volatility : float     Annualised market volatility σ, 0.0–1.0 (default 0.3)

    Probability parameters
    ----------------------
    sample_size : int             Observations in the dataset (default 100)
    confidence_level : float      Desired confidence level, 0.0–1.0 (default 0.95)
    distribution : str            Assumed distribution ("normal", "poisson", "binomial")
                                  (default "normal")
    """

    name: str
    domains: List[str] = field(default_factory=lambda: list(SUPPORTED_DOMAINS))
    parameters: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
class DomainInsight:
    """A single finding produced for one discipline."""

    domain: str
    finding: str
    confidence: float              # 0.0 (low) – 1.0 (high)
    risk_level: str                # "low", "medium", "high"
    related_domains: List[str] = field(default_factory=list)


@dataclass
class CrossDomainLink:
    """Describes a synergy or shared risk between two disciplines."""

    domains: List[str]             # exactly two domain names
    link_type: str                 # "synergy", "shared_risk", "dependency"
    description: str


@dataclass
class CrossDisciplinaryAnalysis:
    """Complete analysis report produced by CrossDisciplinaryAgent."""

    problem_name: str
    domains_analysed: List[str]
    insights: List[DomainInsight] = field(default_factory=list)
    cross_domain_links: List[CrossDomainLink] = field(default_factory=list)
    overall_risk_score: float = 0.0   # 0.0 (no risk) – 1.0 (critical)
    recommendations: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    def insights_for(self, domain: str) -> List[DomainInsight]:
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
            f"CrossDisciplinaryAnalysis — '{self.problem_name}'",
            f"  Domains analysed : {', '.join(self.domains_analysed)}",
            f"  Overall risk     : {self.overall_risk_score:.2f}  "
            f"({'low' if self.overall_risk_score < 0.4 else 'medium' if self.overall_risk_score < 0.7 else 'high'})",
            "",
            "  Insights:",
        ]
        for ins in self.insights:
            lines.append(
                f"    [{ins.domain:12s}] ({ins.risk_level:6s} | conf={ins.confidence:.2f})  "
                f"{ins.finding}"
            )
        if self.cross_domain_links:
            lines.append("")
            lines.append("  Cross-domain links:")
            for lnk in self.cross_domain_links:
                pair = " ↔ ".join(lnk.domains)
                lines.append(f"    [{lnk.link_type:12s}] {pair}: {lnk.description}")
        if self.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Domain-specific analysis rules (deterministic, no neural inference)
# ---------------------------------------------------------------------------

def _p(params: Dict, key: str, default: object) -> object:
    """Retrieve a parameter with a typed default."""
    return params.get(key, default)


# ── Logistics ──────────────────────────────────────────────────────────────

def _analyse_logistics(params: Dict) -> List[DomainInsight]:
    units: int = int(_p(params, "units", 1000))
    routes: int = int(_p(params, "routes", 10))
    demand_var: float = float(_p(params, "demand_variability", 0.2))
    lead_time: int = int(_p(params, "lead_time_days", 14))

    if lead_time < 0:
        raise ValueError(
            f"lead_time_days must be >= 0, got {lead_time}."
        )

    insights: List[DomainInsight] = []

    # Throughput assessment
    throughput_per_route = units / max(routes, 1)
    if throughput_per_route > 200:
        risk = "high"
        finding = (
            f"High per-route load ({throughput_per_route:.0f} units/route): "
            "route congestion risk; consider load balancing across additional paths."
        )
    elif throughput_per_route > 100:
        risk = "medium"
        finding = (
            f"Moderate per-route load ({throughput_per_route:.0f} units/route): "
            "monitor for seasonal spikes."
        )
    else:
        risk = "low"
        finding = (
            f"Low per-route load ({throughput_per_route:.0f} units/route): "
            "capacity headroom is adequate."
        )
    insights.append(DomainInsight(
        domain="logistics",
        finding=finding,
        confidence=0.85,
        risk_level=risk,
        related_domains=["probability"],
    ))

    # Safety-stock / demand variability
    safety_stock_factor = 1.65 * demand_var * (lead_time ** 0.5)
    if demand_var > 0.5:
        risk = "high"
        finding = (
            f"High demand variability (CV={demand_var:.2f}): "
            f"recommended safety-stock multiplier ≥ {safety_stock_factor:.2f}; "
            "probabilistic reorder-point modelling advised."
        )
    elif demand_var > 0.2:
        risk = "medium"
        finding = (
            f"Moderate demand variability (CV={demand_var:.2f}): "
            f"safety-stock multiplier of {safety_stock_factor:.2f} appropriate."
        )
    else:
        risk = "low"
        finding = (
            f"Low demand variability (CV={demand_var:.2f}): "
            "deterministic reorder-point policy is sufficient."
        )
    insights.append(DomainInsight(
        domain="logistics",
        finding=finding,
        confidence=0.80,
        risk_level=risk,
        related_domains=["probability", "fintech"],
    ))

    # Lead-time risk
    if lead_time > 30:
        insights.append(DomainInsight(
            domain="logistics",
            finding=(
                f"Long lead time ({lead_time} days): supply-chain finance instruments "
                "(e.g. reverse factoring) can reduce working-capital exposure."
            ),
            confidence=0.75,
            risk_level="medium",
            related_domains=["fintech"],
        ))

    return insights


# ── Biotech ────────────────────────────────────────────────────────────────

# Historical average phase-transition success probabilities (industry benchmarks)
_PHASE_SUCCESS: Dict[int, float] = {1: 0.63, 2: 0.31, 3: 0.58}
_PHASE_DURATION_YEARS: Dict[int, float] = {1: 1.5, 2: 2.5, 3: 3.5}


def _analyse_biotech(params: Dict) -> List[DomainInsight]:
    compound_count: int = int(_p(params, "compound_count", 5))
    trial_phase: int = max(1, min(3, int(_p(params, "trial_phase", 1))))
    indication: str = str(_p(params, "target_indication", "oncology"))

    if compound_count < 1:
        raise ValueError(
            f"compound_count must be >= 1, got {compound_count}."
        )

    insights: List[DomainInsight] = []

    # Pipeline probability of at least one success
    phase_prob = _PHASE_SUCCESS[trial_phase]
    # P(≥1 success from N compounds) = 1 − (1 − p)^N
    pipeline_success_prob = 1.0 - (1.0 - phase_prob) ** compound_count
    duration_yrs = _PHASE_DURATION_YEARS[trial_phase]

    risk = (
        "low" if pipeline_success_prob > 0.7
        else "medium" if pipeline_success_prob > 0.4
        else "high"
    )
    insights.append(DomainInsight(
        domain="biotech",
        finding=(
            f"Phase {trial_phase} pipeline ({compound_count} compounds, {indication}): "
            f"P(≥1 approval) ≈ {pipeline_success_prob:.1%} "
            f"(per-compound phase success rate: {phase_prob:.0%}); "
            f"expected phase duration ≈ {duration_yrs:.1f} years."
        ),
        confidence=0.70,
        risk_level=risk,
        related_domains=["probability", "fintech"],
    ))

    # Diversification insight
    if compound_count < 3:
        insights.append(DomainInsight(
            domain="biotech",
            finding=(
                f"Low compound count ({compound_count}): pipeline concentration risk is high. "
                "Expanding to ≥ 3 compounds in parallel significantly improves P(success)."
            ),
            confidence=0.82,
            risk_level="high",
            related_domains=["probability"],
        ))
    elif compound_count >= 8:
        insights.append(DomainInsight(
            domain="biotech",
            finding=(
                f"Large pipeline ({compound_count} compounds): diversification adequate, "
                "but portfolio management costs increase; prioritise compounds by "
                "indication-specific probability of success."
            ),
            confidence=0.78,
            risk_level="low",
            related_domains=["fintech"],
        ))

    return insights


# ── Fintech ────────────────────────────────────────────────────────────────

def _analyse_fintech(params: Dict) -> List[DomainInsight]:
    tx_volume: int = int(_p(params, "transaction_volume", 10_000))
    risk_tol: float = float(_p(params, "risk_tolerance", 0.5))
    volatility: float = float(_p(params, "market_volatility", 0.3))

    insights: List[DomainInsight] = []

    # Fraud-risk index (heuristic: higher volume + lower tolerance → more scrutiny needed)
    fraud_index = (tx_volume / 100_000) * (1.0 - risk_tol)
    if fraud_index > 0.6:
        fraud_risk = "high"
        fraud_note = "real-time anomaly detection and multi-factor authentication are critical."
    elif fraud_index > 0.2:
        fraud_risk = "medium"
        fraud_note = "rule-based transaction monitoring with periodic model refresh is advised."
    else:
        fraud_risk = "low"
        fraud_note = "standard fraud controls are sufficient at this volume and tolerance."
    insights.append(DomainInsight(
        domain="fintech",
        finding=(
            f"Transaction volume {tx_volume:,}/day with risk tolerance {risk_tol:.2f}: "
            f"fraud-risk index = {fraud_index:.2f} — {fraud_note}"
        ),
        confidence=0.80,
        risk_level=fraud_risk,
        related_domains=["probability"],
    ))

    # Value-at-Risk commentary (simplified 1-day 95% VaR proxy)
    # VaR proxy ∝ volatility (no dollar amount; expressed as a normalised score)
    var_score = volatility * (1.0 - risk_tol)
    if var_score > 0.4:
        var_risk = "high"
        var_note = (
            "hedging strategies (options, futures) and dynamic rebalancing are warranted."
        )
    elif var_score > 0.2:
        var_risk = "medium"
        var_note = (
            "periodic stress-testing and scenario analysis recommended."
        )
    else:
        var_risk = "low"
        var_note = "current risk posture is within acceptable bounds."
    insights.append(DomainInsight(
        domain="fintech",
        finding=(
            f"Market volatility σ={volatility:.2f}: normalised VaR proxy = {var_score:.2f} — "
            f"{var_note}"
        ),
        confidence=0.75,
        risk_level=var_risk,
        related_domains=["probability"],
    ))

    return insights


# ── Probability ────────────────────────────────────────────────────────────

# z-scores for common confidence levels
_Z_SCORES: Dict[float, float] = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}
_KNOWN_DISTRIBUTIONS: List[str] = ["normal", "poisson", "binomial"]


def _analyse_probability(params: Dict) -> List[DomainInsight]:
    n: int = int(_p(params, "sample_size", 100))
    conf: float = float(_p(params, "confidence_level", 0.95))
    dist: str = str(_p(params, "distribution", "normal")).lower()

    if n < 1:
        raise ValueError(
            f"sample_size must be >= 1, got {n}."
        )

    insights: List[DomainInsight] = []

    # Determine z-score (nearest supported confidence level)
    if conf in _Z_SCORES:
        z = _Z_SCORES[conf]
    else:
        supported = sorted(_Z_SCORES)
        z = _Z_SCORES[min(supported, key=lambda c: abs(c - conf))]

    # Margin of error for a proportion (worst-case p=0.5)
    margin_of_error = z * 0.5 / (n ** 0.5)
    power_comment = (
        "adequate" if n >= 100
        else "borderline" if n >= 50
        else "insufficient"
    )
    risk = "low" if n >= 100 else "medium" if n >= 50 else "high"
    insights.append(DomainInsight(
        domain="probability",
        finding=(
            f"Sample size n={n} at {conf:.0%} confidence (z={z:.3f}): "
            f"worst-case margin of error = ±{margin_of_error:.3f}; "
            f"statistical power is {power_comment}."
        ),
        confidence=0.90,
        risk_level=risk,
        related_domains=["logistics", "biotech", "fintech"],
    ))

    # Distribution suitability
    if dist not in _KNOWN_DISTRIBUTIONS:
        dist_note = (
            f"Distribution '{dist}' is not one of the recognised options "
            f"({', '.join(_KNOWN_DISTRIBUTIONS)}); falling back to normal approximation."
        )
        dist_risk = "medium"
    elif dist == "normal":
        dist_note = (
            "Normal distribution assumed; verify sample skewness and kurtosis "
            "before relying on parametric tests."
        )
        dist_risk = "low"
    elif dist == "poisson":
        dist_note = (
            "Poisson distribution appropriate for count data with low event rates; "
            "check that mean ≈ variance (equidispersion)."
        )
        dist_risk = "low"
    else:  # binomial
        dist_note = (
            "Binomial distribution appropriate; ensure n·p ≥ 5 and n·(1−p) ≥ 5 "
            "for normal approximation validity."
        )
        dist_risk = "low"
    insights.append(DomainInsight(
        domain="probability",
        finding=dist_note,
        confidence=0.88,
        risk_level=dist_risk,
        related_domains=["biotech", "logistics"],
    ))

    return insights


# ---------------------------------------------------------------------------
# Cross-domain link inference
# ---------------------------------------------------------------------------

_CROSS_DOMAIN_RULES: List[Dict] = [
    {
        "required": {"logistics", "probability"},
        "link_type": "synergy",
        "description": (
            "Probabilistic demand forecasting (e.g. ARIMA, Bayesian) directly improves "
            "logistics reorder-point accuracy and safety-stock optimisation."
        ),
    },
    {
        "required": {"logistics", "fintech"},
        "link_type": "synergy",
        "description": (
            "Supply-chain finance (reverse factoring, dynamic discounting) reduces "
            "working-capital costs and logistics disruption risk."
        ),
    },
    {
        "required": {"biotech", "probability"},
        "link_type": "synergy",
        "description": (
            "Bayesian adaptive trial designs leverage probability theory to reduce "
            "required sample sizes and accelerate phase transitions."
        ),
    },
    {
        "required": {"biotech", "fintech"},
        "link_type": "shared_risk",
        "description": (
            "Drug-pipeline valuation (rNPV) depends on both probability-of-success "
            "estimates and discount-rate assumptions — a key fintech–biotech interface."
        ),
    },
    {
        "required": {"fintech", "probability"},
        "link_type": "dependency",
        "description": (
            "Financial risk models (VaR, CVaR, Monte Carlo simulation) are fundamentally "
            "probabilistic; model uncertainty propagates directly into risk scores."
        ),
    },
    {
        "required": {"logistics", "biotech"},
        "link_type": "dependency",
        "description": (
            "Cold-chain management and temperature-controlled distribution are critical "
            "for biologic and cell-therapy products; logistics failures directly affect "
            "biotech outcomes."
        ),
    },
]


def _build_cross_domain_links(
    active_domains: List[str],
) -> List[CrossDomainLink]:
    active_set = set(active_domains)
    links: List[CrossDomainLink] = []
    for rule in _CROSS_DOMAIN_RULES:
        required: set = rule["required"]  # type: ignore[assignment]
        if required.issubset(active_set):
            links.append(CrossDomainLink(
                domains=sorted(required),
                link_type=rule["link_type"],  # type: ignore[arg-type]
                description=rule["description"],  # type: ignore[arg-type]
            ))
    return links


# ---------------------------------------------------------------------------
# Recommendation synthesis
# ---------------------------------------------------------------------------

def _synthesise_recommendations(
    insights: List[DomainInsight],
    links: List[CrossDomainLink],
) -> List[str]:
    recs: List[str] = []

    high_risk_domains = {i.domain for i in insights if i.risk_level == "high"}
    for domain in sorted(high_risk_domains):
        recs.append(
            f"[{domain}] Address high-risk findings before proceeding to the next stage."
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
# Agent
# ---------------------------------------------------------------------------

_DOMAIN_ANALYSERS = {
    "logistics":   _analyse_logistics,
    "biotech":     _analyse_biotech,
    "fintech":     _analyse_fintech,
    "probability": _analyse_probability,
}


class CrossDisciplinaryAgent:
    """
    Deterministically analyses a multi-domain problem by applying rule-based
    knowledge from logistics, biotech, fintech, and probability.

    Usage::

        agent    = CrossDisciplinaryAgent()
        problem  = DomainProblem(
            name="pharma-cold-chain",
            domains=["logistics", "biotech", "probability"],
            parameters={
                "units": 5000,
                "demand_variability": 0.4,
                "compound_count": 3,
                "trial_phase": 2,
                "sample_size": 200,
            },
        )
        analysis = agent.run(problem)
        print(analysis.summary())
    """

    def run(self, problem: DomainProblem) -> CrossDisciplinaryAnalysis:
        """
        Run cross-disciplinary analysis for *problem*.

        Raises
        ------
        ValueError
            If no recognised domain is present in ``problem.domains``.
        """
        active = [d for d in problem.domains if d in SUPPORTED_DOMAINS]
        unknown = [d for d in problem.domains if d not in SUPPORTED_DOMAINS]

        if not active:
            raise ValueError(
                f"None of the requested domains {problem.domains} are supported. "
                f"Supported: {SUPPORTED_DOMAINS}"
            )

        params = problem.parameters
        all_insights: List[DomainInsight] = []

        for domain in active:
            all_insights.extend(_DOMAIN_ANALYSERS[domain](params))

        links = _build_cross_domain_links(active)

        # Overall risk score: weighted average of per-insight risk levels
        _risk_weight = {"low": 0.1, "medium": 0.5, "high": 0.9}
        if all_insights:
            overall_risk = sum(
                _risk_weight[i.risk_level] * i.confidence for i in all_insights
            ) / sum(i.confidence for i in all_insights)
        else:
            overall_risk = 0.0

        recommendations = _synthesise_recommendations(all_insights, links)

        analysis = CrossDisciplinaryAnalysis(
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
                f"Supported domains: {SUPPORTED_DOMAINS}"
            )

        return analysis
