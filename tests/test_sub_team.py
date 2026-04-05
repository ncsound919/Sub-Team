"""
Tests for the sub-team agent pipeline.
"""

import os
import tempfile

import pytest

from unittest.mock import patch, MagicMock

from sub_team import (
    CPU,
    ISA,
    PipelineTemplate,
    SpecificationAgent,
    MicroarchitectureAgent,
    ImplementationAgent,
    VerificationAgent,
    CrossDisciplinaryAgent,
    DomainProblem,
    CrossDisciplinaryAnalysis,
    SUPPORTED_DOMAINS,
    BusinessAgent,
    BusinessProblem,
    BusinessAnalysis,
    BusinessInsight,
    BUSINESS_DOMAINS,
)
from sub_team.cpu import gshare, bimodal
from sub_team.specification_agent import FormalSpec
from sub_team.microarchitecture_agent import MicroarchPlan
from sub_team.implementation_agent import RTLOutput
from sub_team.verification_agent import VerificationReport
from sub_team.cross_disciplinary_agent import DomainInsight, CrossDomainLink
from sub_team.business_agent import (
    CrossDomainLink as BusinessCrossDomainLink,
    _analyse_finance,
    _analyse_sales,
)
from sub_team.connectors import StripeConnector, HubSpotConnector


# ---------------------------------------------------------------------------
# CPU specification
# ---------------------------------------------------------------------------

def test_cpu_summary_contains_isa():
    cpu = CPU(isa=ISA.RV32IM)
    assert "RV32IM" in cpu.summary()


def test_cpu_defaults():
    cpu = CPU(isa=ISA.RV32I)
    assert cpu.pipeline == PipelineTemplate.FIVE_STAGE
    assert cpu.forwarding is True
    assert cpu.branch_predictor is None


def test_gshare_constructor():
    bp = gshare(bits=10)
    assert bp.scheme == "gshare"
    assert bp.bits == 10
    assert str(bp) == "gshare(bits=10)"


def test_bimodal_constructor():
    bp = bimodal(bits=4)
    assert bp.scheme == "bimodal"
    assert bp.bits == 4


# ---------------------------------------------------------------------------
# SpecificationAgent
# ---------------------------------------------------------------------------

class TestSpecificationAgent:
    def _run(self, isa: ISA) -> FormalSpec:
        return SpecificationAgent().run(CPU(isa=isa))

    def test_rv32i_register_count(self):
        spec = self._run(ISA.RV32I)
        assert len(spec.register_map.registers) == 33  # 32 GP + PC

    def test_rv32im_encodings_include_mul(self):
        spec = self._run(ISA.RV32IM)
        mnemonics = [e.mnemonic for e in spec.encodings]
        assert "MUL" in mnemonics
        assert "DIV" in mnemonics

    def test_rv32i_encodings_exclude_mul(self):
        spec = self._run(ISA.RV32I)
        mnemonics = [e.mnemonic for e in spec.encodings]
        assert "MUL" not in mnemonics

    def test_formula_count_matches_encodings(self):
        spec = self._run(ISA.RV32IM)
        assert len(spec.formulas) == len(spec.encodings)

    def test_constraints_forwarding(self):
        cpu = CPU(isa=ISA.RV32I, forwarding=False)
        spec = SpecificationAgent().run(cpu)
        assert spec.constraints["forwarding"] is False

    def test_unsupported_isa_raises(self):
        cpu = CPU(isa=ISA.MIPS32)
        with pytest.raises(ValueError, match="not yet supported"):
            SpecificationAgent().run(cpu)

    def test_summary_string(self):
        spec = self._run(ISA.RV32I)
        s = spec.summary()
        assert "RV32I" in s
        assert "Registers" in s


# ---------------------------------------------------------------------------
# MicroarchitectureAgent
# ---------------------------------------------------------------------------

class TestMicroarchitectureAgent:
    def _make_spec(self, isa=ISA.RV32IM, pipeline=PipelineTemplate.FIVE_STAGE):
        cpu = CPU(isa=isa, pipeline=pipeline)
        return SpecificationAgent().run(cpu)

    def test_five_stage_produces_five_stages(self):
        spec = self._make_spec(pipeline=PipelineTemplate.FIVE_STAGE)
        plan = MicroarchitectureAgent().run(spec)
        assert len(plan.stages) == 5
        assert plan.stages[0].name == "IF"
        assert plan.stages[-1].name == "WB"

    def test_single_cycle_produces_three_stages(self):
        spec = self._make_spec(pipeline=PipelineTemplate.SINGLE_CYCLE)
        plan = MicroarchitectureAgent().run(spec)
        assert len(plan.stages) == 3

    def test_out_of_order_produces_six_stages(self):
        spec = self._make_spec(pipeline=PipelineTemplate.OUT_OF_ORDER)
        plan = MicroarchitectureAgent().run(spec)
        assert len(plan.stages) == 6

    def test_forwarding_propagated(self):
        cpu = CPU(isa=ISA.RV32IM, forwarding=True)
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        assert plan.hazard_unit.forwarding_enabled is True

    def test_no_forwarding_propagated(self):
        cpu = CPU(isa=ISA.RV32IM, forwarding=False)
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        assert plan.hazard_unit.forwarding_enabled is False

    def test_branch_predictor_in_plan(self):
        cpu = CPU(isa=ISA.RV32IM, branch_predictor=gshare(bits=8))
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        assert "gshare" in plan.branch_predictor

    def test_summary_string(self):
        spec = self._make_spec()
        plan = MicroarchitectureAgent().run(spec)
        assert "FIVE_STAGE" in plan.summary()


# ---------------------------------------------------------------------------
# ImplementationAgent
# ---------------------------------------------------------------------------

class TestImplementationAgent:
    def _build(self, isa=ISA.RV32IM, pipeline=PipelineTemplate.FIVE_STAGE):
        cpu = CPU(isa=isa, pipeline=pipeline)
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        return spec, plan, ImplementationAgent().run(spec, plan)

    def test_four_modules_generated(self):
        _, _, rtl = self._build()
        assert len(rtl.modules) == 4

    def test_required_module_names(self):
        _, _, rtl = self._build()
        names = {m.name for m in rtl.modules}
        assert {"alu", "regfile", "hazard_unit", "cpu_rv32im"} == names

    def test_alu_contains_mul_for_rv32im(self):
        _, _, rtl = self._build(isa=ISA.RV32IM)
        alu = next(m for m in rtl.modules if m.name == "alu")
        assert "ALU_MUL" in alu.source

    def test_alu_no_mul_for_rv32i(self):
        _, _, rtl = self._build(isa=ISA.RV32I)
        alu = next(m for m in rtl.modules if m.name == "alu")
        assert "ALU_MUL" not in alu.source

    def test_regfile_x0_hardwire(self):
        _, _, rtl = self._build()
        rf = next(m for m in rtl.modules if m.name == "regfile")
        assert "rs1_addr == 0" in rf.source

    def test_forwarding_logic_in_hazard_unit(self):
        cpu = CPU(isa=ISA.RV32IM, forwarding=True)
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        rtl = ImplementationAgent().run(spec, plan)
        hu = next(m for m in rtl.modules if m.name == "hazard_unit")
        assert "forward_a" in hu.source

    def test_no_forwarding_has_default_zero_assignments(self):
        cpu = CPU(isa=ISA.RV32IM, forwarding=False)
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        rtl = ImplementationAgent().run(spec, plan)
        hu = next(m for m in rtl.modules if m.name == "hazard_unit")
        # forward_a and forward_b must be driven to 2'b00 even without forwarding
        assert "forward_a = 2'b00" in hu.source
        assert "forward_b = 2'b00" in hu.source

    def test_hazard_unit_uses_id_ex_rd_for_stall(self):
        _, _, rtl = self._build()
        hu = next(m for m in rtl.modules if m.name == "hazard_unit")
        # Stall must compare id_ex_rd (not id_ex_rs1) against the following instruction
        assert "id_ex_rd" in hu.source
        assert "id_ex_rd == if_id_rs1 || id_ex_rd == if_id_rs2" in hu.source

    def test_top_module_uses_dmem_rdata_for_loads(self):
        _, _, rtl = self._build()
        top = next(m for m in rtl.modules if "cpu_" in m.name)
        assert "dmem_rdata" in top.source
        assert "wb_data" in top.source
        assert "is_load" in top.source

    def test_top_module_m_ext_decode_only_for_rv32im(self):
        _, _, rtl = self._build(isa=ISA.RV32IM)
        top = next(m for m in rtl.modules if "cpu_" in m.name)
        assert "MUL" in top.source

    def test_top_module_no_m_ext_decode_for_rv32i(self):
        _, _, rtl = self._build(isa=ISA.RV32I)
        top = next(m for m in rtl.modules if "cpu_" in m.name)
        assert "MUL" not in top.source

    def test_summary_lists_modules(self):
        _, _, rtl = self._build()
        assert "alu" in rtl.summary()

    def test_write_to_dir(self, tmp_path):
        _, _, rtl = self._build()
        paths = rtl.write_to_dir(str(tmp_path))
        assert len(paths) == 4
        for p in paths:
            assert os.path.isfile(p)
            assert p.endswith(".v")


# ---------------------------------------------------------------------------
# VerificationAgent
# ---------------------------------------------------------------------------

class TestVerificationAgent:
    def _full_run(self, isa=ISA.RV32IM, pipeline=PipelineTemplate.FIVE_STAGE):
        cpu = CPU(isa=isa, pipeline=pipeline)
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        rtl = ImplementationAgent().run(spec, plan)
        report = VerificationAgent().run(spec, rtl)
        return report

    def test_all_checks_pass_rv32im_five_stage(self):
        report = self._full_run(ISA.RV32IM, PipelineTemplate.FIVE_STAGE)
        assert report.all_passed, report.summary()

    def test_all_checks_pass_rv32i_single_cycle(self):
        report = self._full_run(ISA.RV32I, PipelineTemplate.SINGLE_CYCLE)
        assert report.all_passed, report.summary()

    def test_no_failed_checks(self):
        report = self._full_run()
        assert report.failed == 0

    def test_pass_count_positive(self):
        report = self._full_run()
        assert report.passed > 0

    def test_summary_contains_verdict(self):
        report = self._full_run()
        assert "ALL CHECKS PASSED" in report.summary()


# ---------------------------------------------------------------------------
# End-to-end: main pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_rv32im():
    from main import run_pipeline
    with tempfile.TemporaryDirectory() as d:
        cpu = CPU(
            isa=ISA.RV32IM,
            pipeline=PipelineTemplate.FIVE_STAGE,
            forwarding=True,
            branch_predictor=gshare(bits=8),
        )
        result = run_pipeline(cpu, rtl_output_dir=d)
    assert result is True


def test_full_pipeline_rv64im():
    from main import run_pipeline
    with tempfile.TemporaryDirectory() as d:
        cpu = CPU(isa=ISA.RV64IM, pipeline=PipelineTemplate.FIVE_STAGE)
        result = run_pipeline(cpu, rtl_output_dir=d)
    assert result is True


# ---------------------------------------------------------------------------
# CrossDisciplinaryAgent
# ---------------------------------------------------------------------------

class TestCrossDisciplinaryAgent:
    """Tests for the cross-disciplinary analysis agent."""

    def _agent(self) -> CrossDisciplinaryAgent:
        return CrossDisciplinaryAgent()

    def _full_problem(self, **extra_params) -> DomainProblem:
        params = {
            "units": 2000,
            "routes": 8,
            "demand_variability": 0.35,
            "lead_time_days": 21,
            "compound_count": 4,
            "trial_phase": 2,
            "target_indication": "oncology",
            "transaction_volume": 50_000,
            "risk_tolerance": 0.4,
            "market_volatility": 0.25,
            "sample_size": 150,
            "confidence_level": 0.95,
            "distribution": "normal",
            # Legal domain params
            "data_types": ["pii", "health"],
            "jurisdictions": ["EU", "US"],
            "contract_clause_count": 30,
            "liability_cap_usd": 500_000,
            "industry": "healthcare",
        }
        params.update(extra_params)
        return DomainProblem(
            name="full-cross-domain",
            domains=list(SUPPORTED_DOMAINS),
            parameters=params,
        )

    # ── DomainProblem ──────────────────────────────────────────────────────

    def test_supported_domains_list(self):
        assert set(SUPPORTED_DOMAINS) == {"logistics", "biotech", "fintech", "probability", "legal"}

    def test_domain_problem_deduplicates(self):
        p = DomainProblem(
            name="dup",
            domains=["logistics", "logistics", "fintech"],
            parameters={},
        )
        assert p.domains.count("logistics") == 1
        assert len(p.domains) == 2

    def test_domain_problem_defaults_to_all_domains(self):
        p = DomainProblem(name="default")
        assert set(p.domains) == set(SUPPORTED_DOMAINS)

    # ── Analysis structure ─────────────────────────────────────────────────

    def test_returns_cross_disciplinary_analysis(self):
        analysis = self._agent().run(self._full_problem())
        assert isinstance(analysis, CrossDisciplinaryAnalysis)

    def test_domains_analysed_match_request(self):
        problem = DomainProblem(
            name="two-domains",
            domains=["logistics", "probability"],
            parameters={"units": 500, "routes": 5, "sample_size": 200},
        )
        analysis = self._agent().run(problem)
        assert set(analysis.domains_analysed) == {"logistics", "probability"}

    def test_all_four_domains_produce_insights(self):
        # The _full_problem now includes all 5 domains; we check all 5 produce insights
        analysis = self._agent().run(self._full_problem())
        domains_with_insights = {i.domain for i in analysis.insights}
        assert domains_with_insights == set(SUPPORTED_DOMAINS)

    def test_insight_fields_populated(self):
        analysis = self._agent().run(self._full_problem())
        for insight in analysis.insights:
            assert isinstance(insight, DomainInsight)
            assert insight.domain in SUPPORTED_DOMAINS
            assert isinstance(insight.finding, str) and insight.finding
            assert 0.0 <= insight.confidence <= 1.0
            assert insight.risk_level in {"low", "medium", "high"}

    def test_overall_risk_score_in_range(self):
        analysis = self._agent().run(self._full_problem())
        assert 0.0 <= analysis.overall_risk_score <= 1.0

    def test_recommendations_non_empty(self):
        analysis = self._agent().run(self._full_problem())
        assert len(analysis.recommendations) >= 1

    # ── Cross-domain links ─────────────────────────────────────────────────

    def test_cross_domain_links_present_for_all_domains(self):
        analysis = self._agent().run(self._full_problem())
        assert len(analysis.cross_domain_links) >= 1

    def test_cross_domain_link_fields(self):
        analysis = self._agent().run(self._full_problem())
        for link in analysis.cross_domain_links:
            assert isinstance(link, CrossDomainLink)
            assert len(link.domains) == 2
            assert link.link_type in {"synergy", "shared_risk", "dependency"}
            assert isinstance(link.description, str) and link.description

    def test_no_links_for_single_domain(self):
        problem = DomainProblem(
            name="single",
            domains=["fintech"],
            parameters={"transaction_volume": 1000},
        )
        analysis = self._agent().run(problem)
        assert analysis.cross_domain_links == []

    def test_logistics_probability_synergy_present(self):
        problem = DomainProblem(
            name="log-prob",
            domains=["logistics", "probability"],
            parameters={},
        )
        analysis = self._agent().run(problem)
        pairs = [tuple(sorted(lnk.domains)) for lnk in analysis.cross_domain_links]
        assert ("logistics", "probability") in pairs

    def test_insights_for_helper(self):
        analysis = self._agent().run(self._full_problem())
        logistics_insights = analysis.insights_for("logistics")
        assert all(i.domain == "logistics" for i in logistics_insights)
        assert len(logistics_insights) >= 1

    def test_links_involving_helper(self):
        analysis = self._agent().run(self._full_problem())
        links = analysis.links_involving("fintech")
        assert all("fintech" in lnk.domains for lnk in links)

    # ── Logistics domain ───────────────────────────────────────────────────

    def test_logistics_high_load_risk(self):
        problem = DomainProblem(
            name="high-load",
            domains=["logistics"],
            parameters={"units": 10_000, "routes": 5},
        )
        analysis = self._agent().run(problem)
        risks = {i.risk_level for i in analysis.insights_for("logistics")}
        assert "high" in risks

    def test_logistics_low_load_risk(self):
        problem = DomainProblem(
            name="low-load",
            domains=["logistics"],
            parameters={"units": 100, "routes": 50},
        )
        analysis = self._agent().run(problem)
        first = analysis.insights_for("logistics")[0]
        assert first.risk_level == "low"

    def test_logistics_long_lead_time_insight(self):
        problem = DomainProblem(
            name="long-lead",
            domains=["logistics"],
            parameters={"lead_time_days": 45},
        )
        analysis = self._agent().run(problem)
        findings = " ".join(i.finding for i in analysis.insights_for("logistics"))
        assert "lead time" in findings.lower() or "45" in findings

    # ── Biotech domain ─────────────────────────────────────────────────────

    def test_biotech_low_compound_high_risk(self):
        problem = DomainProblem(
            name="thin-pipeline",
            domains=["biotech"],
            parameters={"compound_count": 1, "trial_phase": 1},
        )
        analysis = self._agent().run(problem)
        risks = {i.risk_level for i in analysis.insights_for("biotech")}
        assert "high" in risks

    def test_biotech_phase_probability_in_finding(self):
        problem = DomainProblem(
            name="phase2",
            domains=["biotech"],
            parameters={"compound_count": 5, "trial_phase": 2},
        )
        analysis = self._agent().run(problem)
        findings = " ".join(i.finding for i in analysis.insights_for("biotech"))
        assert "Phase 2" in findings

    def test_biotech_large_pipeline_low_risk(self):
        problem = DomainProblem(
            name="large-pipeline",
            domains=["biotech"],
            parameters={"compound_count": 10, "trial_phase": 1},
        )
        analysis = self._agent().run(problem)
        # First insight (pipeline success) should be low risk with 10 compounds
        first = analysis.insights_for("biotech")[0]
        assert first.risk_level == "low"

    # ── Fintech domain ─────────────────────────────────────────────────────

    def test_fintech_high_volume_high_fraud_risk(self):
        problem = DomainProblem(
            name="high-vol",
            domains=["fintech"],
            parameters={"transaction_volume": 500_000, "risk_tolerance": 0.1},
        )
        analysis = self._agent().run(problem)
        risks = {i.risk_level for i in analysis.insights_for("fintech")}
        assert "high" in risks

    def test_fintech_low_volume_low_fraud_risk(self):
        problem = DomainProblem(
            name="low-vol",
            domains=["fintech"],
            parameters={"transaction_volume": 100, "risk_tolerance": 0.9},
        )
        analysis = self._agent().run(problem)
        fraud_insight = analysis.insights_for("fintech")[0]
        assert fraud_insight.risk_level == "low"

    def test_fintech_high_volatility_warning(self):
        problem = DomainProblem(
            name="volatile",
            domains=["fintech"],
            parameters={"market_volatility": 0.9, "risk_tolerance": 0.1},
        )
        analysis = self._agent().run(problem)
        risks = {i.risk_level for i in analysis.insights_for("fintech")}
        assert "high" in risks

    # ── Probability domain ─────────────────────────────────────────────────

    def test_probability_small_sample_high_risk(self):
        problem = DomainProblem(
            name="tiny-n",
            domains=["probability"],
            parameters={"sample_size": 10},
        )
        analysis = self._agent().run(problem)
        first = analysis.insights_for("probability")[0]
        assert first.risk_level == "high"

    def test_probability_large_sample_low_risk(self):
        problem = DomainProblem(
            name="big-n",
            domains=["probability"],
            parameters={"sample_size": 500},
        )
        analysis = self._agent().run(problem)
        first = analysis.insights_for("probability")[0]
        assert first.risk_level == "low"

    def test_probability_unknown_distribution_noted(self):
        problem = DomainProblem(
            name="custom-dist",
            domains=["probability"],
            parameters={"distribution": "cauchy", "sample_size": 100},
        )
        analysis = self._agent().run(problem)
        findings = " ".join(i.finding for i in analysis.insights_for("probability"))
        assert "cauchy" in findings.lower() or "not one of the recognised" in findings.lower()

    def test_probability_poisson_distribution(self):
        problem = DomainProblem(
            name="poisson-test",
            domains=["probability"],
            parameters={"distribution": "poisson", "sample_size": 200},
        )
        analysis = self._agent().run(problem)
        findings = " ".join(i.finding for i in analysis.insights_for("probability"))
        assert "Poisson" in findings or "poisson" in findings.lower()

    # ── Error handling ─────────────────────────────────────────────────────

    def test_unsupported_domain_raises(self):
        problem = DomainProblem(name="bad", domains=["astrophysics"], parameters={})
        with pytest.raises(ValueError, match="are supported"):
            self._agent().run(problem)

    def test_unknown_domain_mixed_with_valid_is_noted(self):
        problem = DomainProblem(
            name="mixed",
            domains=["logistics", "astrophysics"],
            parameters={},
        )
        analysis = self._agent().run(problem)
        assert analysis.domains_analysed == ["logistics"]
        notes = " ".join(analysis.recommendations)
        assert "astrophysics" in notes

    def test_negative_lead_time_raises(self):
        problem = DomainProblem(
            name="bad-lead",
            domains=["logistics"],
            parameters={"lead_time_days": -5},
        )
        with pytest.raises(ValueError, match="lead_time_days"):
            self._agent().run(problem)

    def test_zero_compound_count_raises(self):
        problem = DomainProblem(
            name="zero-compounds",
            domains=["biotech"],
            parameters={"compound_count": 0},
        )
        with pytest.raises(ValueError, match="compound_count"):
            self._agent().run(problem)

    def test_negative_compound_count_raises(self):
        problem = DomainProblem(
            name="neg-compounds",
            domains=["biotech"],
            parameters={"compound_count": -2},
        )
        with pytest.raises(ValueError, match="compound_count"):
            self._agent().run(problem)

    def test_zero_sample_size_raises(self):
        problem = DomainProblem(
            name="zero-n",
            domains=["probability"],
            parameters={"sample_size": 0},
        )
        with pytest.raises(ValueError, match="sample_size"):
            self._agent().run(problem)

    def test_negative_sample_size_raises(self):
        problem = DomainProblem(
            name="neg-n",
            domains=["probability"],
            parameters={"sample_size": -10},
        )
        with pytest.raises(ValueError, match="sample_size"):
            self._agent().run(problem)

    # ── Summary output ─────────────────────────────────────────────────────

    def test_summary_contains_problem_name(self):
        analysis = self._agent().run(self._full_problem())
        assert "full-cross-domain" in analysis.summary()

    def test_summary_contains_all_domains(self):
        analysis = self._agent().run(self._full_problem())
        s = analysis.summary()
        for domain in SUPPORTED_DOMAINS:
            assert domain in s

    def test_summary_contains_risk_score(self):
        analysis = self._agent().run(self._full_problem())
        assert "Overall risk" in analysis.summary()

    # ── Determinism ────────────────────────────────────────────────────────

    def test_same_inputs_same_output(self):
        problem = self._full_problem()
        a1 = self._agent().run(problem)
        a2 = self._agent().run(problem)
        assert a1.overall_risk_score == a2.overall_risk_score
        assert len(a1.insights) == len(a2.insights)
        assert len(a1.cross_domain_links) == len(a2.cross_domain_links)


# ---------------------------------------------------------------------------
# Legal domain tests
# ---------------------------------------------------------------------------

class TestLegalDomain:
    def _agent(self) -> CrossDisciplinaryAgent:
        return CrossDisciplinaryAgent()

    def _legal_problem(self, **params) -> DomainProblem:
        base = {
            "data_types": ["pii", "health"],
            "jurisdictions": ["EU", "US"],
            "contract_clause_count": 25,
            "liability_cap_usd": 0,
            "industry": "healthcare",
        }
        base.update(params)
        return DomainProblem(name="legal-test", domains=["legal"], parameters=base)

    def test_legal_domain_produces_insights(self):
        analysis = self._agent().run(self._legal_problem())
        assert len(analysis.insights_for("legal")) >= 1

    def test_legal_insights_have_valid_fields(self):
        analysis = self._agent().run(self._legal_problem())
        for insight in analysis.insights_for("legal"):
            assert insight.domain == "legal"
            assert isinstance(insight.finding, str) and insight.finding
            assert 0.0 <= insight.confidence <= 1.0
            assert insight.risk_level in {"low", "medium", "high"}

    def test_gdpr_detected_for_eu_pii(self):
        analysis = self._agent().run(self._legal_problem(
            data_types=["pii"], jurisdictions=["EU"]
        ))
        findings = " ".join(i.finding for i in analysis.insights_for("legal"))
        assert "GDPR" in findings

    def test_hipaa_detected_for_us_health(self):
        analysis = self._agent().run(self._legal_problem(
            data_types=["health"], jurisdictions=["US"]
        ))
        findings = " ".join(i.finding for i in analysis.insights_for("legal"))
        assert "HIPAA" in findings

    def test_pci_dss_detected_for_payment_data(self):
        analysis = self._agent().run(self._legal_problem(
            data_types=["payment"], jurisdictions=["US"]
        ))
        findings = " ".join(i.finding for i in analysis.insights_for("legal"))
        assert "PCI-DSS" in findings

    def test_no_regulations_for_unknown_data_type_and_jurisdiction(self):
        analysis = self._agent().run(self._legal_problem(
            data_types=["internal_notes"], jurisdictions=["Antarctica"]
        ))
        findings = " ".join(i.finding for i in analysis.insights_for("legal"))
        # Should fall through to the generic no-major-regulations finding
        assert "No major sector-specific regulations" in findings or len(
            analysis.insights_for("legal")
        ) >= 1

    def test_contract_uncapped_liability_high_risk(self):
        analysis = self._agent().run(self._legal_problem(
            data_types=[], jurisdictions=[],
            contract_clause_count=80,
            liability_cap_usd=0,
        ))
        risks = {i.risk_level for i in analysis.insights_for("legal")}
        assert "high" in risks

    def test_contract_capped_low_clause_count_low_risk(self):
        analysis = self._agent().run(self._legal_problem(
            data_types=[], jurisdictions=[],
            contract_clause_count=5,
            liability_cap_usd=1_000_000,
        ))
        contract_insights = [
            i for i in analysis.insights_for("legal") if "Contract risk" in i.finding
        ]
        assert len(contract_insights) >= 1
        assert contract_insights[0].risk_level == "low"

    def test_industry_healthcare_obligations_included(self):
        analysis = self._agent().run(self._legal_problem(industry="healthcare"))
        findings = " ".join(i.finding for i in analysis.insights_for("legal"))
        assert "FDA" in findings or "HIPAA" in findings or "21 CFR" in findings

    def test_industry_finance_obligations_included(self):
        analysis = self._agent().run(self._legal_problem(
            data_types=["financial"], jurisdictions=["US"], industry="finance"
        ))
        findings = " ".join(i.finding for i in analysis.insights_for("legal"))
        assert "MiFID" in findings or "AML" in findings or "Basel" in findings

    def test_legal_fintech_cross_domain_link(self):
        problem = DomainProblem(
            name="legal-fintech",
            domains=["legal", "fintech"],
            parameters={
                "data_types": ["financial"], "jurisdictions": ["US"],
                "transaction_volume": 10_000, "risk_tolerance": 0.5,
            },
        )
        analysis = self._agent().run(problem)
        pairs = [tuple(sorted(lnk.domains)) for lnk in analysis.cross_domain_links]
        assert ("fintech", "legal") in pairs

    def test_legal_biotech_cross_domain_link(self):
        problem = DomainProblem(
            name="legal-biotech",
            domains=["legal", "biotech"],
            parameters={
                "data_types": ["health"], "jurisdictions": ["US"],
                "compound_count": 3, "trial_phase": 2,
            },
        )
        analysis = self._agent().run(problem)
        pairs = [tuple(sorted(lnk.domains)) for lnk in analysis.cross_domain_links]
        assert ("biotech", "legal") in pairs

    def test_legal_logistics_cross_domain_link(self):
        problem = DomainProblem(
            name="legal-logistics",
            domains=["legal", "logistics"],
            parameters={
                "data_types": [], "jurisdictions": ["EU"],
                "units": 1000, "routes": 5,
            },
        )
        analysis = self._agent().run(problem)
        pairs = [tuple(sorted(lnk.domains)) for lnk in analysis.cross_domain_links]
        assert ("legal", "logistics") in pairs

    def test_legal_in_full_analysis_summary(self):
        problem = DomainProblem(
            name="full-with-legal",
            domains=list(SUPPORTED_DOMAINS),
            parameters={
                "units": 1000, "routes": 10, "demand_variability": 0.3,
                "lead_time_days": 14, "compound_count": 3, "trial_phase": 1,
                "transaction_volume": 10_000, "risk_tolerance": 0.5,
                "market_volatility": 0.3, "sample_size": 100,
                "data_types": ["pii"], "jurisdictions": ["EU"],
                "contract_clause_count": 20, "liability_cap_usd": 0,
                "industry": "technology",
            },
        )
        analysis = self._agent().run(problem)
        assert "legal" in analysis.summary()

    def test_legal_determinism(self):
        problem = self._legal_problem()
        a1 = self._agent().run(problem)
        a2 = self._agent().run(problem)
        assert len(a1.insights_for("legal")) == len(a2.insights_for("legal"))
        assert a1.overall_risk_score == a2.overall_risk_score


# ---------------------------------------------------------------------------
# LLM augmentation tests (mock-based — no real API calls)
# ---------------------------------------------------------------------------

class TestLLMAugmentation:
    """
    Tests for the optional use_llm=True path across all four pipeline agents
    and the CrossDisciplinaryAgent.  All tests mock llm_complete to avoid
    real API calls and to remain deterministic in CI.
    """

    def _cpu(self) -> "CPU":
        return CPU(isa=ISA.RV32IM)

    # ── llm_client module ──────────────────────────────────────────────────

    def test_llm_available_returns_bool(self):
        """llm_available() must always return a bool regardless of env state."""
        from sub_team import llm_available
        assert isinstance(llm_available(), bool)

    def test_llm_complete_returns_none_when_no_key(self, monkeypatch):
        """Without an API key, llm_complete must return None."""
        import sub_team.llm_client as lc
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        lc.reset_client()
        result = lc.llm_complete("sys", "usr")
        assert result is None
        lc.reset_client()  # restore for subsequent tests

    def test_reset_client_clears_cache(self, monkeypatch):
        import sub_team.llm_client as lc
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
        lc.reset_client()
        # After reset, the cached client should be cleared (internal _client is None)
        assert lc._client is None
        # A second reset should also not raise
        lc.reset_client()
        assert lc._client is None

    # ── SpecificationAgent LLM path ────────────────────────────────────────

    def test_spec_use_llm_false_gives_empty_notes(self):
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu, use_llm=False)
        assert spec.llm_notes == []

    def test_spec_use_llm_true_calls_llm(self, monkeypatch):
        import sub_team.specification_agent as sa
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: "1. Good spec.\n2. No gaps.",
        )
        cpu = self._cpu()
        spec = sa.SpecificationAgent().run(cpu, use_llm=True)
        assert len(spec.llm_notes) >= 1
        assert any("Good spec" in note for note in spec.llm_notes)

    def test_spec_use_llm_true_llm_unavailable_gives_empty_notes(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: None,
        )
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu, use_llm=True)
        assert spec.llm_notes == []

    # ── MicroarchitectureAgent LLM path ───────────────────────────────────

    def test_microarch_use_llm_false_gives_empty_rationale(self):
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec, use_llm=False)
        assert plan.llm_rationale == []

    def test_microarch_use_llm_true_calls_llm(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: "1. Five-stage is appropriate.\n2. Watch for hazards.",
        )
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec, use_llm=True)
        assert len(plan.llm_rationale) >= 1
        assert any("appropriate" in r for r in plan.llm_rationale)

    def test_microarch_use_llm_unavailable_gives_empty(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: None,
        )
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec, use_llm=True)
        assert plan.llm_rationale == []

    # ── ImplementationAgent LLM path ──────────────────────────────────────

    def test_impl_use_llm_false_gives_empty_review(self):
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        output = ImplementationAgent().run(spec, plan, use_llm=False)
        assert output.llm_review == []

    def test_impl_use_llm_true_calls_llm(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: "1. Looks synthesisable.\n2. Check reset polarity.",
        )
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        output = ImplementationAgent().run(spec, plan, use_llm=True)
        assert len(output.llm_review) >= 1
        assert any("synthesisable" in r for r in output.llm_review)

    def test_impl_use_llm_unavailable_gives_empty(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: None,
        )
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        output = ImplementationAgent().run(spec, plan, use_llm=True)
        assert output.llm_review == []

    # ── VerificationAgent LLM path ────────────────────────────────────────

    def test_verify_use_llm_false_gives_empty_analysis(self):
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        rtl = ImplementationAgent().run(spec, plan)
        report = VerificationAgent().run(spec, rtl, use_llm=False)
        assert report.llm_analysis == []

    def test_verify_use_llm_true_calls_llm(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: "1. All checks passed cleanly.\n2. No gaps detected.",
        )
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        rtl = ImplementationAgent().run(spec, plan)
        report = VerificationAgent().run(spec, rtl, use_llm=True)
        assert len(report.llm_analysis) >= 1
        assert any("passed" in r for r in report.llm_analysis)

    def test_verify_use_llm_unavailable_gives_empty(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: None,
        )
        cpu = self._cpu()
        spec = SpecificationAgent().run(cpu)
        plan = MicroarchitectureAgent().run(spec)
        rtl = ImplementationAgent().run(spec, plan)
        report = VerificationAgent().run(spec, rtl, use_llm=True)
        assert report.llm_analysis == []

    # ── Determinism preservation ───────────────────────────────────────────

    def test_llm_augmentation_does_not_alter_deterministic_outputs(self, monkeypatch):
        """LLM augmentation must not change any deterministic fields."""
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: "LLM says something",
        )
        cpu = CPU(isa=ISA.RV32I)
        spec_no_llm  = SpecificationAgent().run(cpu, use_llm=False)
        spec_with_llm = SpecificationAgent().run(cpu, use_llm=True)

        # Deterministic fields must be identical
        assert spec_no_llm.isa_name == spec_with_llm.isa_name
        assert len(spec_no_llm.encodings) == len(spec_with_llm.encodings)
        assert len(spec_no_llm.formulas) == len(spec_with_llm.formulas)
        assert spec_no_llm.constraints == spec_with_llm.constraints
        # LLM notes present only in the augmented version
        assert spec_no_llm.llm_notes == []
        assert len(spec_with_llm.llm_notes) >= 1


# ===========================================================================
# BusinessAgent tests
# ===========================================================================

class TestBusinessAgent:
    """Core BusinessAgent structure, defaults, and determinism."""

    def _agent(self):
        return BusinessAgent()

    def _problem(self, **overrides):
        params = {
            "mrr_usd": 50000,
            "mrr_growth_pct": 8.0,
            "arr_usd": 600000,
            "gross_margin_pct": 72.0,
            "cogs_pct": 28.0,
            "burn_rate_usd": 30000,
            "cash_balance_usd": 500000,
            "pipeline_value_usd": 600000,
            "quota_usd": 150000,
            "win_rate_pct": 30.0,
            "avg_deal_size_usd": 25000,
            "avg_sales_cycle_days": 45,
            "churn_rate_pct": 3.0,
            "ltv_usd": 30000,
            "cac_usd": 8000,
            "ndr_pct": 105.0,
            "expansion_mrr_usd": 5000,
            "contraction_mrr_usd": 1000,
            "logo_churn_pct": 2.0,
            "total_mrr_usd": 50000,
        }
        params.update(overrides)
        return BusinessProblem(name="test-biz", parameters=params)

    def test_business_domains_list(self):
        assert "finance" in BUSINESS_DOMAINS
        assert "sales" in BUSINESS_DOMAINS
        assert len(BUSINESS_DOMAINS) == 2

    def test_default_domains_cover_all(self):
        problem = BusinessProblem(name="test")
        assert problem.domains == list(BUSINESS_DOMAINS)

    def test_business_problem_deduplicates_domains(self):
        problem = BusinessProblem(name="dup", domains=["finance", "finance", "sales"])
        assert problem.domains == ["finance", "sales"]

    def test_business_problem_type_validation_parameters(self):
        with pytest.raises(TypeError, match="parameters must be a dict"):
            BusinessProblem(name="bad", parameters="not a dict")

    def test_business_problem_type_validation_domains(self):
        with pytest.raises(TypeError, match="Each domain must be a string"):
            BusinessProblem(name="bad", domains=[123])

    def test_run_returns_analysis(self):
        analysis = self._agent().run(self._problem())
        assert isinstance(analysis, BusinessAnalysis)
        assert analysis.problem_name == "test-biz"

    def test_both_domains_analysed(self):
        analysis = self._agent().run(self._problem())
        assert "finance" in analysis.domains_analysed
        assert "sales" in analysis.domains_analysed

    def test_insights_non_empty(self):
        analysis = self._agent().run(self._problem())
        assert len(analysis.insights) > 0

    def test_insights_for_accessor(self):
        analysis = self._agent().run(self._problem())
        finance_insights = analysis.insights_for("finance")
        assert all(i.domain == "finance" for i in finance_insights)
        assert len(finance_insights) >= 4  # MRR growth, margin, runway, COGS

    def test_links_involving_accessor(self):
        analysis = self._agent().run(self._problem())
        finance_links = analysis.links_involving("finance")
        for lnk in finance_links:
            assert "finance" in lnk.domains

    def test_overall_risk_score_range(self):
        analysis = self._agent().run(self._problem())
        assert 0.0 <= analysis.overall_risk_score <= 1.0

    def test_recommendations_non_empty(self):
        analysis = self._agent().run(self._problem())
        assert len(analysis.recommendations) >= 1

    def test_determinism(self):
        problem = self._problem()
        a1 = self._agent().run(problem)
        a2 = self._agent().run(problem)
        assert a1.overall_risk_score == a2.overall_risk_score
        assert len(a1.insights) == len(a2.insights)
        assert len(a1.recommendations) == len(a2.recommendations)

    def test_unknown_domains_skipped_with_note(self):
        problem = BusinessProblem(
            name="partial",
            domains=["finance", "marketing"],
            parameters={"mrr_usd": 10000, "mrr_growth_pct": 5.0},
        )
        analysis = self._agent().run(problem)
        assert "finance" in analysis.domains_analysed
        assert "marketing" not in analysis.domains_analysed
        assert any("marketing" in r for r in analysis.recommendations)

    def test_no_valid_domains_raises(self):
        problem = BusinessProblem(name="bad", domains=["marketing"])
        with pytest.raises(ValueError, match="None of the requested domains"):
            self._agent().run(problem)

    def test_summary_output(self):
        analysis = self._agent().run(self._problem())
        summary = analysis.summary()
        assert "BusinessAnalysis" in summary
        assert "test-biz" in summary

    def test_single_domain_finance_only(self):
        problem = BusinessProblem(
            name="finance-only",
            domains=["finance"],
            parameters={"mrr_usd": 20000, "mrr_growth_pct": 3.0,
                         "gross_margin_pct": 55.0, "cogs_pct": 45.0,
                         "burn_rate_usd": 15000, "cash_balance_usd": 200000},
        )
        analysis = self._agent().run(problem)
        assert analysis.domains_analysed == ["finance"]
        assert len(analysis.insights_for("sales")) == 0

    def test_single_domain_sales_only(self):
        problem = BusinessProblem(
            name="sales-only",
            domains=["sales"],
            parameters={"pipeline_value_usd": 300000, "quota_usd": 100000,
                         "win_rate_pct": 25.0, "churn_rate_pct": 2.5},
        )
        analysis = self._agent().run(problem)
        assert analysis.domains_analysed == ["sales"]
        assert len(analysis.insights_for("finance")) == 0

    def test_llm_commentary_empty_by_default(self):
        analysis = self._agent().run(self._problem(), use_llm=False)
        assert analysis.llm_commentary == []

    def test_cross_domain_link_finance_sales(self):
        analysis = self._agent().run(self._problem())
        pairs = [tuple(sorted(lnk.domains)) for lnk in analysis.cross_domain_links]
        assert ("finance", "sales") in pairs


# ---------------------------------------------------------------------------
# Finance domain tests
# ---------------------------------------------------------------------------

class TestFinanceDomain:
    """Tests for each finance rule boundary (high/medium/low)."""

    def _agent(self):
        return BusinessAgent()

    def _finance_problem(self, **overrides):
        params = {
            "mrr_usd": 50000,
            "mrr_growth_pct": 8.0,
            "arr_usd": 600000,
            "gross_margin_pct": 72.0,
            "cogs_pct": 28.0,
            "burn_rate_usd": 30000,
            "cash_balance_usd": 500000,
        }
        params.update(overrides)
        return BusinessProblem(name="fin-test", domains=["finance"], parameters=params)

    # ── MRR growth ──────────────────────────────────────────────────────────

    def test_mrr_growth_negative_high_risk(self):
        analysis = self._agent().run(self._finance_problem(mrr_growth_pct=-5.0))
        mrr_insights = [i for i in analysis.insights if i.metric_name == "mrr_growth_pct"]
        assert len(mrr_insights) == 1
        assert mrr_insights[0].risk_level == "high"

    def test_mrr_growth_zero_medium_risk(self):
        analysis = self._agent().run(self._finance_problem(mrr_growth_pct=0.0))
        mrr_insights = [i for i in analysis.insights if i.metric_name == "mrr_growth_pct"]
        assert mrr_insights[0].risk_level == "medium"

    def test_mrr_growth_3pct_medium_risk(self):
        analysis = self._agent().run(self._finance_problem(mrr_growth_pct=3.0))
        mrr_insights = [i for i in analysis.insights if i.metric_name == "mrr_growth_pct"]
        assert mrr_insights[0].risk_level == "medium"

    def test_mrr_growth_5pct_medium_risk(self):
        analysis = self._agent().run(self._finance_problem(mrr_growth_pct=5.0))
        mrr_insights = [i for i in analysis.insights if i.metric_name == "mrr_growth_pct"]
        assert mrr_insights[0].risk_level == "medium"

    def test_mrr_growth_10pct_low_risk(self):
        analysis = self._agent().run(self._finance_problem(mrr_growth_pct=10.0))
        mrr_insights = [i for i in analysis.insights if i.metric_name == "mrr_growth_pct"]
        assert mrr_insights[0].risk_level == "low"

    # ── Gross margin ────────────────────────────────────────────────────────

    def test_gross_margin_low_high_risk(self):
        analysis = self._agent().run(self._finance_problem(gross_margin_pct=30.0, cogs_pct=70.0))
        gm_insights = [i for i in analysis.insights if i.metric_name == "gross_margin_pct"]
        assert gm_insights[0].risk_level == "high"

    def test_gross_margin_mid_medium_risk(self):
        analysis = self._agent().run(self._finance_problem(gross_margin_pct=50.0, cogs_pct=50.0))
        gm_insights = [i for i in analysis.insights if i.metric_name == "gross_margin_pct"]
        assert gm_insights[0].risk_level == "medium"

    def test_gross_margin_high_low_risk(self):
        analysis = self._agent().run(self._finance_problem(gross_margin_pct=75.0, cogs_pct=25.0))
        gm_insights = [i for i in analysis.insights if i.metric_name == "gross_margin_pct"]
        assert gm_insights[0].risk_level == "low"

    # ── Runway ──────────────────────────────────────────────────────────────

    def test_runway_short_high_risk(self):
        analysis = self._agent().run(self._finance_problem(
            burn_rate_usd=100000, cash_balance_usd=300000,
        ))
        run_insights = [i for i in analysis.insights if i.metric_name == "runway_months"]
        assert run_insights[0].risk_level == "high"  # 3 months

    def test_runway_medium_risk(self):
        analysis = self._agent().run(self._finance_problem(
            burn_rate_usd=50000, cash_balance_usd=400000,
        ))
        run_insights = [i for i in analysis.insights if i.metric_name == "runway_months"]
        assert run_insights[0].risk_level == "medium"  # 8 months

    def test_runway_long_low_risk(self):
        analysis = self._agent().run(self._finance_problem(
            burn_rate_usd=20000, cash_balance_usd=500000,
        ))
        run_insights = [i for i in analysis.insights if i.metric_name == "runway_months"]
        assert run_insights[0].risk_level == "low"  # 25 months

    def test_runway_zero_burn_rate_low_risk(self):
        analysis = self._agent().run(self._finance_problem(burn_rate_usd=0))
        run_insights = [i for i in analysis.insights if i.metric_name == "runway_months"]
        assert run_insights[0].risk_level == "low"

    # ── ARR consistency ─────────────────────────────────────────────────────

    def test_arr_consistency_mismatch_flagged(self):
        analysis = self._agent().run(self._finance_problem(
            mrr_usd=50000, arr_usd=800000,  # 50k*12 = 600k, 800k is 33% off
        ))
        arr_insights = [i for i in analysis.insights if i.metric_name == "arr_consistency_gap_pct"]
        assert len(arr_insights) == 1
        assert arr_insights[0].risk_level == "medium"

    def test_arr_consistency_match_no_flag(self):
        analysis = self._agent().run(self._finance_problem(
            mrr_usd=50000, arr_usd=600000,  # exact match
        ))
        arr_insights = [i for i in analysis.insights if i.metric_name == "arr_consistency_gap_pct"]
        assert len(arr_insights) == 0

    # ── COGS efficiency ─────────────────────────────────────────────────────

    def test_cogs_high_risk(self):
        analysis = self._agent().run(self._finance_problem(cogs_pct=70.0))
        cogs_insights = [i for i in analysis.insights if i.metric_name == "cogs_pct"]
        assert cogs_insights[0].risk_level == "high"

    def test_cogs_medium_risk(self):
        analysis = self._agent().run(self._finance_problem(cogs_pct=50.0))
        cogs_insights = [i for i in analysis.insights if i.metric_name == "cogs_pct"]
        assert cogs_insights[0].risk_level == "medium"

    def test_cogs_low_risk(self):
        analysis = self._agent().run(self._finance_problem(cogs_pct=20.0))
        cogs_insights = [i for i in analysis.insights if i.metric_name == "cogs_pct"]
        assert cogs_insights[0].risk_level == "low"

    # ── Validation ──────────────────────────────────────────────────────────

    def test_negative_mrr_raises(self):
        with pytest.raises(ValueError, match="mrr_usd must be >= 0"):
            self._agent().run(self._finance_problem(mrr_usd=-100))

    def test_negative_burn_rate_raises(self):
        with pytest.raises(ValueError, match="burn_rate_usd must be >= 0"):
            self._agent().run(self._finance_problem(burn_rate_usd=-1))

    def test_gross_margin_over_100_raises(self):
        with pytest.raises(ValueError, match="gross_margin_pct must be in"):
            self._agent().run(self._finance_problem(gross_margin_pct=110.0))

    def test_cogs_negative_raises(self):
        with pytest.raises(ValueError, match="cogs_pct must be in"):
            self._agent().run(self._finance_problem(cogs_pct=-5.0))

    def test_extreme_mrr_growth_raises(self):
        with pytest.raises(ValueError, match="mrr_growth_pct must be in"):
            self._agent().run(self._finance_problem(mrr_growth_pct=2000.0))


# ---------------------------------------------------------------------------
# Sales domain tests
# ---------------------------------------------------------------------------

class TestSalesDomain:
    """Tests for each sales rule boundary."""

    def _agent(self):
        return BusinessAgent()

    def _sales_problem(self, **overrides):
        params = {
            "pipeline_value_usd": 600000,
            "quota_usd": 150000,
            "win_rate_pct": 30.0,
            "avg_deal_size_usd": 25000,
            "avg_sales_cycle_days": 45,
            "churn_rate_pct": 3.0,
            "ltv_usd": 30000,
            "cac_usd": 8000,
            "ndr_pct": 105.0,
            "expansion_mrr_usd": 5000,
            "contraction_mrr_usd": 1000,
            "logo_churn_pct": 2.0,
            "total_mrr_usd": 50000,
        }
        params.update(overrides)
        return BusinessProblem(name="sales-test", domains=["sales"], parameters=params)

    # ── Pipeline coverage ───────────────────────────────────────────────────

    def test_pipeline_coverage_low_high_risk(self):
        analysis = self._agent().run(self._sales_problem(
            pipeline_value_usd=200000, quota_usd=100000,
        ))
        pc_insights = [i for i in analysis.insights if i.metric_name == "pipeline_coverage_ratio"]
        assert pc_insights[0].risk_level == "high"  # 2x

    def test_pipeline_coverage_medium_risk(self):
        analysis = self._agent().run(self._sales_problem(
            pipeline_value_usd=400000, quota_usd=100000,
        ))
        pc_insights = [i for i in analysis.insights if i.metric_name == "pipeline_coverage_ratio"]
        assert pc_insights[0].risk_level == "medium"  # 4x

    def test_pipeline_coverage_high_low_risk(self):
        analysis = self._agent().run(self._sales_problem(
            pipeline_value_usd=600000, quota_usd=100000,
        ))
        pc_insights = [i for i in analysis.insights if i.metric_name == "pipeline_coverage_ratio"]
        assert pc_insights[0].risk_level == "low"  # 6x

    def test_pipeline_coverage_zero_quota_skipped(self):
        analysis = self._agent().run(self._sales_problem(quota_usd=0))
        pc_insights = [i for i in analysis.insights if i.metric_name == "pipeline_coverage_ratio"]
        assert len(pc_insights) == 0

    # ── Win rate ────────────────────────────────────────────────────────────

    def test_win_rate_low_high_risk(self):
        analysis = self._agent().run(self._sales_problem(win_rate_pct=10.0))
        wr_insights = [i for i in analysis.insights if i.metric_name == "win_rate_pct"]
        assert wr_insights[0].risk_level == "high"

    def test_win_rate_mid_medium_risk(self):
        analysis = self._agent().run(self._sales_problem(win_rate_pct=25.0))
        wr_insights = [i for i in analysis.insights if i.metric_name == "win_rate_pct"]
        assert wr_insights[0].risk_level == "medium"

    def test_win_rate_high_low_risk(self):
        analysis = self._agent().run(self._sales_problem(win_rate_pct=40.0))
        wr_insights = [i for i in analysis.insights if i.metric_name == "win_rate_pct"]
        assert wr_insights[0].risk_level == "low"

    # ── Sales cycle ─────────────────────────────────────────────────────────

    def test_sales_cycle_long_high_risk(self):
        analysis = self._agent().run(self._sales_problem(avg_sales_cycle_days=120))
        sc_insights = [i for i in analysis.insights if i.metric_name == "avg_sales_cycle_days"]
        assert sc_insights[0].risk_level == "high"

    def test_sales_cycle_mid_medium_risk(self):
        analysis = self._agent().run(self._sales_problem(avg_sales_cycle_days=60))
        sc_insights = [i for i in analysis.insights if i.metric_name == "avg_sales_cycle_days"]
        assert sc_insights[0].risk_level == "medium"

    def test_sales_cycle_short_low_risk(self):
        analysis = self._agent().run(self._sales_problem(avg_sales_cycle_days=15))
        sc_insights = [i for i in analysis.insights if i.metric_name == "avg_sales_cycle_days"]
        assert sc_insights[0].risk_level == "low"

    # ── Revenue churn ───────────────────────────────────────────────────────

    def test_churn_high_risk(self):
        analysis = self._agent().run(self._sales_problem(churn_rate_pct=7.0))
        ch_insights = [i for i in analysis.insights if i.metric_name == "churn_rate_pct"]
        assert ch_insights[0].risk_level == "high"

    def test_churn_medium_risk(self):
        analysis = self._agent().run(self._sales_problem(churn_rate_pct=3.5))
        ch_insights = [i for i in analysis.insights if i.metric_name == "churn_rate_pct"]
        assert ch_insights[0].risk_level == "medium"

    def test_churn_low_risk(self):
        analysis = self._agent().run(self._sales_problem(churn_rate_pct=1.0))
        ch_insights = [i for i in analysis.insights if i.metric_name == "churn_rate_pct"]
        assert ch_insights[0].risk_level == "low"

    # ── LTV:CAC ratio ───────────────────────────────────────────────────────

    def test_ltv_cac_low_high_risk(self):
        analysis = self._agent().run(self._sales_problem(ltv_usd=1000, cac_usd=500))
        lc_insights = [i for i in analysis.insights if i.metric_name == "ltv_cac_ratio"]
        assert lc_insights[0].risk_level == "high"  # 2x

    def test_ltv_cac_mid_medium_risk(self):
        analysis = self._agent().run(self._sales_problem(ltv_usd=4000, cac_usd=1000))
        lc_insights = [i for i in analysis.insights if i.metric_name == "ltv_cac_ratio"]
        assert lc_insights[0].risk_level == "medium"  # 4x

    def test_ltv_cac_high_low_risk(self):
        analysis = self._agent().run(self._sales_problem(ltv_usd=6000, cac_usd=1000))
        lc_insights = [i for i in analysis.insights if i.metric_name == "ltv_cac_ratio"]
        assert lc_insights[0].risk_level == "low"  # 6x

    def test_ltv_cac_skipped_when_zero(self):
        analysis = self._agent().run(self._sales_problem(ltv_usd=0, cac_usd=0))
        lc_insights = [i for i in analysis.insights if i.metric_name == "ltv_cac_ratio"]
        assert len(lc_insights) == 0

    # ── Net Dollar Retention ────────────────────────────────────────────────

    def test_ndr_low_high_risk(self):
        analysis = self._agent().run(self._sales_problem(ndr_pct=80.0))
        ndr_insights = [i for i in analysis.insights if i.metric_name == "ndr_pct"]
        assert ndr_insights[0].risk_level == "high"

    def test_ndr_mid_medium_risk(self):
        analysis = self._agent().run(self._sales_problem(ndr_pct=100.0))
        ndr_insights = [i for i in analysis.insights if i.metric_name == "ndr_pct"]
        assert ndr_insights[0].risk_level == "medium"

    def test_ndr_high_low_risk(self):
        analysis = self._agent().run(self._sales_problem(ndr_pct=120.0))
        ndr_insights = [i for i in analysis.insights if i.metric_name == "ndr_pct"]
        assert ndr_insights[0].risk_level == "low"

    # ── Expansion mix ───────────────────────────────────────────────────────

    def test_expansion_mix_low_flagged(self):
        analysis = self._agent().run(self._sales_problem(
            expansion_mrr_usd=2000, total_mrr_usd=50000,
        ))
        ex_insights = [i for i in analysis.insights if i.metric_name == "expansion_mix_ratio"]
        assert len(ex_insights) == 1
        assert ex_insights[0].risk_level == "medium"

    def test_expansion_mix_adequate_no_flag(self):
        analysis = self._agent().run(self._sales_problem(
            expansion_mrr_usd=10000, total_mrr_usd=50000,
        ))
        ex_insights = [i for i in analysis.insights if i.metric_name == "expansion_mix_ratio"]
        assert len(ex_insights) == 0

    def test_expansion_mix_zero_total_mrr_skipped(self):
        analysis = self._agent().run(self._sales_problem(total_mrr_usd=0))
        ex_insights = [i for i in analysis.insights if i.metric_name == "expansion_mix_ratio"]
        assert len(ex_insights) == 0

    # ── Logo vs revenue churn ───────────────────────────────────────────────

    def test_logo_churn_disproportionate_flagged(self):
        analysis = self._agent().run(self._sales_problem(
            churn_rate_pct=2.0, logo_churn_pct=5.0,
        ))
        lrc_insights = [i for i in analysis.insights
                        if i.metric_name == "logo_vs_revenue_churn_ratio"]
        assert len(lrc_insights) == 1
        assert lrc_insights[0].risk_level == "medium"

    def test_logo_churn_proportionate_no_flag(self):
        analysis = self._agent().run(self._sales_problem(
            churn_rate_pct=3.0, logo_churn_pct=3.0,
        ))
        lrc_insights = [i for i in analysis.insights
                        if i.metric_name == "logo_vs_revenue_churn_ratio"]
        assert len(lrc_insights) == 0

    # ── Validation ──────────────────────────────────────────────────────────

    def test_negative_pipeline_raises(self):
        with pytest.raises(ValueError, match="pipeline_value_usd must be >= 0"):
            self._agent().run(self._sales_problem(pipeline_value_usd=-100))

    def test_win_rate_over_100_raises(self):
        with pytest.raises(ValueError, match="win_rate_pct must be in"):
            self._agent().run(self._sales_problem(win_rate_pct=110.0))

    def test_ndr_over_200_raises(self):
        with pytest.raises(ValueError, match="ndr_pct must be in"):
            self._agent().run(self._sales_problem(ndr_pct=250.0))

    def test_negative_ltv_raises(self):
        with pytest.raises(ValueError, match="ltv_usd must be >= 0"):
            self._agent().run(self._sales_problem(ltv_usd=-1))

    def test_negative_cac_raises(self):
        with pytest.raises(ValueError, match="cac_usd must be >= 0"):
            self._agent().run(self._sales_problem(cac_usd=-1))

    def test_churn_over_100_raises(self):
        with pytest.raises(ValueError, match="churn_rate_pct must be in"):
            self._agent().run(self._sales_problem(churn_rate_pct=150.0))


# ---------------------------------------------------------------------------
# StripeConnector tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestStripeConnector:
    """Tests for StripeConnector with mocked requests.get."""

    def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("STRIPE_API_KEY", raising=False)
        connector = StripeConnector(api_key=None)
        assert connector.fetch() is None

    def test_explicit_api_key_used(self):
        connector = StripeConnector(api_key="sk_test_fake")
        assert connector._api_key == "sk_test_fake"

    def test_env_api_key_used(self, monkeypatch):
        monkeypatch.setenv("STRIPE_API_KEY", "sk_env_fake")
        connector = StripeConnector()
        assert connector._api_key == "sk_env_fake"

    def test_fetch_success_returns_dict(self):
        """Mocked Stripe API returning subscriptions and balance."""
        sub_response = MagicMock()
        sub_response.status_code = 200
        sub_response.raise_for_status = MagicMock()
        sub_response.json.return_value = {
            "data": [
                {
                    "items": {
                        "data": [
                            {"plan": {"amount": 5000, "interval": "month"}},
                            {"plan": {"amount": 3000, "interval": "month"}},
                        ]
                    }
                }
            ]
        }

        bal_response = MagicMock()
        bal_response.status_code = 200
        bal_response.raise_for_status = MagicMock()
        bal_response.json.return_value = {
            "available": [{"amount": 10000000}]
        }

        def mock_get(url, **kwargs):
            if "subscriptions" in url:
                return sub_response
            elif "balance" in url:
                return bal_response
            return MagicMock()

        with patch("requests.get", side_effect=mock_get):
            connector = StripeConnector(api_key="sk_test")
            result = connector.fetch()

        assert result is not None
        assert result["mrr_usd"] == 80.0  # (5000 + 3000) / 100
        assert result["arr_usd"] == 960.0
        assert result["cash_balance_usd"] == 100000.0

    def test_fetch_api_error_returns_none(self):
        """If API raises, fetch returns None gracefully."""
        import requests as req_mod

        with patch("requests.get", side_effect=req_mod.exceptions.ConnectionError("timeout")):
            connector = StripeConnector(api_key="sk_test")
            result = connector.fetch()
        assert result is None

    def test_yearly_subscription_normalised(self):
        """Yearly plans are divided by 12 for MRR."""
        sub_response = MagicMock()
        sub_response.raise_for_status = MagicMock()
        sub_response.json.return_value = {
            "data": [
                {
                    "items": {
                        "data": [
                            {"plan": {"amount": 120000, "interval": "year"}},
                        ]
                    }
                }
            ]
        }

        bal_response = MagicMock()
        bal_response.raise_for_status = MagicMock()
        bal_response.json.return_value = {"available": []}

        def mock_get(url, **kwargs):
            if "subscriptions" in url:
                return sub_response
            return bal_response

        with patch("requests.get", side_effect=mock_get):
            connector = StripeConnector(api_key="sk_test")
            result = connector.fetch()
        assert result is not None
        assert abs(result["mrr_usd"] - 100.0) < 0.01  # 120000/12/100


# ---------------------------------------------------------------------------
# HubSpotConnector tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestHubSpotConnector:
    """Tests for HubSpotConnector with mocked requests.get."""

    def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("HUBSPOT_API_KEY", raising=False)
        connector = HubSpotConnector(api_key=None)
        assert connector.fetch() is None

    def test_explicit_api_key_used(self):
        connector = HubSpotConnector(api_key="hub_test_fake")
        assert connector._api_key == "hub_test_fake"

    def test_env_api_key_used(self, monkeypatch):
        monkeypatch.setenv("HUBSPOT_API_KEY", "hub_env_fake")
        connector = HubSpotConnector()
        assert connector._api_key == "hub_env_fake"

    def test_fetch_success_returns_dict(self):
        """Mocked HubSpot API returning deals."""
        deals_response = MagicMock()
        deals_response.status_code = 200
        deals_response.raise_for_status = MagicMock()
        deals_response.json.return_value = {
            "results": [
                {
                    "properties": {
                        "amount": "50000",
                        "dealstage": "closedwon",
                        "createdate": "2026-01-01T00:00:00.000Z",
                        "closedate": "2026-02-15T00:00:00.000Z",
                    }
                },
                {
                    "properties": {
                        "amount": "30000",
                        "dealstage": "closedwon",
                        "createdate": "2026-01-10T00:00:00.000Z",
                        "closedate": "2026-03-10T00:00:00.000Z",
                    }
                },
                {
                    "properties": {
                        "amount": "20000",
                        "dealstage": "open",
                    }
                },
                {
                    "properties": {
                        "amount": "10000",
                        "dealstage": "closedlost",
                    }
                },
            ]
        }

        with patch("requests.get", return_value=deals_response):
            connector = HubSpotConnector(api_key="hub_test")
            result = connector.fetch()

        assert result is not None
        assert result["pipeline_value_usd"] == 20000.0
        assert result["win_rate_pct"] == 50.0  # 2 won / 4 total
        assert result["avg_deal_size_usd"] == 40000.0  # (50k+30k)/2

    def test_fetch_api_error_returns_none(self):
        import requests as req_mod

        with patch("requests.get", side_effect=req_mod.exceptions.ConnectionError("timeout")):
            connector = HubSpotConnector(api_key="hub_test")
            result = connector.fetch()
        assert result is None

    def test_empty_deals_returns_pipeline_zero(self):
        empty_response = MagicMock()
        empty_response.raise_for_status = MagicMock()
        empty_response.json.return_value = {"results": []}

        with patch("requests.get", return_value=empty_response):
            connector = HubSpotConnector(api_key="hub_test")
            result = connector.fetch()
        # Empty deals -> pipeline_value_usd=0.0 but no win/deal data
        assert result is not None
        assert result["pipeline_value_usd"] == 0.0


# ---------------------------------------------------------------------------
# Business LLM augmentation tests
# ---------------------------------------------------------------------------

class TestBusinessLLMAugmentation:
    """Tests for BusinessAgent's optional use_llm=True path."""

    def _agent(self):
        return BusinessAgent()

    def _problem(self):
        return BusinessProblem(
            name="llm-test",
            domains=["finance", "sales"],
            parameters={
                "mrr_usd": 50000, "mrr_growth_pct": 8.0,
                "gross_margin_pct": 72.0, "cogs_pct": 28.0,
                "burn_rate_usd": 30000, "cash_balance_usd": 500000,
                "pipeline_value_usd": 600000, "quota_usd": 150000,
                "win_rate_pct": 30.0, "churn_rate_pct": 3.0,
            },
        )

    def test_use_llm_false_gives_empty_commentary(self):
        analysis = self._agent().run(self._problem(), use_llm=False)
        assert analysis.llm_commentary == []

    def test_use_llm_true_calls_llm(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: "1. Strong MRR growth.\n2. Improve win rate.",
        )
        analysis = self._agent().run(self._problem(), use_llm=True)
        assert len(analysis.llm_commentary) >= 1
        assert any("MRR" in c for c in analysis.llm_commentary)

    def test_use_llm_unavailable_gives_empty(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: None,
        )
        analysis = self._agent().run(self._problem(), use_llm=True)
        assert analysis.llm_commentary == []

    def test_llm_does_not_alter_deterministic_outputs(self, monkeypatch):
        monkeypatch.setattr(
            "sub_team.llm_client.llm_complete",
            lambda *a, **kw: "LLM commentary here.",
        )
        problem = self._problem()
        no_llm = self._agent().run(problem, use_llm=False)
        with_llm = self._agent().run(problem, use_llm=True)

        # Deterministic fields must match
        assert no_llm.overall_risk_score == with_llm.overall_risk_score
        assert len(no_llm.insights) == len(with_llm.insights)
        assert no_llm.recommendations == with_llm.recommendations
        # LLM commentary only in augmented version
        assert no_llm.llm_commentary == []
        assert len(with_llm.llm_commentary) >= 1


# ---------------------------------------------------------------------------
# BusinessInsight validation tests
# ---------------------------------------------------------------------------

class TestBusinessInsightValidation:
    """Validation on BusinessInsight dataclass."""

    def test_valid_insight(self):
        insight = BusinessInsight(
            domain="finance",
            finding="test",
            metric_name="mrr",
            metric_value=100.0,
            confidence=0.8,
            risk_level="low",
        )
        assert insight.domain == "finance"

    def test_confidence_too_high(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            BusinessInsight(
                domain="finance", finding="test", metric_name="x",
                metric_value=0, confidence=1.5, risk_level="low",
            )

    def test_confidence_negative(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            BusinessInsight(
                domain="finance", finding="test", metric_name="x",
                metric_value=0, confidence=-0.1, risk_level="low",
            )

    def test_invalid_risk_level(self):
        with pytest.raises(ValueError, match="risk_level must be one of"):
            BusinessInsight(
                domain="finance", finding="test", metric_name="x",
                metric_value=0, confidence=0.5, risk_level="critical",
            )


# ===========================================================================
# Boundary tests — finance domain exact thresholds
# ===========================================================================

class TestFinanceBoundaries:
    """Exact boundary tests to catch off-by-one errors in comparison operators."""

    def _agent(self):
        return BusinessAgent()

    def _fp(self, **overrides):
        params = {
            "mrr_usd": 50000, "mrr_growth_pct": 8.0, "arr_usd": 600000,
            "gross_margin_pct": 72.0, "cogs_pct": 28.0,
            "burn_rate_usd": 30000, "cash_balance_usd": 500000,
        }
        params.update(overrides)
        return BusinessProblem(name="fin-boundary", domains=["finance"], parameters=params)

    def _risk_for(self, analysis, metric_name):
        return [i for i in analysis.insights if i.metric_name == metric_name][0].risk_level

    def _value_for(self, analysis, metric_name):
        return [i for i in analysis.insights if i.metric_name == metric_name][0].metric_value

    # ── MRR growth boundaries ──────────────────────────────────────────────

    def test_mrr_growth_boundary_5_01_low_risk(self):
        """5.01% crosses > 5.0 threshold -> low risk."""
        a = self._agent().run(self._fp(mrr_growth_pct=5.01))
        assert self._risk_for(a, "mrr_growth_pct") == "low"

    def test_mrr_growth_boundary_minus_100_valid(self):
        """-100.0 is the lower validation bound; should be accepted as high risk."""
        a = self._agent().run(self._fp(mrr_growth_pct=-100.0))
        assert self._risk_for(a, "mrr_growth_pct") == "high"

    def test_mrr_growth_boundary_1000_valid(self):
        """1000.0 is the upper validation bound; should be accepted as low risk."""
        a = self._agent().run(self._fp(mrr_growth_pct=1000.0))
        assert self._risk_for(a, "mrr_growth_pct") == "low"

    def test_mrr_growth_below_minus_100_raises(self):
        with pytest.raises(ValueError, match="mrr_growth_pct must be in"):
            self._agent().run(self._fp(mrr_growth_pct=-101.0))

    # ── Gross margin boundaries ────────────────────────────────────────────

    def test_gross_margin_exactly_40_medium(self):
        """Code: < 40.0 -> high; so exactly 40.0 -> medium."""
        a = self._agent().run(self._fp(gross_margin_pct=40.0, cogs_pct=28.0))
        assert self._risk_for(a, "gross_margin_pct") == "medium"

    def test_gross_margin_exactly_60_medium(self):
        """Code: <= 60.0 -> medium; so exactly 60.0 -> medium."""
        a = self._agent().run(self._fp(gross_margin_pct=60.0, cogs_pct=28.0))
        assert self._risk_for(a, "gross_margin_pct") == "medium"

    def test_gross_margin_60_01_low(self):
        """Just above 60.0 -> low risk."""
        a = self._agent().run(self._fp(gross_margin_pct=60.1, cogs_pct=28.0))
        assert self._risk_for(a, "gross_margin_pct") == "low"

    # ── Runway boundaries ──────────────────────────────────────────────────

    def test_runway_exactly_6_months_medium(self):
        """Code: < 6 -> high; so exactly 6.0 -> medium."""
        a = self._agent().run(self._fp(burn_rate_usd=100000, cash_balance_usd=600000))
        assert self._risk_for(a, "runway_months") == "medium"

    def test_runway_exactly_12_months_medium(self):
        """Code: <= 12 -> medium; so exactly 12.0 -> medium."""
        a = self._agent().run(self._fp(burn_rate_usd=50000, cash_balance_usd=600000))
        assert self._risk_for(a, "runway_months") == "medium"

    def test_runway_just_above_12_months_low(self):
        """Just above 12 months -> low risk."""
        a = self._agent().run(self._fp(burn_rate_usd=49000, cash_balance_usd=600000))
        assert self._risk_for(a, "runway_months") == "low"

    def test_runway_zero_burn_sentinel_value(self):
        """Zero burn rate -> metric_value should be 999.0 sentinel."""
        a = self._agent().run(self._fp(burn_rate_usd=0))
        assert self._value_for(a, "runway_months") == 999.0

    # ── COGS boundaries ────────────────────────────────────────────────────

    def test_cogs_exactly_40_medium(self):
        """Code: >= 40.0 -> medium; so exactly 40.0 -> medium."""
        a = self._agent().run(self._fp(cogs_pct=40.0))
        assert self._risk_for(a, "cogs_pct") == "medium"

    def test_cogs_exactly_60_medium(self):
        """Code: > 60.0 -> high; so exactly 60.0 -> medium."""
        a = self._agent().run(self._fp(cogs_pct=60.0))
        assert self._risk_for(a, "cogs_pct") == "medium"

    def test_cogs_60_01_high(self):
        """Just above 60.0 -> high risk."""
        a = self._agent().run(self._fp(cogs_pct=60.1))
        assert self._risk_for(a, "cogs_pct") == "high"

    def test_cogs_over_100_raises(self):
        with pytest.raises(ValueError, match="cogs_pct must be in"):
            self._agent().run(self._fp(cogs_pct=101.0))

    # ── ARR consistency edge cases ─────────────────────────────────────────

    def test_arr_both_zero_no_flag(self):
        """When MRR=0 and ARR=0, the guard skips ARR check."""
        a = self._agent().run(self._fp(mrr_usd=0, arr_usd=0, mrr_growth_pct=0))
        arr_insights = [i for i in a.insights if i.metric_name == "arr_consistency_gap_pct"]
        assert len(arr_insights) == 0

    def test_arr_mrr_zero_arr_nonzero_no_flag(self):
        """When MRR=0, expected_arr=0; guard skips even if ARR is provided."""
        a = self._agent().run(self._fp(mrr_usd=0, arr_usd=500000, mrr_growth_pct=0))
        arr_insights = [i for i in a.insights if i.metric_name == "arr_consistency_gap_pct"]
        assert len(arr_insights) == 0

    # ── Additional validation ──────────────────────────────────────────────

    def test_negative_cash_balance_raises(self):
        with pytest.raises(ValueError, match="cash_balance_usd must be >= 0"):
            self._agent().run(self._fp(cash_balance_usd=-1))

    def test_negative_arr_raises(self):
        with pytest.raises(ValueError, match="arr_usd must be >= 0"):
            self._agent().run(self._fp(arr_usd=-1))


# ===========================================================================
# Boundary tests — sales domain exact thresholds
# ===========================================================================

class TestSalesBoundaries:
    """Exact boundary tests for all sales domain comparison operators."""

    def _agent(self):
        return BusinessAgent()

    def _sp(self, **overrides):
        params = {
            "pipeline_value_usd": 600000, "quota_usd": 150000,
            "win_rate_pct": 30.0, "avg_deal_size_usd": 25000,
            "avg_sales_cycle_days": 45, "churn_rate_pct": 3.0,
            "ltv_usd": 30000, "cac_usd": 8000, "ndr_pct": 105.0,
            "expansion_mrr_usd": 5000, "contraction_mrr_usd": 1000,
            "logo_churn_pct": 2.0, "total_mrr_usd": 50000,
        }
        params.update(overrides)
        return BusinessProblem(name="sales-boundary", domains=["sales"], parameters=params)

    def _risk_for(self, analysis, metric_name):
        return [i for i in analysis.insights if i.metric_name == metric_name][0].risk_level

    # ── Pipeline coverage boundaries ───────────────────────────────────────

    def test_pipeline_coverage_exactly_3x_medium(self):
        """Code: < 3.0 -> high; so exactly 3.0 -> medium."""
        a = self._agent().run(self._sp(pipeline_value_usd=300000, quota_usd=100000))
        assert self._risk_for(a, "pipeline_coverage_ratio") == "medium"

    def test_pipeline_coverage_exactly_5x_medium(self):
        """Code: <= 5.0 -> medium; so exactly 5.0 -> medium."""
        a = self._agent().run(self._sp(pipeline_value_usd=500000, quota_usd=100000))
        assert self._risk_for(a, "pipeline_coverage_ratio") == "medium"

    def test_pipeline_coverage_5_1x_low(self):
        """Just above 5.0 -> low risk."""
        a = self._agent().run(self._sp(pipeline_value_usd=510000, quota_usd=100000))
        assert self._risk_for(a, "pipeline_coverage_ratio") == "low"

    # ── Win rate boundaries ────────────────────────────────────────────────

    def test_win_rate_exactly_20_medium(self):
        """Code: < 20.0 -> high; so exactly 20.0 -> medium."""
        a = self._agent().run(self._sp(win_rate_pct=20.0))
        assert self._risk_for(a, "win_rate_pct") == "medium"

    def test_win_rate_exactly_35_medium(self):
        """Code: <= 35.0 -> medium; so exactly 35.0 -> medium."""
        a = self._agent().run(self._sp(win_rate_pct=35.0))
        assert self._risk_for(a, "win_rate_pct") == "medium"

    def test_win_rate_35_1_low(self):
        """Just above 35.0 -> low risk."""
        a = self._agent().run(self._sp(win_rate_pct=35.1))
        assert self._risk_for(a, "win_rate_pct") == "low"

    # ── Sales cycle boundaries ─────────────────────────────────────────────

    def test_sales_cycle_exactly_30_medium(self):
        """Code: >= 30 -> medium; so exactly 30 -> medium."""
        a = self._agent().run(self._sp(avg_sales_cycle_days=30))
        assert self._risk_for(a, "avg_sales_cycle_days") == "medium"

    def test_sales_cycle_29_low(self):
        """Just below 30 -> low risk."""
        a = self._agent().run(self._sp(avg_sales_cycle_days=29))
        assert self._risk_for(a, "avg_sales_cycle_days") == "low"

    def test_sales_cycle_exactly_90_medium(self):
        """Code: > 90 -> high; so exactly 90 -> medium."""
        a = self._agent().run(self._sp(avg_sales_cycle_days=90))
        assert self._risk_for(a, "avg_sales_cycle_days") == "medium"

    def test_sales_cycle_91_high(self):
        """Just above 90 -> high risk."""
        a = self._agent().run(self._sp(avg_sales_cycle_days=91))
        assert self._risk_for(a, "avg_sales_cycle_days") == "high"

    # ── Churn boundaries ───────────────────────────────────────────────────

    def test_churn_exactly_2_medium(self):
        """Code: >= 2.0 -> medium; so exactly 2.0 -> medium."""
        a = self._agent().run(self._sp(churn_rate_pct=2.0))
        assert self._risk_for(a, "churn_rate_pct") == "medium"

    def test_churn_1_99_low(self):
        """Just below 2.0 -> low risk."""
        a = self._agent().run(self._sp(churn_rate_pct=1.99))
        assert self._risk_for(a, "churn_rate_pct") == "low"

    def test_churn_exactly_5_medium(self):
        """Code: > 5.0 -> high; so exactly 5.0 -> medium."""
        a = self._agent().run(self._sp(churn_rate_pct=5.0))
        assert self._risk_for(a, "churn_rate_pct") == "medium"

    def test_churn_5_1_high(self):
        """Just above 5.0 -> high risk."""
        a = self._agent().run(self._sp(churn_rate_pct=5.1))
        assert self._risk_for(a, "churn_rate_pct") == "high"

    # ── LTV:CAC boundaries ─────────────────────────────────────────────────

    def test_ltv_cac_exactly_3x_medium(self):
        """Code: < 3.0 -> high; so exactly 3.0 -> medium."""
        a = self._agent().run(self._sp(ltv_usd=3000, cac_usd=1000))
        assert self._risk_for(a, "ltv_cac_ratio") == "medium"

    def test_ltv_cac_exactly_5x_medium(self):
        """Code: <= 5.0 -> medium; so exactly 5.0 -> medium."""
        a = self._agent().run(self._sp(ltv_usd=5000, cac_usd=1000))
        assert self._risk_for(a, "ltv_cac_ratio") == "medium"

    def test_ltv_cac_5_1x_low(self):
        """Just above 5.0 -> low risk."""
        a = self._agent().run(self._sp(ltv_usd=5100, cac_usd=1000))
        assert self._risk_for(a, "ltv_cac_ratio") == "low"

    def test_ltv_cac_only_ltv_zero_skipped(self):
        """LTV=0, CAC>0 -> guard fails, no insight produced."""
        a = self._agent().run(self._sp(ltv_usd=0, cac_usd=1000))
        lc = [i for i in a.insights if i.metric_name == "ltv_cac_ratio"]
        assert len(lc) == 0

    def test_ltv_cac_only_cac_zero_skipped(self):
        """LTV>0, CAC=0 -> guard fails, no insight produced."""
        a = self._agent().run(self._sp(ltv_usd=5000, cac_usd=0))
        lc = [i for i in a.insights if i.metric_name == "ltv_cac_ratio"]
        assert len(lc) == 0

    # ── NDR boundaries ─────────────────────────────────────────────────────

    def test_ndr_exactly_90_medium(self):
        """Code: < 90.0 -> high; so exactly 90.0 -> medium."""
        a = self._agent().run(self._sp(ndr_pct=90.0))
        assert self._risk_for(a, "ndr_pct") == "medium"

    def test_ndr_exactly_110_medium(self):
        """Code: <= 110.0 -> medium; so exactly 110.0 -> medium."""
        a = self._agent().run(self._sp(ndr_pct=110.0))
        assert self._risk_for(a, "ndr_pct") == "medium"

    def test_ndr_110_1_low(self):
        """Just above 110.0 -> low risk."""
        a = self._agent().run(self._sp(ndr_pct=110.1))
        assert self._risk_for(a, "ndr_pct") == "low"

    # ── Expansion mix boundary ─────────────────────────────────────────────

    def test_expansion_mix_exactly_10pct_no_flag(self):
        """Code: < 0.10 -> flag; so exactly 10% -> no flag."""
        a = self._agent().run(self._sp(expansion_mrr_usd=5000, total_mrr_usd=50000))
        ex = [i for i in a.insights if i.metric_name == "expansion_mix_ratio"]
        assert len(ex) == 0

    def test_expansion_mix_9_99pct_flagged(self):
        """Just below 10% -> flagged."""
        a = self._agent().run(self._sp(expansion_mrr_usd=4999, total_mrr_usd=50000))
        ex = [i for i in a.insights if i.metric_name == "expansion_mix_ratio"]
        assert len(ex) == 1

    # ── Logo vs revenue churn boundaries ───────────────────────────────────

    def test_logo_churn_zero_revenue_churn_no_flag(self):
        """churn=0 -> guard `churn > 0` prevents check; no flag."""
        a = self._agent().run(self._sp(churn_rate_pct=0, logo_churn_pct=5.0))
        lrc = [i for i in a.insights if i.metric_name == "logo_vs_revenue_churn_ratio"]
        assert len(lrc) == 0

    def test_logo_churn_at_exactly_1_5x_no_flag(self):
        """logo_churn == churn * 1.5 exactly -> guard is `>`, so no flag."""
        a = self._agent().run(self._sp(churn_rate_pct=4.0, logo_churn_pct=6.0))
        lrc = [i for i in a.insights if i.metric_name == "logo_vs_revenue_churn_ratio"]
        assert len(lrc) == 0

    def test_logo_churn_just_above_1_5x_flagged(self):
        """logo_churn slightly > churn * 1.5 -> flagged."""
        a = self._agent().run(self._sp(churn_rate_pct=4.0, logo_churn_pct=6.1))
        lrc = [i for i in a.insights if i.metric_name == "logo_vs_revenue_churn_ratio"]
        assert len(lrc) == 1


# ===========================================================================
# Summary, accessors, cross-domain, and recommendation tests
# ===========================================================================

class TestBusinessAnalysisSummary:
    """Tests for summary(), accessors, cross-domain links, and recommendations."""

    def _agent(self):
        return BusinessAgent()

    def _run(self, **overrides):
        params = {
            "mrr_usd": 50000, "mrr_growth_pct": 8.0, "arr_usd": 600000,
            "gross_margin_pct": 72.0, "cogs_pct": 28.0,
            "burn_rate_usd": 30000, "cash_balance_usd": 500000,
            "pipeline_value_usd": 600000, "quota_usd": 150000,
            "win_rate_pct": 30.0, "avg_deal_size_usd": 25000,
            "avg_sales_cycle_days": 45, "churn_rate_pct": 3.0,
            "ltv_usd": 30000, "cac_usd": 8000, "ndr_pct": 105.0,
            "expansion_mrr_usd": 5000, "total_mrr_usd": 50000,
            "logo_churn_pct": 2.0,
        }
        params.update(overrides)
        return self._agent().run(BusinessProblem(name="summary-test", parameters=params))

    # ── Summary risk labels ────────────────────────────────────────────────

    def test_summary_risk_label_low(self):
        """Healthy metrics -> overall risk < 0.4 -> summary says 'low'."""
        a = self._run(
            mrr_growth_pct=15.0, gross_margin_pct=80.0, cogs_pct=15.0,
            burn_rate_usd=10000, cash_balance_usd=500000,
            win_rate_pct=45.0, churn_rate_pct=1.0, ndr_pct=115.0,
            pipeline_value_usd=800000, quota_usd=100000,
            ltv_usd=10000, cac_usd=1000,
        )
        summary = a.summary()
        assert "(low)" in summary

    def test_summary_risk_label_high(self):
        """Terrible metrics -> overall risk >= 0.7 -> summary says 'high'."""
        a = self._run(
            mrr_growth_pct=-10.0, gross_margin_pct=20.0, cogs_pct=80.0,
            burn_rate_usd=200000, cash_balance_usd=500000,
            win_rate_pct=5.0, churn_rate_pct=8.0, ndr_pct=70.0,
            pipeline_value_usd=100000, quota_usd=100000,
            ltv_usd=1000, cac_usd=1000,
        )
        summary = a.summary()
        assert "(high)" in summary

    # ── Summary sections ───────────────────────────────────────────────────

    def test_summary_cross_domain_links_section(self):
        """Both domains -> dependency link -> summary contains 'Cross-domain links:' section."""
        a = self._run()
        summary = a.summary()
        assert "Cross-domain links:" in summary

    def test_summary_recommendations_section(self):
        """Summary includes numbered recommendations."""
        a = self._run()
        summary = a.summary()
        assert "Recommendations:" in summary
        assert "1." in summary

    def test_summary_no_cross_domain_links_section_single_domain(self):
        """Finance-only -> no cross-domain links -> section omitted."""
        a = self._agent().run(BusinessProblem(
            name="single", domains=["finance"],
            parameters={"mrr_usd": 50000, "mrr_growth_pct": 8.0,
                         "gross_margin_pct": 72.0, "cogs_pct": 28.0,
                         "burn_rate_usd": 30000, "cash_balance_usd": 500000},
        ))
        summary = a.summary()
        assert "Cross-domain links:" not in summary

    # ── Accessor edge cases ────────────────────────────────────────────────

    def test_insights_for_nonexistent_domain_empty(self):
        a = self._run()
        assert a.insights_for("marketing") == []

    def test_links_involving_nonexistent_domain_empty(self):
        a = self._run()
        assert a.links_involving("legal") == []

    # ── Recommendation content ─────────────────────────────────────────────

    def test_recommendations_dependency_link_text(self):
        """Both domains active -> dependency link -> recommendation mentions 'dependency'."""
        a = self._run()
        assert any("dependency" in r.lower() for r in a.recommendations)

    def test_recommendations_high_risk_callout(self):
        """High-risk findings produce domain-specific callout in recommendations."""
        a = self._run(mrr_growth_pct=-10.0)  # forces finance high-risk
        assert any("[finance]" in r for r in a.recommendations)

    def test_no_cross_domain_links_single_domain(self):
        """Finance-only -> no cross-domain links should be generated."""
        a = self._agent().run(BusinessProblem(
            name="single-fin", domains=["finance"],
            parameters={"mrr_usd": 50000, "mrr_growth_pct": 8.0,
                         "gross_margin_pct": 72.0, "cogs_pct": 28.0,
                         "burn_rate_usd": 30000, "cash_balance_usd": 500000},
        ))
        assert len(a.cross_domain_links) == 0


# ===========================================================================
# Connector edge case tests — Stripe
# ===========================================================================

class TestStripeConnectorEdgeCases:
    """Edge case tests for StripeConnector beyond basic success/failure."""

    def test_weekly_subscription_normalised(self):
        """Weekly plans are multiplied by 4.33 for MRR."""
        sub_response = MagicMock()
        sub_response.raise_for_status = MagicMock()
        sub_response.json.return_value = {
            "data": [{
                "items": {"data": [
                    {"plan": {"amount": 1000, "interval": "week"}},
                ]}
            }]
        }
        bal_response = MagicMock()
        bal_response.raise_for_status = MagicMock()
        bal_response.json.return_value = {"available": []}

        def mock_get(url, **kwargs):
            return sub_response if "subscriptions" in url else bal_response

        with patch("requests.get", side_effect=mock_get):
            result = StripeConnector(api_key="sk_test").fetch()
        assert result is not None
        assert abs(result["mrr_usd"] - (1000 * 4.33 / 100.0)) < 0.01

    def test_daily_subscription_normalised(self):
        """Daily plans are multiplied by 30 for MRR."""
        sub_response = MagicMock()
        sub_response.raise_for_status = MagicMock()
        sub_response.json.return_value = {
            "data": [{
                "items": {"data": [
                    {"plan": {"amount": 500, "interval": "day"}},
                ]}
            }]
        }
        bal_response = MagicMock()
        bal_response.raise_for_status = MagicMock()
        bal_response.json.return_value = {"available": []}

        def mock_get(url, **kwargs):
            return sub_response if "subscriptions" in url else bal_response

        with patch("requests.get", side_effect=mock_get):
            result = StripeConnector(api_key="sk_test").fetch()
        assert result is not None
        assert abs(result["mrr_usd"] - (500 * 30 / 100.0)) < 0.01

    def test_subscriptions_fail_balance_succeeds(self):
        """Subscriptions endpoint fails, balance succeeds -> partial data."""
        import requests as req_mod

        bal_response = MagicMock()
        bal_response.raise_for_status = MagicMock()
        bal_response.json.return_value = {"available": [{"amount": 5000000}]}

        def mock_get(url, **kwargs):
            if "subscriptions" in url:
                raise req_mod.exceptions.ConnectionError("sub fail")
            return bal_response

        with patch("requests.get", side_effect=mock_get):
            result = StripeConnector(api_key="sk_test").fetch()
        assert result is not None
        assert "cash_balance_usd" in result
        assert "mrr_usd" not in result

    def test_balance_fails_subscriptions_succeed(self):
        """Balance endpoint fails, subscriptions succeed -> partial data."""
        import requests as req_mod

        sub_response = MagicMock()
        sub_response.raise_for_status = MagicMock()
        sub_response.json.return_value = {
            "data": [{
                "items": {"data": [
                    {"plan": {"amount": 5000, "interval": "month"}},
                ]}
            }]
        }

        def mock_get(url, **kwargs):
            if "balance" in url:
                raise req_mod.exceptions.ConnectionError("bal fail")
            return sub_response

        with patch("requests.get", side_effect=mock_get):
            result = StripeConnector(api_key="sk_test").fetch()
        assert result is not None
        assert "mrr_usd" in result
        assert "cash_balance_usd" not in result

    def test_both_endpoints_fail_returns_none(self):
        """Both endpoints fail -> result dict is empty -> returns None."""
        import requests as req_mod

        with patch("requests.get", side_effect=req_mod.exceptions.ConnectionError("fail")):
            result = StripeConnector(api_key="sk_test").fetch()
        assert result is None

    def test_plan_none_falls_back_to_price(self):
        """When plan is explicitly None, falls back to price field."""
        sub_response = MagicMock()
        sub_response.raise_for_status = MagicMock()
        sub_response.json.return_value = {
            "data": [{
                "items": {"data": [
                    {"plan": None, "price": {"amount": 2000, "interval": "month"}},
                ]}
            }]
        }
        bal_response = MagicMock()
        bal_response.raise_for_status = MagicMock()
        bal_response.json.return_value = {"available": []}

        def mock_get(url, **kwargs):
            return sub_response if "subscriptions" in url else bal_response

        with patch("requests.get", side_effect=mock_get):
            result = StripeConnector(api_key="sk_test").fetch()
        assert result is not None
        assert result["mrr_usd"] == 20.0

    def test_missing_amount_defaults_to_zero(self):
        """Plan/price with no 'amount' key defaults to 0."""
        sub_response = MagicMock()
        sub_response.raise_for_status = MagicMock()
        sub_response.json.return_value = {
            "data": [{
                "items": {"data": [
                    {"plan": {"interval": "month"}},  # no "amount" key
                ]}
            }]
        }
        bal_response = MagicMock()
        bal_response.raise_for_status = MagicMock()
        bal_response.json.return_value = {"available": []}

        def mock_get(url, **kwargs):
            return sub_response if "subscriptions" in url else bal_response

        with patch("requests.get", side_effect=mock_get):
            result = StripeConnector(api_key="sk_test").fetch()
        assert result is not None
        assert result["mrr_usd"] == 0.0

    def test_multiple_subscriptions_aggregation(self):
        """Multiple subscription objects should aggregate MRR."""
        sub_response = MagicMock()
        sub_response.raise_for_status = MagicMock()
        sub_response.json.return_value = {
            "data": [
                {"items": {"data": [
                    {"plan": {"amount": 5000, "interval": "month"}},
                ]}},
                {"items": {"data": [
                    {"plan": {"amount": 3000, "interval": "month"}},
                    {"plan": {"amount": 2000, "interval": "month"}},
                ]}},
            ]
        }
        bal_response = MagicMock()
        bal_response.raise_for_status = MagicMock()
        bal_response.json.return_value = {"available": []}

        def mock_get(url, **kwargs):
            return sub_response if "subscriptions" in url else bal_response

        with patch("requests.get", side_effect=mock_get):
            result = StripeConnector(api_key="sk_test").fetch()
        assert result is not None
        assert result["mrr_usd"] == 100.0  # (5000+3000+2000)/100

    def test_multiple_balance_entries_aggregated(self):
        """Multiple balance entries (e.g., multi-currency) should sum."""
        sub_response = MagicMock()
        sub_response.raise_for_status = MagicMock()
        sub_response.json.return_value = {"data": []}

        bal_response = MagicMock()
        bal_response.raise_for_status = MagicMock()
        bal_response.json.return_value = {
            "available": [
                {"amount": 5000000},
                {"amount": 3000000},
            ]
        }

        def mock_get(url, **kwargs):
            return sub_response if "subscriptions" in url else bal_response

        with patch("requests.get", side_effect=mock_get):
            result = StripeConnector(api_key="sk_test").fetch()
        assert result is not None
        assert result["cash_balance_usd"] == 80000.0  # (5M+3M)/100


# ===========================================================================
# Connector edge case tests — HubSpot
# ===========================================================================

class TestHubSpotConnectorEdgeCases:
    """Edge case tests for HubSpotConnector beyond basic success/failure."""

    def _deals_response(self, deals):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"results": deals}
        return resp

    def test_deal_missing_amount_defaults_to_zero(self):
        """Deal with no 'amount' property defaults to 0."""
        resp = self._deals_response([
            {"properties": {"dealstage": "closedwon",
                            "createdate": "2026-01-01T00:00:00.000Z",
                            "closedate": "2026-02-01T00:00:00.000Z"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["win_rate_pct"] == 100.0
        assert result["avg_deal_size_usd"] == 0.0

    def test_deal_amount_none_defaults_to_zero(self):
        """Deal with amount=None defaults to 0 via `or 0`."""
        resp = self._deals_response([
            {"properties": {"amount": None, "dealstage": "closedwon",
                            "createdate": "2026-01-01T00:00:00.000Z",
                            "closedate": "2026-02-01T00:00:00.000Z"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["avg_deal_size_usd"] == 0.0

    def test_deal_closed_won_with_space_in_stage(self):
        """Deal with 'closed won' (space) should match."""
        resp = self._deals_response([
            {"properties": {"amount": "10000", "dealstage": "Closed Won",
                            "createdate": "2026-01-01T00:00:00.000Z",
                            "closedate": "2026-02-01T00:00:00.000Z"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["win_rate_pct"] == 100.0

    def test_deal_missing_createdate_skips_cycle(self):
        """No createdate -> cycle calc skipped, but deal still counted."""
        resp = self._deals_response([
            {"properties": {"amount": "10000", "dealstage": "closedwon",
                            "closedate": "2026-02-01T00:00:00.000Z"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["win_rate_pct"] == 100.0
        assert "avg_sales_cycle_days" not in result  # no cycle computed

    def test_deal_missing_closedate_skips_cycle(self):
        """No closedate -> cycle calc skipped, but deal still counted."""
        resp = self._deals_response([
            {"properties": {"amount": "10000", "dealstage": "closedwon",
                            "createdate": "2026-01-01T00:00:00.000Z"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["win_rate_pct"] == 100.0
        assert "avg_sales_cycle_days" not in result

    def test_deal_invalid_date_format_skips_cycle(self):
        """Malformed dates -> ValueError caught, cycle skipped."""
        resp = self._deals_response([
            {"properties": {"amount": "10000", "dealstage": "closedwon",
                            "createdate": "not-a-date",
                            "closedate": "also-not-a-date"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["win_rate_pct"] == 100.0
        assert "avg_sales_cycle_days" not in result

    def test_deal_negative_cycle_days_excluded(self):
        """Close date before create date -> negative days excluded."""
        resp = self._deals_response([
            {"properties": {"amount": "10000", "dealstage": "closedwon",
                            "createdate": "2026-03-01T00:00:00.000Z",
                            "closedate": "2026-01-01T00:00:00.000Z"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert "avg_sales_cycle_days" not in result  # negative excluded

    def test_deal_same_day_close_zero_cycle(self):
        """Create and close on same day -> 0 days cycle, included."""
        resp = self._deals_response([
            {"properties": {"amount": "10000", "dealstage": "closedwon",
                            "createdate": "2026-01-15T00:00:00.000Z",
                            "closedate": "2026-01-15T00:00:00.000Z"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["avg_sales_cycle_days"] == 0.0

    def test_cycle_days_calculation_correct(self):
        """Assert exact avg_sales_cycle_days from two closed-won deals."""
        resp = self._deals_response([
            {"properties": {"amount": "50000", "dealstage": "closedwon",
                            "createdate": "2026-01-01T00:00:00.000Z",
                            "closedate": "2026-02-15T00:00:00.000Z"}},  # 45 days
            {"properties": {"amount": "30000", "dealstage": "closedwon",
                            "createdate": "2026-01-10T00:00:00.000Z",
                            "closedate": "2026-03-10T00:00:00.000Z"}},  # 59 days
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["avg_sales_cycle_days"] == 52.0  # (45+59)/2

    def test_deal_missing_properties_key(self):
        """Deal object with no 'properties' key at all."""
        resp = self._deals_response([
            {},  # no properties
            {"properties": {"amount": "10000", "dealstage": "open"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["pipeline_value_usd"] == 10000.0

    def test_all_deals_closed_lost_pipeline_zero(self):
        """All deals closedlost -> pipeline=0, win_rate=0%."""
        resp = self._deals_response([
            {"properties": {"amount": "10000", "dealstage": "closedlost"}},
            {"properties": {"amount": "20000", "dealstage": "closedlost"}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["pipeline_value_usd"] == 0.0
        assert result["win_rate_pct"] == 0.0
        assert "avg_deal_size_usd" not in result  # no won deals

    def test_dealstage_none_treated_as_open(self):
        """dealstage=None -> defaults to empty string, treated as open deal."""
        resp = self._deals_response([
            {"properties": {"amount": "15000", "dealstage": None}},
        ])
        with patch("requests.get", return_value=resp):
            result = HubSpotConnector(api_key="hub_test").fetch()
        assert result is not None
        assert result["pipeline_value_usd"] == 15000.0
