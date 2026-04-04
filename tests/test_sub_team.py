"""
Tests for the sub-team agent pipeline.
"""

import pytest

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
)
from sub_team.cpu import gshare, bimodal
from sub_team.specification_agent import FormalSpec
from sub_team.microarchitecture_agent import MicroarchPlan
from sub_team.implementation_agent import RTLOutput
from sub_team.verification_agent import VerificationReport
from sub_team.cross_disciplinary_agent import DomainInsight, CrossDomainLink


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
            import os
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
    import tempfile
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
    import tempfile
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
        }
        params.update(extra_params)
        return DomainProblem(
            name="full-cross-domain",
            domains=list(SUPPORTED_DOMAINS),
            parameters=params,
        )

    # ── DomainProblem ──────────────────────────────────────────────────────

    def test_supported_domains_list(self):
        assert set(SUPPORTED_DOMAINS) == {"logistics", "biotech", "fintech", "probability"}

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
