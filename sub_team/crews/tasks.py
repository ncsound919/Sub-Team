"""
Task Definitions — Task templates for the Sub-Team workforce.

Provides structured task creation for common cross-disciplinary work
patterns. Each task type maps to the right agent(s) and includes
expected output formats.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from crewai import Agent, Task

_log = logging.getLogger(__name__)


# ============================================================================
# Task Types
# ============================================================================


class TaskType(str, Enum):
    """Common task types the workforce can handle."""

    # Research & Analysis
    RESEARCH = "research"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    TECHNOLOGY_EVALUATION = "technology_evaluation"
    MARKET_SIZING = "market_sizing"

    # Software Engineering
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    TESTING = "testing"

    # Data Science
    DATA_ANALYSIS = "data_analysis"
    MODEL_BUILDING = "model_building"
    VISUALIZATION = "visualization"

    # Business
    FINANCIAL_MODELING = "financial_modeling"
    BUSINESS_PLAN = "business_plan"
    PITCH_DECK = "pitch_deck"

    # Creative
    CONTENT_CREATION = "content_creation"
    COPYWRITING = "copywriting"
    DOCUMENTATION = "documentation"

    # Security
    SECURITY_AUDIT = "security_audit"
    THREAT_MODEL = "threat_model"
    COMPLIANCE_CHECK = "compliance_check"

    # Architecture
    SYSTEM_DESIGN = "system_design"
    API_DESIGN = "api_design"
    DATABASE_DESIGN = "database_design"

    # Hardware
    CPU_DESIGN = "cpu_design"
    RTL_GENERATION = "rtl_generation"
    HARDWARE_VERIFICATION = "hardware_verification"

    # Cross-disciplinary
    CROSS_DOMAIN = "cross_domain"
    CUSTOM = "custom"


# ============================================================================
# Task Templates
# ============================================================================

_TASK_TEMPLATES: Dict[TaskType, Dict[str, str]] = {
    TaskType.RESEARCH: {
        "description": (
            "Conduct thorough research on the following topic:\n\n"
            "{objective}\n\n"
            "Search the web, explore relevant GitHub repositories, and scrape "
            "key documentation pages. Cross-reference at least 3 sources. "
            "Identify key trends, major players, and potential risks."
        ),
        "expected_output": (
            "A structured research report with:\n"
            "1. Executive summary (3-5 sentences)\n"
            "2. Key findings (bulleted, with source citations)\n"
            "3. Trend analysis\n"
            "4. Recommendations\n"
            "5. Sources list with URLs"
        ),
    },
    TaskType.COMPETITIVE_ANALYSIS: {
        "description": (
            "Perform a competitive analysis for:\n\n"
            "{objective}\n\n"
            "Identify the top 5-10 competitors, analyze their strengths and "
            "weaknesses, pricing models, market positioning, and technology stacks. "
            "Use web search and GitHub exploration."
        ),
        "expected_output": (
            "A competitive analysis report with:\n"
            "1. Competitor landscape overview\n"
            "2. Feature comparison matrix\n"
            "3. Pricing comparison\n"
            "4. SWOT analysis for each major competitor\n"
            "5. Strategic recommendations"
        ),
    },
    TaskType.TECHNOLOGY_EVALUATION: {
        "description": (
            "Evaluate the following technology/framework/tool:\n\n"
            "{objective}\n\n"
            "Research its GitHub repo, documentation, community activity, "
            "performance benchmarks, and real-world adoption. Compare with "
            "alternatives."
        ),
        "expected_output": (
            "A technology evaluation report with:\n"
            "1. Overview and key features\n"
            "2. Pros and cons\n"
            "3. Comparison with alternatives\n"
            "4. Community health metrics (stars, contributors, issue response time)\n"
            "5. Adoption risk assessment\n"
            "6. Recommendation with confidence level"
        ),
    },
    TaskType.MARKET_SIZING: {
        "description": (
            "Estimate the market size for:\n\n"
            "{objective}\n\n"
            "Use both top-down and bottom-up approaches. Research industry reports, "
            "public filings, and analyst estimates. Identify TAM, SAM, and SOM with "
            "clear assumptions and data sources."
        ),
        "expected_output": (
            "A market sizing report with:\n"
            "1. TAM, SAM, SOM estimates with methodology\n"
            "2. Top-down analysis (industry reports, macro data)\n"
            "3. Bottom-up analysis (unit economics, adoption curves)\n"
            "4. Growth rate projections and drivers\n"
            "5. Key assumptions and sensitivity analysis\n"
            "6. Sources and data quality assessment"
        ),
    },
    TaskType.CODE_GENERATION: {
        "description": (
            "Implement the following feature/component:\n\n"
            "{objective}\n\n"
            "Write clean, production-quality code with proper error handling, "
            "type hints, and documentation. Follow the project's existing patterns "
            "and conventions. Include unit tests."
        ),
        "expected_output": (
            "Implementation including:\n"
            "1. Source code files with full implementation\n"
            "2. Unit tests with good coverage\n"
            "3. Brief documentation/docstrings\n"
            "4. Any necessary configuration changes"
        ),
    },
    TaskType.CODE_REVIEW: {
        "description": (
            "Review the following code for quality, security, and correctness:\n\n"
            "{objective}\n\n"
            "Check for bugs, security vulnerabilities, performance issues, "
            "code style violations, and architectural concerns."
        ),
        "expected_output": (
            "A code review report with:\n"
            "1. Summary assessment (approve/request changes)\n"
            "2. Critical issues (bugs, security, data loss risks)\n"
            "3. Performance concerns\n"
            "4. Style and maintainability suggestions\n"
            "5. Positive highlights"
        ),
    },
    TaskType.BUG_FIX: {
        "description": (
            "Diagnose and fix the following bug:\n\n"
            "{objective}\n\n"
            "Read the relevant source code, reproduce the issue, identify the "
            "root cause, implement a fix, and verify it doesn't introduce "
            "regressions."
        ),
        "expected_output": (
            "Bug fix deliverable:\n"
            "1. Root cause analysis\n"
            "2. Fix implementation (code changes)\n"
            "3. Test case that reproduces the bug\n"
            "4. Verification that the fix works"
        ),
    },
    TaskType.REFACTORING: {
        "description": (
            "Refactor the following code/module:\n\n"
            "{objective}\n\n"
            "Improve code structure, readability, and maintainability without "
            "changing external behavior. Apply SOLID principles, extract "
            "abstractions where appropriate, and eliminate duplication."
        ),
        "expected_output": (
            "Refactoring deliverable:\n"
            "1. Summary of changes and rationale\n"
            "2. Refactored code files\n"
            "3. Before/after comparison of key metrics (complexity, duplication)\n"
            "4. Updated or added tests confirming behavior preservation\n"
            "5. Migration notes if API changed"
        ),
    },
    TaskType.TESTING: {
        "description": (
            "Write tests for the following code/feature:\n\n"
            "{objective}\n\n"
            "Write comprehensive unit tests, integration tests, and edge case "
            "tests. Aim for high coverage of critical paths. Use appropriate "
            "mocking and fixtures."
        ),
        "expected_output": (
            "Test suite deliverable:\n"
            "1. Unit tests with descriptive names\n"
            "2. Integration tests for key workflows\n"
            "3. Edge case and error handling tests\n"
            "4. Coverage report summary\n"
            "5. Test fixtures and helper setup"
        ),
    },
    TaskType.DATA_ANALYSIS: {
        "description": (
            "Analyze the following data/dataset:\n\n"
            "{objective}\n\n"
            "Load the data, compute summary statistics, identify patterns and "
            "anomalies, and create relevant visualizations."
        ),
        "expected_output": (
            "Data analysis report with:\n"
            "1. Dataset overview (shape, types, quality)\n"
            "2. Key statistics and distributions\n"
            "3. Pattern analysis and correlations\n"
            "4. Anomaly detection results\n"
            "5. Actionable insights and recommendations"
        ),
    },
    TaskType.MODEL_BUILDING: {
        "description": (
            "Build a machine learning model for:\n\n"
            "{objective}\n\n"
            "Select appropriate algorithms, engineer features, train and evaluate "
            "models, and tune hyperparameters. Compare at least 2-3 approaches "
            "and justify your final model choice."
        ),
        "expected_output": (
            "ML model deliverable:\n"
            "1. Problem formulation and success metrics\n"
            "2. Feature engineering pipeline\n"
            "3. Model comparison (accuracy, precision, recall, F1)\n"
            "4. Hyperparameter tuning results\n"
            "5. Final model with evaluation on holdout set\n"
            "6. Deployment recommendations and monitoring plan"
        ),
    },
    TaskType.VISUALIZATION: {
        "description": (
            "Create data visualizations for:\n\n"
            "{objective}\n\n"
            "Design clear, informative charts and dashboards that communicate "
            "key insights to the target audience. Use appropriate chart types "
            "and follow data visualization best practices."
        ),
        "expected_output": (
            "Visualization deliverable:\n"
            "1. Chart/dashboard designs with annotations\n"
            "2. Code to reproduce visualizations\n"
            "3. Data preparation steps\n"
            "4. Narrative explanation of key insights shown\n"
            "5. Recommendations for interactive/dynamic versions"
        ),
    },
    TaskType.FINANCIAL_MODELING: {
        "description": (
            "Build a financial model for:\n\n"
            "{objective}\n\n"
            "Include revenue projections, cost structure, unit economics, "
            "and scenario analysis (base, bull, bear cases)."
        ),
        "expected_output": (
            "Financial model including:\n"
            "1. Revenue model with key assumptions\n"
            "2. Cost structure breakdown\n"
            "3. P&L projections (3-5 years)\n"
            "4. Unit economics (CAC, LTV, payback period)\n"
            "5. Scenario analysis with sensitivity tables\n"
            "6. Key metrics dashboard"
        ),
    },
    TaskType.BUSINESS_PLAN: {
        "description": (
            "Develop a business plan for:\n\n"
            "{objective}\n\n"
            "Cover the value proposition, target market, competitive landscape, "
            "go-to-market strategy, team requirements, financial projections, "
            "and key milestones."
        ),
        "expected_output": (
            "Business plan document with:\n"
            "1. Executive summary\n"
            "2. Problem/solution statement\n"
            "3. Market analysis and sizing\n"
            "4. Competitive positioning\n"
            "5. Go-to-market strategy\n"
            "6. Financial projections (3-year)\n"
            "7. Team and resource requirements\n"
            "8. Key milestones and timeline"
        ),
    },
    TaskType.PITCH_DECK: {
        "description": (
            "Create a pitch deck for:\n\n"
            "{objective}\n\n"
            "Structure the narrative for investors or stakeholders. Cover "
            "problem, solution, market, traction, team, financials, and ask. "
            "Keep it concise (10-15 slides)."
        ),
        "expected_output": (
            "Pitch deck deliverable:\n"
            "1. Slide-by-slide content and speaker notes\n"
            "2. Key data points and metrics per slide\n"
            "3. Narrative flow and story arc\n"
            "4. Competitive differentiation highlights\n"
            "5. Financial ask and use of funds\n"
            "6. Appendix slides for deep-dive questions"
        ),
    },
    TaskType.CONTENT_CREATION: {
        "description": (
            "Create content for the following brief:\n\n"
            "{objective}\n\n"
            "Research the topic, understand the target audience, and produce "
            "engaging content that achieves the stated goals."
        ),
        "expected_output": (
            "Content deliverable:\n"
            "1. Final content piece (formatted and polished)\n"
            "2. SEO keywords / hashtags (if applicable)\n"
            "3. Distribution suggestions\n"
            "4. Performance metrics to track"
        ),
    },
    TaskType.COPYWRITING: {
        "description": (
            "Write persuasive copy for:\n\n"
            "{objective}\n\n"
            "Craft compelling headlines, body copy, and calls-to-action. "
            "Match the brand voice and optimize for the target platform "
            "(web, email, ad, social)."
        ),
        "expected_output": (
            "Copywriting deliverable:\n"
            "1. Primary copy with headline variations (3+)\n"
            "2. Supporting body copy\n"
            "3. Call-to-action options\n"
            "4. A/B test suggestions\n"
            "5. Platform-specific adaptations"
        ),
    },
    TaskType.DOCUMENTATION: {
        "description": (
            "Write technical documentation for:\n\n"
            "{objective}\n\n"
            "Read the relevant source code, understand the architecture, and "
            "produce clear documentation for the target audience (developers, "
            "users, or operators)."
        ),
        "expected_output": (
            "Documentation deliverable:\n"
            "1. Overview and purpose\n"
            "2. Quick start / getting started guide\n"
            "3. API reference or usage guide\n"
            "4. Configuration and environment setup\n"
            "5. Troubleshooting and FAQ\n"
            "6. Architecture notes (if technical)"
        ),
    },
    TaskType.SECURITY_AUDIT: {
        "description": (
            "Conduct a security audit of:\n\n"
            "{objective}\n\n"
            "Review code for OWASP Top 10 vulnerabilities, check authentication "
            "patterns, evaluate dependency security, and assess the overall "
            "security posture."
        ),
        "expected_output": (
            "Security audit report with:\n"
            "1. Executive summary with risk rating\n"
            "2. Critical vulnerabilities (with severity)\n"
            "3. Dependency audit results\n"
            "4. Authentication/authorization review\n"
            "5. Remediation roadmap (prioritized)\n"
            "6. Compliance status (relevant frameworks)"
        ),
    },
    TaskType.THREAT_MODEL: {
        "description": (
            "Perform threat modeling for:\n\n"
            "{objective}\n\n"
            "Identify assets, threat actors, attack surfaces, and potential "
            "attack vectors using STRIDE or similar frameworks. Assess "
            "likelihood and impact for each threat."
        ),
        "expected_output": (
            "Threat model report with:\n"
            "1. Asset inventory and data flow diagram\n"
            "2. Trust boundaries identification\n"
            "3. Threat enumeration (STRIDE categories)\n"
            "4. Risk matrix (likelihood x impact)\n"
            "5. Prioritized mitigations\n"
            "6. Residual risk assessment"
        ),
    },
    TaskType.COMPLIANCE_CHECK: {
        "description": (
            "Assess compliance status for:\n\n"
            "{objective}\n\n"
            "Evaluate against relevant regulatory frameworks (SOC 2, GDPR, "
            "HIPAA, PCI-DSS, etc.). Identify gaps and provide remediation "
            "guidance."
        ),
        "expected_output": (
            "Compliance assessment with:\n"
            "1. Applicable frameworks and requirements\n"
            "2. Current compliance status per control area\n"
            "3. Gap analysis with severity ratings\n"
            "4. Remediation plan with effort estimates\n"
            "5. Evidence collection guidance\n"
            "6. Timeline to compliance"
        ),
    },
    TaskType.SYSTEM_DESIGN: {
        "description": (
            "Design the architecture for:\n\n"
            "{objective}\n\n"
            "Consider scalability, reliability, security, and cost. Evaluate "
            "technology choices and document trade-offs."
        ),
        "expected_output": (
            "Architecture document with:\n"
            "1. System context diagram\n"
            "2. Component architecture\n"
            "3. Data flow diagrams\n"
            "4. Technology stack with justification\n"
            "5. Scalability strategy\n"
            "6. ADR (Architecture Decision Records)"
        ),
    },
    TaskType.API_DESIGN: {
        "description": (
            "Design an API for:\n\n"
            "{objective}\n\n"
            "Define endpoints, request/response schemas, authentication, "
            "error handling, pagination, versioning, and rate limiting. "
            "Follow REST or GraphQL best practices as appropriate."
        ),
        "expected_output": (
            "API design document with:\n"
            "1. Resource model and endpoint listing\n"
            "2. Request/response schemas (with examples)\n"
            "3. Authentication and authorization design\n"
            "4. Error response format and codes\n"
            "5. Pagination and filtering strategy\n"
            "6. Rate limiting and versioning approach\n"
            "7. OpenAPI/Swagger specification (if REST)"
        ),
    },
    TaskType.DATABASE_DESIGN: {
        "description": (
            "Design a database schema for:\n\n"
            "{objective}\n\n"
            "Define tables/collections, relationships, indexes, constraints, "
            "and migration strategy. Consider query patterns, data volume, "
            "and performance requirements."
        ),
        "expected_output": (
            "Database design document with:\n"
            "1. Entity-relationship diagram\n"
            "2. Table definitions with column types and constraints\n"
            "3. Index strategy for key query patterns\n"
            "4. Migration scripts (up and down)\n"
            "5. Data volume estimates and partitioning strategy\n"
            "6. Backup and retention policy recommendations"
        ),
    },
    TaskType.CPU_DESIGN: {
        "description": (
            "Design a CPU with the following specifications:\n\n"
            "{objective}\n\n"
            "Use the Sub-Team deterministic pipeline to generate RTL from the "
            "ISA specification. Analyze the output for correctness and performance."
        ),
        "expected_output": (
            "CPU design deliverable:\n"
            "1. ISA specification document\n"
            "2. Microarchitecture plan\n"
            "3. Generated Verilog RTL files\n"
            "4. Verification report\n"
            "5. Performance analysis"
        ),
    },
    TaskType.RTL_GENERATION: {
        "description": (
            "Generate synthesizable RTL for:\n\n"
            "{objective}\n\n"
            "Produce clean Verilog/SystemVerilog code following coding standards. "
            "Include proper reset logic, clock domain handling, and parameterizable "
            "design where appropriate."
        ),
        "expected_output": (
            "RTL generation deliverable:\n"
            "1. Synthesizable Verilog/SystemVerilog source files\n"
            "2. Module interface documentation\n"
            "3. Testbench with stimulus and checkers\n"
            "4. Synthesis constraints notes\n"
            "5. Resource utilization estimates"
        ),
    },
    TaskType.HARDWARE_VERIFICATION: {
        "description": (
            "Verify the following hardware design:\n\n"
            "{objective}\n\n"
            "Create a verification plan, write testbenches, run simulations, "
            "and check functional correctness. Apply formal verification "
            "techniques where applicable."
        ),
        "expected_output": (
            "Verification deliverable:\n"
            "1. Verification plan with coverage goals\n"
            "2. Testbench code (directed + constrained random)\n"
            "3. Simulation results and waveform analysis\n"
            "4. Coverage report (functional + code)\n"
            "5. Bug list with severity ratings\n"
            "6. Sign-off recommendation"
        ),
    },
    TaskType.CROSS_DOMAIN: {
        "description": (
            "Analyze the following cross-disciplinary problem:\n\n"
            "{objective}\n\n"
            "This problem spans multiple domains. Identify the relevant "
            "disciplines, analyze the problem from each perspective, find "
            "cross-domain synergies, and produce an integrated recommendation."
        ),
        "expected_output": (
            "Cross-disciplinary analysis:\n"
            "1. Domain decomposition (which disciplines are involved)\n"
            "2. Per-domain analysis and insights\n"
            "3. Cross-domain synergies and conflicts\n"
            "4. Risk assessment (probability-grounded)\n"
            "5. Integrated recommendations\n"
            "6. Implementation roadmap"
        ),
    },
}


# ============================================================================
# Task Factory
# ============================================================================


def create_task(
    task_type: TaskType,
    objective: str,
    agent: Agent,
    *,
    context_tasks: Optional[List[Task]] = None,
    extra_instructions: str = "",
    output_file: Optional[str] = None,
) -> Task:
    """
    Create a CrewAI Task from a template.

    Parameters
    ----------
    task_type : TaskType
        The type of task to create.
    objective : str
        The specific objective/brief for this task instance.
    agent : Agent
        The agent assigned to execute this task.
    context_tasks : list[Task], optional
        Previous tasks whose output provides context.
    extra_instructions : str
        Additional instructions to append to the description.
    output_file : str, optional
        File path to write the output to.

    Returns
    -------
    Task
        A configured CrewAI Task ready for execution.
    """
    template = _TASK_TEMPLATES.get(task_type)

    if template:
        description = template["description"].format(objective=objective)
        expected_output = template["expected_output"]
    else:
        # Custom / unrecognized task type — use the objective directly
        description = objective
        expected_output = (
            "A clear, well-structured response addressing all aspects of the request."
        )

    if extra_instructions:
        description += f"\n\nAdditional instructions:\n{extra_instructions}"

    kwargs: Dict[str, Any] = {
        "description": description,
        "expected_output": expected_output,
        "agent": agent,
    }

    if context_tasks:
        kwargs["context"] = context_tasks
    if output_file:
        kwargs["output_file"] = output_file

    return Task(**kwargs)
