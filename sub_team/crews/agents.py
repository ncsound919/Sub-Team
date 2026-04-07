"""
Agent Definitions — The Sub-Team Agentic Workforce.

8 specialized agents, each with distinct roles, tools, and expertise.
Designed to collaborate on cross-disciplinary tasks through CrewAI's
delegation and handoff mechanisms.

Agent Roster:
  1. Research Analyst     — Web research, trend analysis, competitive intelligence
  2. Software Engineer    — Code generation, debugging, architecture, DevOps
  3. Data Scientist       — Statistical analysis, ML, data pipelines, visualization
  4. Business Strategist  — Financial modeling, market analysis, growth strategy
  5. Creative Director    — Content creation, copywriting, branding, UX writing
  6. Security Analyst     — Vulnerability assessment, compliance, threat modeling
  7. Systems Architect    — Infrastructure design, API design, system integration
  8. Hardware Engineer    — CPU/RTL design, FPGA, embedded systems (wraps legacy agents)
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Dict, List, Optional

from crewai import Agent, LLM

_log = logging.getLogger(__name__)


# ============================================================================
# Agent Roles
# ============================================================================


class AgentRole(str, Enum):
    """Enumeration of all agent roles in the Sub-Team workforce."""

    RESEARCH_ANALYST = "research_analyst"
    SOFTWARE_ENGINEER = "software_engineer"
    DATA_SCIENTIST = "data_scientist"
    BUSINESS_STRATEGIST = "business_strategist"
    CREATIVE_DIRECTOR = "creative_director"
    SECURITY_ANALYST = "security_analyst"
    SYSTEMS_ARCHITECT = "systems_architect"
    HARDWARE_ENGINEER = "hardware_engineer"


# ============================================================================
# LLM Configuration
# ============================================================================


def get_llm() -> LLM:
    """
    Build the LLM instance for agents.

    Priority:
      1. ANTHROPIC_API_KEY → claude-sonnet-4-5 via Anthropic
      2. OPENAI_API_KEY with standard base → gpt-4o via OpenAI
      3. OPENROUTER_API_KEY → openai/gpt-4o-mini via OpenRouter
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return LLM(
            model="anthropic/claude-sonnet-4-5-20250514",
            api_key=anthropic_key,
            temperature=0.2,
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    # If base URL points to OpenRouter, use it as such
    if openai_key and "openrouter" in base_url.lower():
        model = os.environ.get("SUB_TEAM_LLM_MODEL", "openai/gpt-4o-mini")
        return LLM(
            model=f"openrouter/{model}",
            api_key=openai_key,
            temperature=0.2,
        )

    # Standard OpenAI
    if openai_key:
        return LLM(
            model="openai/gpt-4o",
            api_key=openai_key,
            temperature=0.2,
        )

    # Fallback: will error at runtime if no key is set
    _log.warning("No LLM API key found. Agents will fail on LLM calls.")
    return LLM(
        model="openai/gpt-4o-mini",
        temperature=0.2,
    )


# ============================================================================
# Tool Sets (lazy-loaded to avoid import errors if deps missing)
# ============================================================================


def _research_tools() -> list:
    """Tools for the Research Analyst."""
    from sub_team.tools import (
        WebSearchTool,
        WebScraperTool,
        GitHubSearchTool,
        GitHubRepoInfoTool,
        FileReadTool,
        FileWriteTool,
    )

    return [
        WebSearchTool(),
        WebScraperTool(),
        GitHubSearchTool(),
        GitHubRepoInfoTool(),
        FileReadTool(),
        FileWriteTool(),
    ]


def _engineer_tools() -> list:
    """Tools for the Software Engineer."""
    from sub_team.tools import (
        CodeExecutorTool,
        FileReadTool,
        FileWriteTool,
        DirectoryListTool,
        ShellExecTool,
        GitHubSearchTool,
        WebSearchTool,
    )

    return [
        CodeExecutorTool(),
        FileReadTool(),
        FileWriteTool(),
        DirectoryListTool(),
        ShellExecTool(),
        GitHubSearchTool(),
        WebSearchTool(),
    ]


def _data_science_tools() -> list:
    """Tools for the Data Scientist."""
    from sub_team.tools import (
        CodeExecutorTool,
        DataAnalysisTool,
        FileReadTool,
        FileWriteTool,
        WebSearchTool,
    )

    return [
        CodeExecutorTool(),
        DataAnalysisTool(),
        FileReadTool(),
        FileWriteTool(),
        WebSearchTool(),
    ]


def _business_tools() -> list:
    """Tools for the Business Strategist."""
    from sub_team.tools import (
        WebSearchTool,
        WebScraperTool,
        DataAnalysisTool,
        FileReadTool,
        FileWriteTool,
    )

    return [
        WebSearchTool(),
        WebScraperTool(),
        DataAnalysisTool(),
        FileReadTool(),
        FileWriteTool(),
    ]


def _creative_tools() -> list:
    """Tools for the Creative Director."""
    from sub_team.tools import (
        WebSearchTool,
        WebScraperTool,
        FileReadTool,
        FileWriteTool,
    )

    return [
        WebSearchTool(),
        WebScraperTool(),
        FileReadTool(),
        FileWriteTool(),
    ]


def _security_tools() -> list:
    """Tools for the Security Analyst."""
    from sub_team.tools import (
        CodeExecutorTool,
        FileReadTool,
        DirectoryListTool,
        ShellExecTool,
        WebSearchTool,
        GitHubSearchTool,
    )

    return [
        CodeExecutorTool(),
        FileReadTool(),
        DirectoryListTool(),
        ShellExecTool(),
        WebSearchTool(),
        GitHubSearchTool(),
    ]


def _architect_tools() -> list:
    """Tools for the Systems Architect."""
    from sub_team.tools import (
        FileReadTool,
        FileWriteTool,
        DirectoryListTool,
        WebSearchTool,
        WebScraperTool,
        ShellExecTool,
        CodeExecutorTool,
        GitHubSearchTool,
    )

    return [
        FileReadTool(),
        FileWriteTool(),
        DirectoryListTool(),
        WebSearchTool(),
        WebScraperTool(),
        ShellExecTool(),
        CodeExecutorTool(),
        GitHubSearchTool(),
    ]


def _hardware_tools() -> list:
    """Tools for the Hardware Engineer."""
    from sub_team.tools import (
        CodeExecutorTool,
        FileReadTool,
        FileWriteTool,
        DirectoryListTool,
        ShellExecTool,
        WebSearchTool,
    )

    return [
        CodeExecutorTool(),
        FileReadTool(),
        FileWriteTool(),
        DirectoryListTool(),
        ShellExecTool(),
        WebSearchTool(),
    ]


# ============================================================================
# Agent Definitions
# ============================================================================

_AGENT_CONFIGS: Dict[AgentRole, dict] = {
    AgentRole.RESEARCH_ANALYST: {
        "role": "Senior Research Analyst",
        "goal": (
            "Conduct thorough research on any topic using web search, documentation "
            "scraping, and GitHub exploration. Synthesize findings into clear, "
            "actionable intelligence reports with cited sources."
        ),
        "backstory": (
            "You are a veteran research analyst with 15 years of experience across "
            "technology, finance, science, and market intelligence. You excel at "
            "finding the signal in the noise — quickly identifying the most relevant "
            "sources, cross-referencing claims, and producing reports that decision-makers "
            "actually read. You have deep expertise in evaluating open-source projects, "
            "analyzing technology trends, and conducting competitive analysis. You always "
            "cite your sources and flag uncertainty levels."
        ),
        "tools_fn": _research_tools,
        "allow_delegation": True,
    },
    AgentRole.SOFTWARE_ENGINEER: {
        "role": "Senior Software Engineer",
        "goal": (
            "Write, debug, refactor, and review production-quality code across "
            "multiple languages (Python, JavaScript/TypeScript, Rust, Go). "
            "Design clean architectures, implement features, fix bugs, write tests, "
            "and set up CI/CD pipelines."
        ),
        "backstory": (
            "You are a staff-level software engineer with 12 years of experience "
            "building production systems. You've shipped code at scale across web "
            "backends (FastAPI, Express, Next.js), data pipelines, CLI tools, and "
            "distributed systems. You write clean, well-tested code with clear error "
            "handling. You follow TDD, understand SOLID principles, and always consider "
            "edge cases. You can debug complex issues by reading stack traces, "
            "reproducing problems, and writing targeted fixes. You prefer simple "
            "solutions over clever ones."
        ),
        "tools_fn": _engineer_tools,
        "allow_delegation": True,
    },
    AgentRole.DATA_SCIENTIST: {
        "role": "Senior Data Scientist",
        "goal": (
            "Analyze datasets, build statistical models, create visualizations, "
            "and extract actionable insights from structured and unstructured data. "
            "Design data pipelines, evaluate ML models, and communicate findings "
            "to non-technical stakeholders."
        ),
        "backstory": (
            "You are a principal data scientist with 10 years of experience in "
            "applied ML, statistical modeling, and data engineering. You've built "
            "recommendation engines, fraud detection systems, demand forecasters, "
            "and NLP pipelines. You're proficient with pandas, scikit-learn, PyTorch, "
            "and SQL. You think critically about data quality, bias, and statistical "
            "significance. You create clear visualizations and explain complex models "
            "in plain language. You know when a simple regression beats a deep network."
        ),
        "tools_fn": _data_science_tools,
        "allow_delegation": True,
    },
    AgentRole.BUSINESS_STRATEGIST: {
        "role": "Business Strategy Director",
        "goal": (
            "Analyze market dynamics, build financial models, evaluate business "
            "opportunities, and create growth strategies. Conduct competitive analysis, "
            "size markets, project revenue, and advise on pricing, partnerships, "
            "and go-to-market approaches."
        ),
        "backstory": (
            "You are a seasoned business strategist with 15 years spanning management "
            "consulting (McKinsey), venture capital, and operating roles at growth-stage "
            "startups. You build rigorous financial models, conduct bottom-up market "
            "sizing, and create investor-ready materials. You understand unit economics, "
            "CAC/LTV dynamics, network effects, and platform strategy. You speak the "
            "language of both builders and investors. You ground every recommendation "
            "in data and clearly state your assumptions."
        ),
        "tools_fn": _business_tools,
        "allow_delegation": True,
    },
    AgentRole.CREATIVE_DIRECTOR: {
        "role": "Creative Director",
        "goal": (
            "Create compelling content across formats — blog posts, social media, "
            "product copy, email campaigns, pitch decks, documentation, and brand "
            "guidelines. Adapt voice and tone for different audiences and platforms."
        ),
        "backstory": (
            "You are an award-winning creative director with 12 years in brand "
            "strategy, content marketing, and UX writing. You've led creative for "
            "both B2B SaaS and consumer brands. You understand information architecture, "
            "user psychology, and the art of concise, impactful writing. You write "
            "headlines that convert, stories that resonate, and documentation that "
            "developers actually enjoy reading. You adapt seamlessly between formal "
            "business prose, casual social copy, and technical documentation."
        ),
        "tools_fn": _creative_tools,
        "allow_delegation": False,
    },
    AgentRole.SECURITY_ANALYST: {
        "role": "Senior Security Analyst",
        "goal": (
            "Conduct security assessments, code reviews for vulnerabilities, "
            "compliance audits, and threat modeling. Identify OWASP Top 10 issues, "
            "review authentication/authorization patterns, evaluate dependency "
            "security, and recommend hardening measures."
        ),
        "backstory": (
            "You are a senior security engineer with 10 years of experience in "
            "application security, penetration testing, and compliance (SOC 2, "
            "GDPR, HIPAA, PCI-DSS). You've done security reviews for Fortune 500 "
            "companies and fast-moving startups. You think like an attacker but "
            "communicate like a partner — providing clear, prioritized remediation "
            "guidance rather than just scare tactics. You know the difference between "
            "theoretical and practical risks and always consider the threat model."
        ),
        "tools_fn": _security_tools,
        "allow_delegation": False,
    },
    AgentRole.SYSTEMS_ARCHITECT: {
        "role": "Principal Systems Architect",
        "goal": (
            "Design system architectures, API contracts, database schemas, "
            "infrastructure layouts, and integration patterns. Evaluate technology "
            "choices, create architecture decision records, and ensure systems are "
            "scalable, reliable, and maintainable."
        ),
        "backstory": (
            "You are a principal architect with 18 years designing distributed "
            "systems, from monoliths to microservices to serverless. You've "
            "architected systems handling millions of requests per second and "
            "petabytes of data. You think in terms of trade-offs — CAP theorem, "
            "consistency vs availability, build vs buy. You create clear diagrams, "
            "write ADRs, and know that the best architecture is the simplest one "
            "that meets the requirements. You're fluent in cloud-native patterns "
            "(Kubernetes, event-driven, CQRS) but also know when a monolith is the "
            "right answer."
        ),
        "tools_fn": _architect_tools,
        "allow_delegation": True,
    },
    AgentRole.HARDWARE_ENGINEER: {
        "role": "Senior Hardware Engineer",
        "goal": (
            "Design CPU architectures, generate synthesizable RTL (Verilog/VHDL), "
            "perform formal verification, optimize microarchitectures, and analyze "
            "hardware-software co-design trade-offs. Leverage the Sub-Team deterministic "
            "pipeline for RISC-V CPU generation."
        ),
        "backstory": (
            "You are a senior hardware engineer with 12 years in CPU design, FPGA "
            "development, and ASIC verification. You've worked on RISC-V cores, ARM "
            "processors, and custom accelerators. You understand pipeline hazards, "
            "branch prediction, cache hierarchies, and formal verification techniques. "
            "You leverage the Sub-Team deterministic autocoding pipeline to generate "
            "RTL from ISA specifications, then apply your expertise to optimize and "
            "verify the output. You bridge the gap between hardware and software, "
            "understanding both Verilog and Python equally well."
        ),
        "tools_fn": _hardware_tools,
        "allow_delegation": True,
    },
}


# ============================================================================
# Agent Factory
# ============================================================================

# Cache tool instances per role to avoid recreating on every _build_agent call.
# Tools are stateless, so sharing instances across agents of the same role is safe.
_tool_cache: Dict[AgentRole, list] = {}


def _get_tools(role: AgentRole) -> list:
    """Return cached tool instances for a role, building on first access."""
    if role not in _tool_cache:
        cfg = _AGENT_CONFIGS[role]
        _tool_cache[role] = cfg["tools_fn"]()
    return _tool_cache[role]


def _build_agent(role: AgentRole) -> Agent:
    """Construct a CrewAI Agent from the config for the given role."""
    cfg = _AGENT_CONFIGS[role]
    llm = get_llm()

    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        tools=_get_tools(role),
        llm=llm,
        verbose=True,
        allow_delegation=cfg.get("allow_delegation", False),
        memory=True,
        max_iter=15,
        max_retry_limit=3,
    )


def get_agent_by_role(role: AgentRole) -> Agent:
    """Get a specific agent by its role enum."""
    return _build_agent(role)


def get_all_agents() -> Dict[AgentRole, Agent]:
    """Build and return all 8 agents."""
    return {role: _build_agent(role) for role in AgentRole}


def get_agents_for_roles(roles: List[AgentRole]) -> List[Agent]:
    """Build a subset of agents for specific roles."""
    return [_build_agent(role) for role in roles]
