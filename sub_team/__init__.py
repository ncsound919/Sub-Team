"""
Sub-Team: Full-spectrum agentic workforce + deterministic CPU design pipeline.

Legacy Pipeline (4 deterministic agents):
  - SpecificationAgent       : Extracts formal constraints from ISA specs
  - MicroarchitectureAgent   : Synthesizes structural RTL skeletons
  - ImplementationAgent      : Generates synthesizable Verilog/VHDL via formal grammars
  - VerificationAgent        : Produces formal correctness proofs or counterexamples

Cross-disciplinary analysis agent:
  - CrossDisciplinaryAgent   : Rule-based analysis across logistics, biotech,
                               fintech, probability, and legal domains

Business intelligence agent:
  - BusinessAgent            : Rule-based analysis across finance and sales domains
                               with optional Stripe / HubSpot data connectors

Agentic Workforce (8 CrewAI-based agents):
  - Research Analyst         : Web research, competitive analysis, market sizing
  - Software Engineer        : Code generation, reviews, bug fixes, refactoring
  - Data Scientist           : Data analysis, ML model building, visualization
  - Business Strategist      : Financial modeling, business plans, pitch decks
  - Creative Director        : Content creation, copywriting, documentation
  - Security Analyst         : Security audits, threat modeling, compliance
  - Systems Architect        : System design, API design, database design
  - Hardware Engineer        : CPU design, RTL generation, hardware verification

Tooling:
  - WebSearchTool, WebScraperTool, CodeExecutorTool, FileReadTool, FileWriteTool
  - DirectoryListTool, DataAnalysisTool, GitHubSearchTool, GitHubRepoInfoTool
  - ShellExecTool

Memory:
  - AgentMemory (Mem0-backed with in-memory fallback)
  - Per-agent namespaces + shared team memory

Server:
  - FastAPI HTTP server for Draymond Orchestrator integration
  - Legacy subprocess mode preserved for backward compatibility

LLM augmentation (optional):
  - All four pipeline agents accept ``use_llm=True`` to obtain supplementary
    LLM-generated notes via OpenRouter (requires OPENAI_API_KEY or
    OPENROUTER_API_KEY in the environment).
  - llm_available()          : Returns True if an API key is configured.
"""

# ── Legacy pipeline agents ──────────────────────────────────────────────
from .specification_agent import SpecificationAgent
from .microarchitecture_agent import MicroarchitectureAgent
from .implementation_agent import ImplementationAgent
from .verification_agent import VerificationAgent
from .cpu import CPU, ISA, PipelineTemplate, BranchPredictor
from .cross_disciplinary_agent import (
    CrossDisciplinaryAgent,
    DomainProblem,
    CrossDisciplinaryAnalysis,
    DomainInsight,
    CrossDomainLink,
    SUPPORTED_DOMAINS,
)
from .business_agent import (
    BusinessAgent,
    BusinessProblem,
    BusinessAnalysis,
    BusinessInsight,
    BUSINESS_DOMAINS,
)
from .llm_client import llm_available

# ── Agentic workforce (lazy-importable subpackages) ─────────────────────
# These are subpackages — they need explicit import but are available as:
#   from sub_team.crews import SubTeamWorkforce, AgentRole, TaskType
#   from sub_team.tools import WebSearchTool, CodeExecutorTool, ...
#   from sub_team.memory import AgentMemory, get_memory
#   from sub_team.server import app  # FastAPI ASGI application

__all__ = [
    # Legacy pipeline
    "SpecificationAgent",
    "MicroarchitectureAgent",
    "ImplementationAgent",
    "VerificationAgent",
    "CPU",
    "ISA",
    "PipelineTemplate",
    "BranchPredictor",
    # Cross-disciplinary
    "CrossDisciplinaryAgent",
    "DomainProblem",
    "CrossDisciplinaryAnalysis",
    "DomainInsight",
    "CrossDomainLink",
    "SUPPORTED_DOMAINS",
    # Business intelligence
    "BusinessAgent",
    "BusinessProblem",
    "BusinessAnalysis",
    "BusinessInsight",
    "BUSINESS_DOMAINS",
    # LLM utility
    "llm_available",
]
