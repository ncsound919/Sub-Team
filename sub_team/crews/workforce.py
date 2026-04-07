"""
Sub-Team Workforce — The crew orchestrator.

Routes tasks to the right agents, assembles dynamic crews for
cross-disciplinary work, and manages execution with memory.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from crewai import Crew, Process

from .agents import AgentRole, get_agent_by_role, get_agents_for_roles
from .tasks import TaskType, create_task

_log = logging.getLogger(__name__)


# ============================================================================
# Task Routing — Maps task types to the right agents
# ============================================================================


class ExecutionMode(str, Enum):
    """How a crew should execute its tasks."""

    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"


@dataclass
class TaskRouting:
    """Configuration for how a task type should be routed and executed."""

    primary_agent: AgentRole
    supporting_agents: List[AgentRole] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL


# Maps each task type to the agents best suited to handle it
_ROUTING_TABLE: Dict[TaskType, TaskRouting] = {
    # Research & Analysis
    TaskType.RESEARCH: TaskRouting(
        primary_agent=AgentRole.RESEARCH_ANALYST,
    ),
    TaskType.COMPETITIVE_ANALYSIS: TaskRouting(
        primary_agent=AgentRole.RESEARCH_ANALYST,
        supporting_agents=[AgentRole.BUSINESS_STRATEGIST],
    ),
    TaskType.TECHNOLOGY_EVALUATION: TaskRouting(
        primary_agent=AgentRole.RESEARCH_ANALYST,
        supporting_agents=[AgentRole.SOFTWARE_ENGINEER, AgentRole.SYSTEMS_ARCHITECT],
    ),
    TaskType.MARKET_SIZING: TaskRouting(
        primary_agent=AgentRole.BUSINESS_STRATEGIST,
        supporting_agents=[AgentRole.RESEARCH_ANALYST, AgentRole.DATA_SCIENTIST],
    ),
    # Software Engineering
    TaskType.CODE_GENERATION: TaskRouting(
        primary_agent=AgentRole.SOFTWARE_ENGINEER,
    ),
    TaskType.CODE_REVIEW: TaskRouting(
        primary_agent=AgentRole.SOFTWARE_ENGINEER,
        supporting_agents=[AgentRole.SECURITY_ANALYST],
    ),
    TaskType.BUG_FIX: TaskRouting(
        primary_agent=AgentRole.SOFTWARE_ENGINEER,
    ),
    TaskType.REFACTORING: TaskRouting(
        primary_agent=AgentRole.SOFTWARE_ENGINEER,
        supporting_agents=[AgentRole.SYSTEMS_ARCHITECT],
    ),
    TaskType.TESTING: TaskRouting(
        primary_agent=AgentRole.SOFTWARE_ENGINEER,
    ),
    # Data Science
    TaskType.DATA_ANALYSIS: TaskRouting(
        primary_agent=AgentRole.DATA_SCIENTIST,
    ),
    TaskType.MODEL_BUILDING: TaskRouting(
        primary_agent=AgentRole.DATA_SCIENTIST,
        supporting_agents=[AgentRole.SOFTWARE_ENGINEER],
    ),
    TaskType.VISUALIZATION: TaskRouting(
        primary_agent=AgentRole.DATA_SCIENTIST,
    ),
    # Business
    TaskType.FINANCIAL_MODELING: TaskRouting(
        primary_agent=AgentRole.BUSINESS_STRATEGIST,
        supporting_agents=[AgentRole.DATA_SCIENTIST],
    ),
    TaskType.BUSINESS_PLAN: TaskRouting(
        primary_agent=AgentRole.BUSINESS_STRATEGIST,
        supporting_agents=[AgentRole.RESEARCH_ANALYST, AgentRole.CREATIVE_DIRECTOR],
        execution_mode=ExecutionMode.HIERARCHICAL,
    ),
    TaskType.PITCH_DECK: TaskRouting(
        primary_agent=AgentRole.BUSINESS_STRATEGIST,
        supporting_agents=[AgentRole.CREATIVE_DIRECTOR, AgentRole.DATA_SCIENTIST],
        execution_mode=ExecutionMode.HIERARCHICAL,
    ),
    # Creative
    TaskType.CONTENT_CREATION: TaskRouting(
        primary_agent=AgentRole.CREATIVE_DIRECTOR,
        supporting_agents=[AgentRole.RESEARCH_ANALYST],
    ),
    TaskType.COPYWRITING: TaskRouting(
        primary_agent=AgentRole.CREATIVE_DIRECTOR,
    ),
    TaskType.DOCUMENTATION: TaskRouting(
        primary_agent=AgentRole.CREATIVE_DIRECTOR,
        supporting_agents=[AgentRole.SOFTWARE_ENGINEER],
    ),
    # Security
    TaskType.SECURITY_AUDIT: TaskRouting(
        primary_agent=AgentRole.SECURITY_ANALYST,
        supporting_agents=[AgentRole.SOFTWARE_ENGINEER],
    ),
    TaskType.THREAT_MODEL: TaskRouting(
        primary_agent=AgentRole.SECURITY_ANALYST,
        supporting_agents=[AgentRole.SYSTEMS_ARCHITECT],
    ),
    TaskType.COMPLIANCE_CHECK: TaskRouting(
        primary_agent=AgentRole.SECURITY_ANALYST,
    ),
    # Architecture
    TaskType.SYSTEM_DESIGN: TaskRouting(
        primary_agent=AgentRole.SYSTEMS_ARCHITECT,
        supporting_agents=[AgentRole.SOFTWARE_ENGINEER, AgentRole.SECURITY_ANALYST],
        execution_mode=ExecutionMode.HIERARCHICAL,
    ),
    TaskType.API_DESIGN: TaskRouting(
        primary_agent=AgentRole.SYSTEMS_ARCHITECT,
        supporting_agents=[AgentRole.SOFTWARE_ENGINEER],
    ),
    TaskType.DATABASE_DESIGN: TaskRouting(
        primary_agent=AgentRole.SYSTEMS_ARCHITECT,
        supporting_agents=[AgentRole.DATA_SCIENTIST],
    ),
    # Hardware
    TaskType.CPU_DESIGN: TaskRouting(
        primary_agent=AgentRole.HARDWARE_ENGINEER,
        supporting_agents=[AgentRole.SOFTWARE_ENGINEER],
    ),
    TaskType.RTL_GENERATION: TaskRouting(
        primary_agent=AgentRole.HARDWARE_ENGINEER,
    ),
    TaskType.HARDWARE_VERIFICATION: TaskRouting(
        primary_agent=AgentRole.HARDWARE_ENGINEER,
    ),
    # Cross-disciplinary
    TaskType.CROSS_DOMAIN: TaskRouting(
        primary_agent=AgentRole.SYSTEMS_ARCHITECT,
        supporting_agents=[
            AgentRole.RESEARCH_ANALYST,
            AgentRole.SOFTWARE_ENGINEER,
            AgentRole.DATA_SCIENTIST,
            AgentRole.BUSINESS_STRATEGIST,
        ],
        execution_mode=ExecutionMode.HIERARCHICAL,
    ),
}


# ============================================================================
# Execution Results
# ============================================================================


@dataclass
class WorkforceResult:
    """Result from a workforce execution."""

    task_type: str
    objective: str
    output: str
    agents_used: List[str]
    execution_mode: str
    duration_ms: int
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "objective": self.objective[:200],
            "output": self.output,
            "agents_used": self.agents_used,
            "execution_mode": self.execution_mode,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage,
            "error": self.error,
            "success": self.success,
        }


# ============================================================================
# Workforce Orchestrator
# ============================================================================


class SubTeamWorkforce:
    """
    The Sub-Team agentic workforce.

    Orchestrates CrewAI agents to handle cross-disciplinary tasks.
    Automatically routes tasks to the right agents, assembles crews,
    and manages execution.

    Usage::

        workforce = SubTeamWorkforce()

        # Simple task
        result = workforce.execute(
            task_type=TaskType.RESEARCH,
            objective="Analyze the current state of multi-agent AI frameworks"
        )

        # Cross-disciplinary task with specific agents
        result = workforce.execute(
            task_type=TaskType.CROSS_DOMAIN,
            objective="Design a real-time trading system with ML-based signals",
            agent_roles=[
                AgentRole.SYSTEMS_ARCHITECT,
                AgentRole.DATA_SCIENTIST,
                AgentRole.SOFTWARE_ENGINEER,
                AgentRole.SECURITY_ANALYST,
            ]
        )

        # Custom task
        result = workforce.execute_custom(
            objective="...",
            agents=[AgentRole.SOFTWARE_ENGINEER],
        )
    """

    def __init__(self, *, verbose: bool = True):
        self.verbose = verbose
        self._memory_store = None  # Lazy-loaded mem0 store

    def execute(
        self,
        task_type: TaskType,
        objective: str,
        *,
        agent_roles: Optional[List[AgentRole]] = None,
        extra_instructions: str = "",
        output_file: Optional[str] = None,
        execution_mode: Optional[ExecutionMode] = None,
    ) -> WorkforceResult:
        """
        Execute a task using the optimal agent team.

        Parameters
        ----------
        task_type : TaskType
            Type of task to execute.
        objective : str
            The specific objective/brief.
        agent_roles : list[AgentRole], optional
            Override the default agent routing with specific roles.
        extra_instructions : str
            Additional context or constraints.
        output_file : str, optional
            Write output to this file.
        execution_mode : ExecutionMode, optional
            Override the default execution mode.

        Returns
        -------
        WorkforceResult
            The execution result with output, timing, and metadata.
        """
        start_time = time.time()

        try:
            # Resolve routing
            routing = _ROUTING_TABLE.get(task_type)
            if routing is None and task_type == TaskType.CUSTOM:
                if not agent_roles:
                    raise ValueError(
                        "CUSTOM task type requires explicit agent_roles — "
                        "no default routing exists for CUSTOM tasks"
                    )
                routing = TaskRouting(primary_agent=agent_roles[0])
            elif routing is None:
                routing = TaskRouting(
                    primary_agent=AgentRole.RESEARCH_ANALYST,
                )

            # Determine agents
            if agent_roles:
                roles_to_use = agent_roles
            else:
                roles_to_use = [routing.primary_agent] + routing.supporting_agents

            # Determine execution mode
            mode = execution_mode or routing.execution_mode

            # Build agents
            agents = get_agents_for_roles(roles_to_use)

            # Build tasks — primary agent gets the main task
            primary_task = create_task(
                task_type=task_type,
                objective=objective,
                agent=agents[0],
                extra_instructions=extra_instructions,
                output_file=output_file,
            )

            tasks = [primary_task]

            # For multi-agent crews, add supporting review/validation tasks
            if len(agents) > 1 and mode == ExecutionMode.SEQUENTIAL:
                for i, agent in enumerate(agents[1:], 1):
                    review_task = create_task(
                        task_type=TaskType.CUSTOM,
                        objective=(
                            f"Review and enhance the output from the previous task. "
                            f"Apply your expertise as {agent.role} to validate, "
                            f"add missing insights, and improve the overall quality. "
                            f"Focus on areas specific to your domain expertise."
                        ),
                        agent=agent,
                        context_tasks=[primary_task],
                    )
                    tasks.append(review_task)

            # Assemble and run the crew
            crew_process = (
                Process.hierarchical
                if mode == ExecutionMode.HIERARCHICAL
                else Process.sequential
            )

            crew_kwargs: Dict[str, Any] = {
                "agents": agents,
                "tasks": tasks,
                "process": crew_process,
                "verbose": self.verbose,
                "memory": True,
            }

            # Hierarchical mode needs a manager LLM
            if mode == ExecutionMode.HIERARCHICAL:
                from .agents import get_llm

                crew_kwargs["manager_llm"] = get_llm()

            crew = Crew(**crew_kwargs)
            result = crew.kickoff()

            duration_ms = int((time.time() - start_time) * 1000)

            # Extract output
            output_text = str(result)

            # Try to get token usage
            token_usage = None
            if hasattr(result, "token_usage") and result.token_usage:
                token_usage = {
                    "total_tokens": getattr(result.token_usage, "total_tokens", 0),
                    "prompt_tokens": getattr(result.token_usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        result.token_usage, "completion_tokens", 0
                    ),
                }

            return WorkforceResult(
                task_type=task_type.value,
                objective=objective,
                output=output_text,
                agents_used=[a.role for a in agents],
                execution_mode=mode.value,
                duration_ms=duration_ms,
                token_usage=token_usage,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            _log.error("Workforce execution failed: %s", e, exc_info=True)
            return WorkforceResult(
                task_type=task_type.value,
                objective=objective,
                output="",
                agents_used=[],
                execution_mode="failed",
                duration_ms=duration_ms,
                error=f"Execution failed: {type(e).__name__}",
            )

    def execute_custom(
        self,
        objective: str,
        agents: List[AgentRole],
        *,
        extra_instructions: str = "",
        output_file: Optional[str] = None,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    ) -> WorkforceResult:
        """
        Execute a custom task with explicitly specified agents.

        This bypasses the routing table and lets you compose any team.
        """
        return self.execute(
            task_type=TaskType.CUSTOM,
            objective=objective,
            agent_roles=agents,
            extra_instructions=extra_instructions,
            output_file=output_file,
            execution_mode=execution_mode,
        )

    def classify_and_execute(
        self,
        objective: str,
        *,
        extra_instructions: str = "",
        output_file: Optional[str] = None,
    ) -> WorkforceResult:
        """
        Auto-classify the objective and route to the best task type and agents.

        Uses keyword matching for fast classification. Falls back to
        CROSS_DOMAIN for ambiguous requests.
        """
        task_type = self._classify_task(objective)
        _log.info("Auto-classified task as: %s", task_type.value)

        return self.execute(
            task_type=task_type,
            objective=objective,
            extra_instructions=extra_instructions,
            output_file=output_file,
        )

    def _classify_task(self, objective: str) -> TaskType:
        """Classify an objective into a TaskType using keyword matching.

        Uses word-boundary matching (``\\b``) so that e.g. "code" does not
        match "barcode" and "error" does not match "error-free".
        """
        obj_lower = objective.lower()

        def _has_keyword(kw: str) -> bool:
            """Return True if *kw* appears as a whole phrase in *obj_lower*."""
            # For multi-word keywords, escape and match literally with
            # word boundaries at both ends.
            return bool(re.search(r"\b" + re.escape(kw) + r"\b", obj_lower))

        # Keyword -> TaskType mapping (checked in priority order)
        keyword_map = [
            (
                ["security audit", "vulnerability", "penetration test", "owasp"],
                TaskType.SECURITY_AUDIT,
            ),
            (
                ["threat model", "attack surface", "threat assessment"],
                TaskType.THREAT_MODEL,
            ),
            (
                ["compliance", "gdpr", "hipaa", "soc 2", "pci"],
                TaskType.COMPLIANCE_CHECK,
            ),
            (
                ["cpu design", "risc-v", "processor", "isa specification"],
                TaskType.CPU_DESIGN,
            ),
            (
                ["rtl", "verilog", "vhdl", "fpga", "synthesizable"],
                TaskType.RTL_GENERATION,
            ),
            (
                ["hardware verification", "formal verification"],
                TaskType.HARDWARE_VERIFICATION,
            ),
            (
                ["fix bug", "bug fix", "debug", "error", "crash", "broken"],
                TaskType.BUG_FIX,
            ),
            (
                ["code review", "review code", "review the code", "pr review"],
                TaskType.CODE_REVIEW,
            ),
            (["refactor", "clean up", "simplify", "restructure"], TaskType.REFACTORING),
            (["write test", "unit test", "test case", "testing"], TaskType.TESTING),
            (
                ["implement", "build", "create function", "write code", "code"],
                TaskType.CODE_GENERATION,
            ),
            (
                ["analyze data", "data analysis", "dataset", "csv", "statistics"],
                TaskType.DATA_ANALYSIS,
            ),
            (
                ["ml model", "machine learning", "train", "predict", "classifier"],
                TaskType.MODEL_BUILDING,
            ),
            (
                ["visualiz", "chart", "graph", "plot", "dashboard"],
                TaskType.VISUALIZATION,
            ),
            (
                ["financial model", "revenue projection", "p&l", "unit economics"],
                TaskType.FINANCIAL_MODELING,
            ),
            (["business plan", "go-to-market", "gtm"], TaskType.BUSINESS_PLAN),
            (["pitch deck", "investor", "fundrais"], TaskType.PITCH_DECK),
            (
                ["market size", "tam", "sam", "som", "market opportunity"],
                TaskType.MARKET_SIZING,
            ),
            (
                ["competitive analysis", "competitor", "landscape"],
                TaskType.COMPETITIVE_ANALYSIS,
            ),
            (
                ["evaluate", "compare", "assess", "framework comparison"],
                TaskType.TECHNOLOGY_EVALUATION,
            ),
            (["system design", "architecture", "architect"], TaskType.SYSTEM_DESIGN),
            (
                ["api design", "rest api", "endpoint design", "api contract"],
                TaskType.API_DESIGN,
            ),
            (
                ["database", "schema design", "data model", "erd"],
                TaskType.DATABASE_DESIGN,
            ),
            (
                ["blog post", "article", "content", "social media", "newsletter"],
                TaskType.CONTENT_CREATION,
            ),
            (
                ["copy", "headline", "tagline", "slogan", "ad copy"],
                TaskType.COPYWRITING,
            ),
            (["document", "readme", "guide", "tutorial"], TaskType.DOCUMENTATION),
            (
                ["research", "investigate", "explore", "find out", "what is"],
                TaskType.RESEARCH,
            ),
        ]

        for keywords, task_type in keyword_map:
            if any(_has_keyword(kw) for kw in keywords):
                return task_type

        # Default to cross-domain for ambiguous requests
        return TaskType.CROSS_DOMAIN

    @staticmethod
    def list_capabilities() -> Dict[str, Any]:
        """Return a summary of all workforce capabilities."""
        return {
            "agents": [
                {
                    "role": role.value,
                    "display_name": role.name.replace("_", " ").title(),
                }
                for role in AgentRole
            ],
            "task_types": [t.value for t in TaskType],
            "total_agents": len(AgentRole),
            "total_task_types": len(TaskType),
        }
