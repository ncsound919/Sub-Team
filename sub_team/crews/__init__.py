"""
Sub-Team Crews — CrewAI-based agent team definitions.

This package defines the agentic workforce:
  - agents.py   : 8 specialized agent definitions with roles, tools, and backstories
  - tasks.py    : Task templates for cross-disciplinary work
  - workforce.py: Crew orchestrator that routes tasks to the right team
"""

from .agents import (
    get_all_agents,
    get_agent_by_role,
    get_agents_for_roles,
    get_llm,
    AgentRole,
)
from .tasks import create_task, TaskType
from .workforce import SubTeamWorkforce

__all__ = [
    "get_all_agents",
    "get_agent_by_role",
    "get_agents_for_roles",
    "get_llm",
    "AgentRole",
    "create_task",
    "TaskType",
    "SubTeamWorkforce",
]
