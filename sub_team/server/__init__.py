"""
Sub-Team Server — FastAPI HTTP server for Draymond Orchestrator integration.

Replaces subprocess-based invocation with a proper HTTP API that the
Draymond Orchestrator can call like any other agent in the fleet.

Endpoints:
  GET  /health              — Health check
  GET  /capabilities        — List agent capabilities
  POST /execute             — Execute a task (auto-routed or explicit)
  POST /pipeline/cpu        — Run the legacy CPU RTL pipeline
  POST /pipeline/analyze    — Run cross-disciplinary analysis
  POST /pipeline/business   — Run business analysis
  GET  /memory              — Query agent memory
  POST /memory              — Store agent memory

Run via: uvicorn sub_team.server:app --host 0.0.0.0 --port 8050
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import hmac

from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr, validator

_log = logging.getLogger(__name__)

# ============================================================================
# Rate Limiting (in-memory sliding window)
# ============================================================================

_RATE_WINDOW_SECONDS = 60
_RATE_MAX_REQUESTS = int(os.environ.get("SUB_TEAM_RATE_LIMIT", "30"))
_rate_buckets: Dict[str, List[float]] = defaultdict(list)


def _rate_limit(request: Request):
    """Simple sliding-window rate limiter keyed by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - _RATE_WINDOW_SECONDS

    # Prune old entries
    bucket = _rate_buckets[client_ip]
    _rate_buckets[client_ip] = [t for t in bucket if t > window_start]

    if len(_rate_buckets[client_ip]) >= _RATE_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded — max {_RATE_MAX_REQUESTS} requests per {_RATE_WINDOW_SECONDS}s",
        )
    _rate_buckets[client_ip].append(now)


# ============================================================================
# Auth
# ============================================================================

CRON_SECRET = os.environ.get("CRON_SECRET", "")

if not CRON_SECRET:
    _log.warning("CRON_SECRET is not set — all endpoints are unauthenticated!")


def _authorize(authorization: Optional[str] = Header(None)):
    """Validate Bearer token. Skips auth only if CRON_SECRET is unset."""
    if not CRON_SECRET:
        return  # No auth required if no secret configured
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Malformed Authorization header")
    token = authorization[7:]  # Strip "Bearer " prefix safely
    if not hmac.compare_digest(token, CRON_SECRET):
        raise HTTPException(status_code=403, detail="Invalid token")


# ============================================================================
# App
# ============================================================================

app = FastAPI(
    title="Sub-Team Agentic Workforce",
    description="Cross-disciplinary AI agent team with 8 specialized agents",
    version="2.0.0",
    dependencies=[Depends(_authorize), Depends(_rate_limit)],
)


# ============================================================================
# Global exception handler — prevent stack trace leaks
# ============================================================================


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a generic 500 without leaking internals."""
    _log.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


# ============================================================================
# Models
# ============================================================================


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    agents_available: int = 8
    mode: str = "agentic"
    uptime_seconds: float = 0.0


class ExecuteRequest(BaseModel):
    objective: constr(min_length=1, max_length=10000) = Field(
        ..., description="What the agents should accomplish"
    )
    task_type: Optional[str] = Field(
        None, description="Task type (auto-detected if omitted)"
    )
    agent_roles: Optional[List[str]] = Field(
        None, description="Specific agent roles to use"
    )
    extra_instructions: constr(max_length=5000) = Field(
        "", description="Additional context or constraints"
    )
    output_file: Optional[str] = Field(None, description="Write output to this file")
    execution_mode: Optional[str] = Field(
        None, description="sequential or hierarchical"
    )

    @validator("output_file")
    def validate_output_file(cls, v):
        if v is not None:
            import os as _os

            workspace = _os.environ.get("SUB_TEAM_WORKSPACE", _os.getcwd())
            resolved = _os.path.realpath(_os.path.abspath(v))
            if not resolved.startswith(_os.path.realpath(_os.path.abspath(workspace))):
                raise ValueError("output_file must be within the workspace directory")
        return v


class ExecuteResponse(BaseModel):
    success: bool
    task_type: str
    objective: str
    output: str
    agents_used: List[str]
    execution_mode: str
    duration_ms: int
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None


class CpuPipelineRequest(BaseModel):
    isa: str = Field("RV32IM", description="ISA name (e.g., RV32I, RV32IM, RV64I)")
    pipeline: str = Field("FIVE_STAGE", description="Pipeline template")
    forwarding: bool = Field(True, description="Enable data forwarding")
    branch_predictor_bits: int = Field(8, description="Branch predictor bits (gshare)")
    output_dir: str = Field("rtl_out", description="Output directory for RTL files")

    @validator("output_dir")
    def validate_output_dir(cls, v):
        import os as _os

        workspace = _os.environ.get("SUB_TEAM_WORKSPACE", _os.getcwd())
        resolved = _os.path.realpath(_os.path.abspath(v))
        if not resolved.startswith(_os.path.realpath(_os.path.abspath(workspace))):
            raise ValueError("output_dir must be within the workspace directory")
        return v


class CpuPipelineResponse(BaseModel):
    success: bool
    isa: str
    pipeline: str
    verification_passed: bool
    output_dir: str
    error: Optional[str] = None


class AnalyzeRequest(BaseModel):
    name: str = Field(..., description="Problem name")
    domains: List[str] = Field(..., description="Domains to analyze")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    use_llm: bool = Field(False, description="Enable LLM augmentation")


class MemoryRequest(BaseModel):
    content: str = Field(..., description="Memory content to store")
    agent_id: str = Field("team_shared", description="Agent namespace")
    metadata: Optional[Dict[str, Any]] = None


class MemorySearchRequest(BaseModel):
    query: constr(min_length=1, max_length=2000) = Field(
        ..., description="Search query"
    )
    agent_id: str = Field("team_shared")
    limit: int = Field(5, ge=1, le=100)


# ============================================================================
# Startup
# ============================================================================

_start_time = time.time()


# ============================================================================
# Routes
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/capabilities")
async def capabilities():
    """List all agent capabilities and task types."""
    from sub_team.crews import AgentRole, TaskType

    return {
        "agents": [{"role": role.value, "name": role.name} for role in AgentRole],
        "task_types": [t.value for t in TaskType],
        "total_agents": len(AgentRole),
        "total_task_types": len(TaskType),
        "supported_isa": ["RV32I", "RV32IM", "RV32IMA", "RV64I", "RV64IM"],
        "supported_pipelines": [
            "SINGLE_CYCLE",
            "MULTI_CYCLE",
            "FIVE_STAGE",
            "OUT_OF_ORDER",
        ],
        "domains": {
            "cross_disciplinary": [
                "logistics",
                "biotech",
                "fintech",
                "probability",
                "legal",
            ],
            "business": ["finance", "sales"],
        },
    }


@app.post("/execute", response_model=ExecuteResponse)
async def execute_task(req: ExecuteRequest):
    """Execute a task using the agentic workforce."""
    from sub_team.crews import SubTeamWorkforce, TaskType, AgentRole
    from sub_team.crews.workforce import ExecutionMode

    workforce = SubTeamWorkforce(verbose=True)

    # Parse task type
    task_type = None
    if req.task_type:
        try:
            task_type = TaskType(req.task_type)
        except ValueError:
            raise HTTPException(400, f"Unknown task_type: {req.task_type}")

    # Parse agent roles
    agent_roles = None
    if req.agent_roles:
        try:
            agent_roles = [AgentRole(r) for r in req.agent_roles]
        except ValueError as e:
            raise HTTPException(400, f"Unknown agent_role: {e}")

    # Parse execution mode
    execution_mode = None
    if req.execution_mode:
        try:
            execution_mode = ExecutionMode(req.execution_mode)
        except ValueError:
            raise HTTPException(400, f"Unknown execution_mode: {req.execution_mode}")

    # Execute
    if task_type:
        result = workforce.execute(
            task_type=task_type,
            objective=req.objective,
            agent_roles=agent_roles,
            extra_instructions=req.extra_instructions,
            output_file=req.output_file,
            execution_mode=execution_mode,
        )
    else:
        result = workforce.classify_and_execute(
            objective=req.objective,
            extra_instructions=req.extra_instructions,
            output_file=req.output_file,
        )

    return ExecuteResponse(**result.to_dict())


@app.post("/pipeline/cpu", response_model=CpuPipelineResponse)
async def run_cpu_pipeline(req: CpuPipelineRequest):
    """Run the legacy deterministic CPU RTL pipeline."""
    from sub_team import CPU, ISA, PipelineTemplate
    from sub_team.cpu import gshare
    from sub_team.entry_points import run_pipeline

    try:
        isa = ISA[req.isa]
    except KeyError:
        raise HTTPException(
            400, f"Unknown ISA: {req.isa}. Valid: {[i.name for i in ISA]}"
        )

    try:
        pipeline = PipelineTemplate[req.pipeline]
    except KeyError:
        raise HTTPException(400, f"Unknown pipeline: {req.pipeline}")

    cpu = CPU(
        isa=isa,
        pipeline=pipeline,
        forwarding=req.forwarding,
        branch_predictor=gshare(bits=req.branch_predictor_bits),
    )

    try:
        success = run_pipeline(cpu, rtl_output_dir=req.output_dir)
        return CpuPipelineResponse(
            success=success,
            isa=req.isa,
            pipeline=req.pipeline,
            verification_passed=success,
            output_dir=req.output_dir,
        )
    except Exception as e:
        _log.error("CPU pipeline failed: %s", e, exc_info=True)
        return CpuPipelineResponse(
            success=False,
            isa=req.isa,
            pipeline=req.pipeline,
            verification_passed=False,
            output_dir=req.output_dir,
            error=f"Pipeline failed: {type(e).__name__}",
        )


@app.post("/pipeline/analyze")
async def run_cross_disciplinary(req: AnalyzeRequest):
    """Run the cross-disciplinary analysis agent."""
    from sub_team import CrossDisciplinaryAgent, DomainProblem

    problem = DomainProblem(
        name=req.name,
        domains=req.domains,
        parameters=req.parameters,
    )

    agent = CrossDisciplinaryAgent()
    analysis = agent.run(problem, use_llm=req.use_llm)

    return {
        "name": analysis.problem_name,
        "domains_analyzed": analysis.domains_analyzed,
        "insights_count": len(analysis.insights),
        "overall_risk_score": analysis.overall_risk_score,
        "recommendations": analysis.recommendations,
        "insights": [
            {
                "domain": i.domain,
                "finding": i.finding,
                "confidence": i.confidence,
                "risk_level": i.risk_level,
            }
            for i in analysis.insights
        ],
    }


@app.post("/pipeline/business")
async def run_business_analysis(req: AnalyzeRequest):
    """Run the business intelligence agent."""
    from sub_team import BusinessAgent, BusinessProblem

    problem = BusinessProblem(
        name=req.name,
        domains=req.domains,
        parameters=req.parameters,
    )

    agent = BusinessAgent()
    analysis = agent.run(problem, use_llm=req.use_llm)

    return {
        "name": analysis.problem_name,
        "domains_analyzed": analysis.domains_analyzed,
        "insights_count": len(analysis.insights),
        "overall_risk_score": analysis.overall_risk_score,
        "recommendations": analysis.recommendations,
        "insights": [
            {
                "domain": i.domain,
                "finding": i.finding,
                "metric_name": i.metric_name,
                "metric_value": i.metric_value,
                "confidence": i.confidence,
                "risk_level": i.risk_level,
            }
            for i in analysis.insights
        ],
    }


@app.post("/memory")
async def store_memory(req: MemoryRequest):
    """Store a memory for an agent."""
    from sub_team.memory import get_memory

    memory = get_memory()
    memory_id = memory.add(
        content=req.content,
        agent_id=req.agent_id,
        metadata=req.metadata,
    )
    return {"stored": True, "memory_id": memory_id}


@app.post("/memory/search")
async def search_memory(req: MemorySearchRequest):
    """Search agent memories."""
    from sub_team.memory import get_memory

    memory = get_memory()
    results = memory.search(
        query=req.query,
        agent_id=req.agent_id,
        limit=req.limit,
    )
    return {"results": results, "count": len(results)}


@app.get("/memory/{agent_id}")
async def get_memories(agent_id: str):
    """Get all memories for an agent."""
    from sub_team.memory import get_memory

    memory = get_memory()
    results = memory.get_all(agent_id=agent_id)
    return {"agent_id": agent_id, "memories": results, "count": len(results)}
