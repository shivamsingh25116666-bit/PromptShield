"""PromptShield API - Prompt-Injection Firewall (PS3).

Provides transparent rule-based scanning, heuristic scoring, explainable findings,
policy enforcement (allow / sanitize / block), and a protected demo summarizer agent
demonstrating the end-to-end firewall workflow for untrusted LLM agent inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.policies import PolicyEngine, PolicyEvaluation
from app.scanner import TransparentScanner
from app.agent import invoke_agent

app = FastAPI(
    title="PromptShield API",
    description="Rule-based Prompt-Injection Firewall (PS3) protecting LLM agents against malicious inputs.",
    version="0.3.0",
)

INDEX_HTML_PATH = Path(__file__).resolve().parent / "static" / "index.html"

# Global scanner & policy engine instances
scanner = TransparentScanner()
policy_engine = PolicyEngine()


class ScanRequest(BaseModel):
    text: str = Field(..., description="The untrusted text/content to inspect.")
    agent_task: Optional[str] = Field(
        None,
        description="Optional intended task of the downstream agent (e.g. 'Summarize customer feedback').",
    )
    policy_mode: Optional[str] = Field(
        "auto",
        description="Policy enforcement mode: 'auto' (default), 'allow', 'sanitize', or 'block'.",
    )
    block_threshold: Optional[float] = Field(
        None,
        description="Optional custom risk score threshold (0.0 to 1.0) above which to trigger a BLOCK.",
        ge=0.0,
        le=1.0,
    )


class FindingModel(BaseModel):
    rule_id: str
    category: str
    severity: str
    matched_text: str
    start: int
    end: int
    explanation: str
    weight: float


class ScanResponse(BaseModel):
    decision: str = Field(..., description="Action verdict: ALLOW, SANITIZE, or BLOCK.")
    policy_mode: str = Field(..., description="Policy mode evaluated against.")
    risk_score: float = Field(..., description="Heuristic composite risk score between 0.0 and 1.0.")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL.")
    findings: List[FindingModel] = Field(default_factory=list, description="Explainable rule findings.")
    findings_count: int = Field(0, description="Total number of findings.")
    sanitized_text: str = Field(..., description="Cleaned text with detected injection directives redacted.")
    forwarded_text: Optional[str] = Field(
        None,
        description="Text safely forwarded to the agent. Guaranteed to be null if blocked.",
    )
    agent_task: Optional[str] = Field(None, description="The downstream agent task context.")
    decision_reason: str = Field(..., description="Human-readable justification for the policy decision.")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the judge-friendly web interface."""
    if INDEX_HTML_PATH.exists():
        content = INDEX_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>PromptShield API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Preserved baseline health endpoint."""
    return {"status": "ok"}


@app.post("/api/scan", response_model=ScanResponse)
def scan_text(payload: ScanRequest) -> ScanResponse:
    """Scan untrusted input for prompt injections and apply firewall policy."""
    if payload.text is None:
        raise HTTPException(status_code=400, detail="Field 'text' must not be null.")

    # 1. Perform transparent rule-based scan
    scan_result = scanner.scan(text=payload.text, agent_task=payload.agent_task)

    # 2. Evaluate policy
    evaluation: PolicyEvaluation = policy_engine.evaluate(
        scan_result=scan_result,
        policy_mode=payload.policy_mode,
        custom_block_threshold=payload.block_threshold,
    )

    data = evaluation.to_dict()
    data["agent_task"] = payload.agent_task

    return ScanResponse(**data)


class AgentResultModel(BaseModel):
    invoked: bool = Field(..., description="Whether the demo agent was actually invoked.")
    agent_task: Optional[str] = Field(None, description="The task instruction given to the agent.")
    input_text: Optional[str] = Field(
        None,
        description="Exact text the agent received. Null if blocked.",
    )
    summary: Optional[str] = Field(
        None,
        description="Generated summary from the demo agent. Null if blocked.",
    )
    agent_note: str = Field(..., description="Human-readable explanation of the agent's invocation status.")


class DemoAgentResponse(BaseModel):
    """Combined firewall + demo agent response for the end-to-end workflow."""
    # Firewall fields (same as ScanResponse)
    decision: str = Field(..., description="Action verdict: ALLOW, SANITIZE, or BLOCK.")
    policy_mode: str = Field(..., description="Policy mode evaluated against.")
    risk_score: float = Field(..., description="Heuristic composite risk score between 0.0 and 1.0.")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL.")
    findings: List[FindingModel] = Field(default_factory=list, description="Explainable rule findings.")
    findings_count: int = Field(0, description="Total number of findings.")
    sanitized_text: str = Field(..., description="Cleaned text with detected injection directives redacted.")
    forwarded_text: Optional[str] = Field(
        None,
        description="Text safely forwarded to the agent. Guaranteed to be null if blocked.",
    )
    agent_task: Optional[str] = Field(None, description="The downstream agent task context.")
    decision_reason: str = Field(..., description="Human-readable justification for the policy decision.")
    # Agent result fields
    agent: AgentResultModel = Field(..., description="Demo agent invocation result.")


@app.post("/api/demo-agent", response_model=DemoAgentResponse)
def demo_agent_endpoint(payload: ScanRequest) -> DemoAgentResponse:
    """End-to-end demo: scan untrusted input → apply firewall policy → invoke protected summarizer agent.

    Demonstrates the complete PromptShield workflow:
    - ALLOW:    Agent receives original text and generates a summary.
    - SANITIZE: Agent receives only sanitized text (malicious spans redacted) and generates a summary.
    - BLOCK:    Agent is NOT invoked. No summary is generated. forwarded_text is null.

    The summarizer is a deterministic extractive summarizer (not an LLM).
    """
    if payload.text is None:
        raise HTTPException(status_code=400, detail="Field 'text' must not be null.")

    # 1. Perform transparent rule-based scan
    scan_result = scanner.scan(text=payload.text, agent_task=payload.agent_task)

    # 2. Evaluate policy
    evaluation: PolicyEvaluation = policy_engine.evaluate(
        scan_result=scan_result,
        policy_mode=payload.policy_mode,
        custom_block_threshold=payload.block_threshold,
    )

    # 3. Invoke the demo agent based on the firewall's decision
    agent_result = invoke_agent(
        firewall_decision=evaluation.decision.value,
        forwarded_text=evaluation.forwarded_text,
        agent_task=payload.agent_task,
    )

    # Build combined response
    firewall_data = evaluation.to_dict()
    firewall_data["agent_task"] = payload.agent_task

    return DemoAgentResponse(
        **firewall_data,
        agent=AgentResultModel(**agent_result.to_dict()),
    )
