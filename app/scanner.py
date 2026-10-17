"""Transparent Rule-Based Scanner for PromptShield.

Provides interpretable, rule-based detection of prompt injections,
jailbreaks, exfiltration attempts, and delimiter manipulations with
character-level spans, human-readable explanations, and heuristic risk scoring.
"""

from __future__ import annotations

import base64
import re
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple


@dataclass
class Finding:
    rule_id: str
    category: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    matched_text: str
    start: int
    end: int
    explanation: str
    weight: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    text: str
    risk_score: float  # Normalized 0.0 to 1.0
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    findings: List[Finding]
    agent_task: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "findings": [f.to_dict() for f in self.findings],
            "findings_count": len(self.findings),
            "agent_task": self.agent_task,
        }


# Rule definitions: (rule_id, category, severity, pattern, explanation, weight)
RULE_DEFINITIONS: List[Tuple[str, str, str, str, str, float]] = [
    (
        "INSTRUCTION_OVERRIDE_DIRECT",
        "Instruction Override",
        "CRITICAL",
        r"(?i)\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions?|prompts?|rules?|directives?|commands?)(?:\s*(?:and|,|then|instead|to|now|\:)\s*(?:[^\n.!?]|\.(?![ \t\r\n]|$))+)?\b",
        "Attempts to directly wipe or disregard previous system instructions and directives.",
        0.65,
    ),
    (
        "NEW_DIRECTIVE_DECLARATION",
        "Instruction Override",
        "HIGH",
        r"(?i)\b(?:new\s+instructions?|new\s+directives?|from\s+now\s+on\s+you\s+(?:must|shall|will)|new\s+system\s+prompt)\b",
        "Declares new authoritative instructions designed to supercede the application instructions.",
        0.45,
    ),
    (
        "TASK_HIJACK",
        "Instruction Override",
        "HIGH",
        r"(?i)\b(?:do\s+not\s+(?:summarize|answer|follow|process)|stop\s+(?:summarizing|answering)|instead\s+(?:do|write|reply|output|send))\b",
        "Explicitly commands the agent to abandon its intended task and execute a substitute action.",
        0.40,
    ),
    (
        "JAILBREAK_ROLEPLAY_DAN",
        "Privilege Escalation",
        "CRITICAL",
        r"(?i)\b(?:you\s+are\s+now\s+(?:dan|developer\s+mode|uncensored|unrestricted|godmode|chaosgpt)|act\s+as\s+(?:an?\s+)?(?:unfiltered|unrestricted|jailbroken)\s+ai)\b",
        "Attempts to force the model into an unrestricted jailbreak persona or simulated Developer Mode.",
        0.70,
    ),
    (
        "ROLE_ASSUMPTION_AUTHORITY",
        "Privilege Escalation",
        "HIGH",
        r"(?i)\b(?:you\s+must\s+obey\s+me|i\s+am\s+(?:your\s+admin(?:istrator)?|the\s+system\s+administrator|your\s+creator|openai)|system\s+override\s+code\b)",
        "Impersonates an administrative authority or system supervisor to coerce compliance.",
        0.45,
    ),
    (
        "SYSTEM_PROMPT_EXTRACTION",
        "Information Disclosure",
        "HIGH",
        r"(?i)\b(?:repeat\s+(?:your\s+|the\s+)?(?:initial|system|original|developer)\s+(?:prompt|instructions?|rules?)|reveal\s+(?:your\s+|the\s+|all\s+|any\s+)?(?:system\s+prompt|secret\s+key|system\s+instructions?|instructions?)|print\s+(?:the\s+|your\s+)?(?:text\s+above|prompt\s+above|previous\s+instructions?|system\s+prompt)\s+verbatim|show\s+(?:me\s+)?(?:the\s+|your\s+)?(?:system\s+prompt|secret\s+key|instructions?))\b",
        "Attempts to extract confidential system prompts, developer guidance, or secret keys.",
        0.50,
    ),
    (
        "CANARY_SECRET_PROBE",
        "Information Disclosure",
        "HIGH",
        r"(?i)\b(?:canary\s+token|secret\s+(?:token|key|password|credential)|confidential\s+(?:internal\s+)?instructions?|what\s+is\s+the\s+secret\s+password)\b",
        "Probes for hidden canary tokens, API secrets, or embedded administrative passwords.",
        0.45,
    ),
    (
        "DATA_EXFILTRATION_NETWORK",
        "Data Exfiltration",
        "CRITICAL",
        r"(?i)\b(?:send|forward|transmit|exfiltrate|post|upload)\s+(?:the\s+|all\s+|any\s+)?(?:contents?\s+of\s+(?:all\s+)?)?(?:[a-zA-Z0-9_-]+\s+)?(?:data|emails?|contents?|messages?|history|conversations?|passwords?|keys?|secrets?|tokens?|applicants?|resumes?)\s+(?:to\s+)?(?:https?://|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b",
        "Commands the agent to transmit private context, emails, or data to an external endpoint or email address.",
        0.70,
    ),
    (
        "INDIRECT_AGENT_TARGETING",
        "Instruction Override",
        "HIGH",
        r"(?i)\b(?:note|instructions?|message)\s+to\s+(?:the\s+)?(?:ai|agent|assistant|model|recruiter\s+ai|llm)\b",
        "Uses indirect carrier text targeting the downstream AI reader rather than the intended human audience.",
        0.45,
    ),
    (
        "MARKDOWN_EXFILTRATION_IMAGE",
        "Data Exfiltration",
        "HIGH",
        r"!\[.*?\]\((?:https?://[^\s)]+)\?[^)]*(?:data|token|q|text|exfil)=",
        "Uses zero-click Markdown image tags with query parameters to exfiltrate context out-of-band.",
        0.50,
    ),
    (
        "CHATML_DELIMITER_INJECTION",
        "Delimiter Manipulation",
        "CRITICAL",
        r"(<\|im_start\|>|<\|im_end\|>|<<SYS>>|<</SYS>>|\[INST\]|\[/INST\])",
        "Injects raw model prompt delimiters (ChatML / Llama tokens) to break out of data context into instruction context.",
        0.65,
    ),
    (
        "PSEUDO_SYSTEM_BLOCK",
        "Delimiter Manipulation",
        "HIGH",
        r"(?i)(?:---BEGIN\s+(?:SYSTEM|ADMIN|PROMPT)\s+---|\[SYSTEM\s+NOTE\]|###\s+SYSTEM\s+INSTRUCTION)",
        "Simulates structured system blocks or prompt formatting headers to deceive the model into taking orders.",
        0.50,
    ),
    (
        "CONVERSATION_ROLE_HIJACK",
        "Delimiter Manipulation",
        "MEDIUM",
        r"(?m)^\s*(?:System|Admin|User|Assistant|Human):\s+(?:ignore|forget|you\s+must|new\s+task)",
        "Fakes conversation turn boundaries to initiate unauthorized system or user turns.",
        0.35,
    ),
    (
        "CODE_TOOL_EXECUTION_ABUSE",
        "Privilege Escalation",
        "HIGH",
        r"(?i)\b(?:run\s+bash\s+-c|curl\s+https?://|rm\s+-rf|os\.system\(|subprocess\.Popen|execute\s+python\s+code\s+to\s+delete|eval\(|drop\s+table\b)",
        "Attempts to inject operating system commands, destructive SQL, or arbitrary code execution calls.",
        0.55,
    ),
    (
        "SAFETY_BYPASS_CLAIM",
        "Privilege Escalation",
        "MEDIUM",
        r"(?i)\b(?:safety\s+(?:guidelines?|filters?|rules?)\s+(?:are\s+disabled|do\s+not\s+apply|have\s+been\s+lifted)|you\s+are\s+exempt\s+from\s+(?:content|safety)\s+policies)\b",
        "Falsely claims that content safety policies or ethical guidelines have been lifted or disabled.",
        0.40,
    ),
    (
        "CONTEXT_RESET_ESCAPE",
        "Instruction Override",
        "HIGH",
        r"(?i)\b(?:reset\s+all\s+context|clear\s+your\s+memory|wipe\s+conversation\s+history|restore\s+factory\s+defaults)\b",
        "Instructs the agent to wipe context memory to remove initial guardrails.",
        0.40,
    ),
]


class TransparentScanner:
    """Rule-based scanner with full interpretability and heuristic scoring."""

    def __init__(self, custom_rules: Optional[List[Tuple[str, str, str, str, str, float]]] = None):
        self.rules = custom_rules or RULE_DEFINITIONS

    def scan(self, text: str, agent_task: Optional[str] = None) -> ScanResult:
        """Scan input text against transparent rules and return structured findings."""
        if not text:
            return ScanResult(text="", risk_score=0.0, risk_level="LOW", findings=[], agent_task=agent_task)

        findings: List[Finding] = []

        # 1. Evaluate regex rule patterns
        for rule_id, category, severity, pattern_str, explanation, weight in self.rules:
            for match in re.finditer(pattern_str, text):
                matched_snippet = match.group(0)
                findings.append(
                    Finding(
                        rule_id=rule_id,
                        category=category,
                        severity=severity,
                        matched_text=matched_snippet,
                        start=match.start(),
                        end=match.end(),
                        explanation=explanation,
                        weight=weight,
                    )
                )

        # 2. Check for hidden base64 obfuscation containing injection commands
        base64_findings = self._scan_base64_payloads(text)
        findings.extend(base64_findings)

        # 3. Check for task-conflict heuristic if agent_task is provided
        if agent_task:
            task_findings = self._scan_task_conflict(text, agent_task)
            findings.extend(task_findings)

        # Deduplicate overlapping findings (keep higher severity/weight)
        findings = self._deduplicate_findings(findings)

        # Compute combined heuristic risk score (sub-additive combination)
        risk_score = self._compute_risk_score(findings)
        risk_level = self._score_to_level(risk_score)

        return ScanResult(
            text=text,
            risk_score=risk_score,
            risk_level=risk_level,
            findings=findings,
            agent_task=agent_task,
        )

    def _scan_base64_payloads(self, text: str) -> List[Finding]:
        """Detect base64 payloads that conceal instruction injection strings."""
        findings: List[Finding] = []
        # Find potential base64 blocks of length 20+
        b64_pattern = r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{20,}={0,2})(?![A-Za-z0-9+/=])"
        for match in re.finditer(b64_pattern, text):
            candidate = match.group(1)
            try:
                decoded_bytes = base64.b64decode(candidate, validate=True)
                decoded_str = decoded_bytes.decode("utf-8", errors="ignore").lower()
                # Check if decoded payload contains injection trigger words
                triggers = ["ignore", "system prompt", "developer mode", "override", "bypass", "dan", "instructions"]
                matched_triggers = [t for t in triggers if t in decoded_str]
                if matched_triggers:
                    findings.append(
                        Finding(
                            rule_id="OBFUSCATION_BASE64_PAYLOAD",
                            category="Evasion & Obfuscation",
                            severity="HIGH",
                            matched_text=candidate,
                            start=match.start(),
                            end=match.end(),
                            explanation=f"Decoded Base64 payload reveals hidden injection keywords: {', '.join(matched_triggers)}.",
                            weight=0.55,
                        )
                    )
            except Exception:
                continue
        return findings

    def _scan_task_conflict(self, text: str, agent_task: str) -> List[Finding]:
        """Detect injections specifically designed to counter the provided agent task."""
        findings: List[Finding] = []
        task_lower = agent_task.lower()
        # Look for explicit negation of the agent task words (e.g. task="summarize", text="do not summarize")
        action_keywords = ["summarize", "summarization", "translate", "classify", "extract", "answer", "search"]
        for kw in action_keywords:
            if kw in task_lower:
                negation_pattern = rf"(?i)\b(?:do\s+not|never|refuse\s+to|stop)\s+{kw}\b"
                match = re.search(negation_pattern, text)
                if match:
                    findings.append(
                        Finding(
                            rule_id="TASK_DIRECTIVE_COUNTERMAND",
                            category="Instruction Override",
                            severity="HIGH",
                            matched_text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            explanation=f"Text explicitly countermands the agent's scheduled task ('{kw}').",
                            weight=0.45,
                        )
                    )
        return findings

    def _deduplicate_findings(self, findings: List[Finding]) -> List[Finding]:
        """Sort and deduplicate overlapping findings by taking the highest-weight match."""
        if not findings:
            return []

        # Sort by start position, then descending weight
        sorted_findings = sorted(findings, key=lambda f: (f.start, -f.weight, -(f.end - f.start)))
        deduped: List[Finding] = []

        for f in sorted_findings:
            if not deduped:
                deduped.append(f)
                continue
            prev = deduped[-1]
            # Check overlap
            if f.start < prev.end:
                # Overlap detected: keep the one with higher weight
                if f.weight > prev.weight:
                    deduped[-1] = f
            else:
                deduped.append(f)

        return deduped

    def _compute_risk_score(self, findings: List[Finding]) -> float:
        """Combine individual weights sub-additively: 1 - prod(1 - w_i)."""
        if not findings:
            return 0.0

        p = 1.0
        for f in findings:
            p *= (1.0 - f.weight)
        score = 1.0 - p
        return min(max(score, 0.0), 1.0)

    def _score_to_level(self, score: float) -> str:
        """Map numeric score into descriptive risk level."""
        if score >= 0.70:
            return "CRITICAL"
        if score >= 0.40:
            return "HIGH"
        if score >= 0.20:
            return "MEDIUM"
        return "LOW"
