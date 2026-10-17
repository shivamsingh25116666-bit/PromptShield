"""Policy Enforcement Engine for PromptShield.

Implements configurable policy evaluation supporting:
- ALLOW: Forward original text.
- SANITIZE: Redact malicious instruction spans and forward sanitized text.
- BLOCK: Completely drop the payload. Forwarded text is GUARANTEED to be None.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional

from app.scanner import Finding, ScanResult
from app.security_decision import (
    AuthorizationClaimType,
    ResourceSensitivity,
    SecurityAction,
    SecurityAssessment,
    SecurityDecisionEngine,
    SecurityIntent,
)


class PolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


@dataclass
class PolicyEvaluation:
    decision: PolicyDecision
    policy_mode: str
    risk_score: float
    risk_level: str
    findings: List[Finding]
    original_text: str
    sanitized_text: str
    forwarded_text: Optional[str]  # None if blocked or held for review
    decision_reason: str
    security_assessment: Optional[SecurityAssessment] = None

    def to_dict(self) -> dict:
        data = {
            "decision": self.decision.value,
            "policy_mode": self.policy_mode,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "findings": [f.to_dict() for f in self.findings],
            "findings_count": len(self.findings),
            "sanitized_text": self.sanitized_text,
            "forwarded_text": self.forwarded_text,
            "decision_reason": self.decision_reason,
        }
        if self.security_assessment is not None:
            data["security_assessment"] = self.security_assessment.to_dict()
        return data


class PolicyEngine:
    """Evaluates scan results against security policies and produces actionable verdicts."""

    def __init__(
        self,
        default_mode: str = "auto",
        sanitize_threshold: float = 0.20,
        block_threshold: float = 0.60,
    ):
        self.default_mode = default_mode.lower()
        self.sanitize_threshold = sanitize_threshold
        self.block_threshold = block_threshold
        self.security_engine = SecurityDecisionEngine()

    def evaluate(
        self,
        scan_result: ScanResult,
        policy_mode: Optional[str] = None,
        custom_block_threshold: Optional[float] = None,
    ) -> PolicyEvaluation:
        """Evaluate a scan result against the chosen policy mode."""
        mode = (policy_mode or self.default_mode).lower()
        block_thresh = custom_block_threshold if custom_block_threshold is not None else self.block_threshold

        # Produce sanitized text by excising or marking malicious spans
        sanitized_text = self._sanitize_text(scan_result.text, scan_result.findings)

        # Evaluate layered security decision across intent, resource, context, and authorization
        sec_assessment = self.security_engine.assess(
            text=scan_result.text,
            agent_task=scan_result.agent_task,
        )

        decision: PolicyDecision
        decision_reason: str
        forwarded_text: Optional[str]

        # Explicit mode overrides
        if mode == "block":
            has_critical = any(f.severity == "CRITICAL" for f in scan_result.findings)
            meets_threshold = bool(scan_result.findings) and (scan_result.risk_score >= block_thresh)
            is_sec_block = sec_assessment.action == SecurityAction.BLOCK
            is_sec_review = sec_assessment.action == SecurityAction.REVIEW

            if meets_threshold or has_critical or is_sec_block or is_sec_review:
                decision = PolicyDecision.BLOCK
                reasons = []
                if meets_threshold:
                    reasons.append(f"Risk score {scan_result.risk_score:.2f} meets/exceeds block threshold {block_thresh:.2f}")
                if has_critical:
                    reasons.append("Contains critical-severity prompt injection or jailbreak indicators")
                if is_sec_block or is_sec_review:
                    reasons.append(sec_assessment.decision_reason)
                decision_reason = f"Blocked: {'; '.join(reasons)}."
                forwarded_text = None
            elif not scan_result.findings and sec_assessment.action == SecurityAction.ALLOW:
                decision = PolicyDecision.ALLOW
                decision_reason = "Allowed: No prompt injection or security threats detected."
                forwarded_text = scan_result.text
                sanitized_text = scan_result.text
            else:
                decision = PolicyDecision.ALLOW
                decision_reason = (
                    f"Allowed: Risk score {scan_result.risk_score:.2f} is below block threshold {block_thresh:.2f}."
                )
                forwarded_text = scan_result.text
                sanitized_text = scan_result.text

        elif mode == "sanitize":
            if scan_result.findings:
                decision = PolicyDecision.SANITIZE
                decision_reason = "Policy mode set to SANITIZE. Malicious instruction spans were redacted."
                forwarded_text = sanitized_text
            elif sec_assessment.action == SecurityAction.BLOCK:
                decision = PolicyDecision.BLOCK
                decision_reason = sec_assessment.decision_reason
                forwarded_text = None
            elif sec_assessment.action == SecurityAction.REVIEW:
                decision = PolicyDecision.REVIEW
                decision_reason = sec_assessment.decision_reason
                forwarded_text = None
            else:
                decision = PolicyDecision.ALLOW
                decision_reason = "Allowed: No prompt injection patterns detected."
                forwarded_text = scan_result.text
                sanitized_text = scan_result.text

        elif mode == "allow":
            decision = PolicyDecision.ALLOW
            decision_reason = "Policy mode explicitly set to ALLOW (bypassed inspection)."
            forwarded_text = scan_result.text

        else:
            # "auto" adaptive mode
            has_critical = any(f.severity == "CRITICAL" for f in scan_result.findings)
            has_exfiltration = any(f.category == "Data Exfiltration" for f in scan_result.findings)
            is_sec_block = sec_assessment.action == SecurityAction.BLOCK
            is_sec_review = sec_assessment.action == SecurityAction.REVIEW

            if scan_result.risk_score >= block_thresh or has_critical or has_exfiltration or is_sec_block:
                decision = PolicyDecision.BLOCK
                reasons = []
                if scan_result.risk_score >= block_thresh:
                    reasons.append(f"Risk score {scan_result.risk_score:.2f} meets/exceeds block threshold {block_thresh:.2f}")
                if has_critical:
                    reasons.append("Contains critical-severity prompt injection or jailbreak indicators")
                if has_exfiltration:
                    reasons.append("Identified unauthorized data exfiltration instructions")
                if is_sec_block:
                    reasons.append(sec_assessment.decision_reason)
                decision_reason = f"Blocked: {'; '.join(reasons)}."
                forwarded_text = None

            elif is_sec_review:
                decision = PolicyDecision.REVIEW
                decision_reason = sec_assessment.decision_reason
                forwarded_text = None

            elif scan_result.risk_score >= self.sanitize_threshold and scan_result.findings:
                # Malicious snippets detected but manageable without fully dropping the document
                decision = PolicyDecision.SANITIZE
                decision_reason = (
                    f"Sanitized: Flagged {len(scan_result.findings)} suspicious snippet(s) "
                    f"(risk score: {scan_result.risk_score:.2f}). Malicious instructions stripped."
                )
                forwarded_text = sanitized_text

            else:
                decision = PolicyDecision.ALLOW
                decision_reason = (
                    "Allowed: No prompt injection patterns detected or score is below policy threshold."
                )
                forwarded_text = scan_result.text

        # Invariant check: Blocked or Reviewed text MUST NEVER be forwarded
        if decision in (PolicyDecision.BLOCK, PolicyDecision.REVIEW):
            forwarded_text = None

        return PolicyEvaluation(
            decision=decision,
            policy_mode=mode,
            risk_score=scan_result.risk_score,
            risk_level=scan_result.risk_level,
            findings=scan_result.findings,
            original_text=scan_result.text,
            sanitized_text=sanitized_text,
            forwarded_text=forwarded_text,
            decision_reason=decision_reason,
            security_assessment=sec_assessment,
        )

    def _sanitize_text(self, text: str, findings: List[Finding]) -> str:
        """Redact flagged character spans in reverse order to preserve indexing."""
        if not text or not findings:
            return text

        # 1. Expand finding spans if followed by conjoined attack clauses within the same sentence
        expanded_intervals = []
        for f in findings:
            start = f.start
            end = f.end
            rule_id = f.rule_id

            rest = text[end:]
            m_boundary = re.search(r"[\r\n]|(?:\.(?:\s+|$))|[!?]", rest)
            boundary_idx = (end + m_boundary.start()) if m_boundary else len(text)
            tail = text[end:boundary_idx]
            if re.match(r"^\s*(?:and|,|\bthen\b|\binstead\b|\bto\b|\bnow\b|\:)\s+", tail, re.IGNORECASE):
                end = boundary_idx
            expanded_intervals.append([start, end, rule_id])

        # 2. Sort intervals by start ascending
        sorted_intervals = sorted(expanded_intervals, key=lambda x: x[0])

        # 3. Merge overlapping intervals or intervals separated only by connective syntax/conjunctions
        merged_intervals = []
        for interval in sorted_intervals:
            if not merged_intervals:
                merged_intervals.append(interval)
            else:
                prev = merged_intervals[-1]
                if interval[0] <= prev[1]:
                    prev[1] = max(prev[1], interval[1])
                    prev[2] = f"{prev[2]}+{interval[2]}"
                else:
                    gap = text[prev[1]:interval[0]]
                    if re.fullmatch(r"[\s,;:\-–—\(\)\[\]]*(?:and|then|also|to|or|plus|&)?[\s,;:\-–—\(\)\[\]]*", gap, re.IGNORECASE):
                        prev[1] = max(prev[1], interval[1])
                        prev[2] = f"{prev[2]}+{interval[2]}"
                    else:
                        merged_intervals.append(interval)

        # 4. Apply redactions in reverse order
        result = list(text)
        for start, end, rule_label in reversed(merged_intervals):
            replacement = f" [REDACTED_INJECTION: {rule_label}] "
            result[start:end] = list(replacement)

        # 5. Clean up excess spaces and spacing before punctuation
        cleaned = "".join(result)
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        cleaned = "\n".join(" ".join(line.split()) for line in cleaned.splitlines())
        return cleaned.strip()
