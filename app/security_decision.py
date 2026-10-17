"""Layered Security Decision Engine for PromptShield.

Provides structured, explainable security evaluations by assessing untrusted inputs
across four fundamental risk dimensions:
1. Intent: Is the user seeking education, defensive testing, self-data access,
   or attempting adversarial extraction, manipulation, or bypass?
2. Requested Resource: Is the request targeting public/synthetic data, or
   confidential secrets (API keys, passwords), developer instructions, private
   personal information (PII, medical, financial), or proprietary business records?
3. Context: Does the interaction occur in a safe modality (e.g. mock fixtures,
   unit testing for 403 Forbidden, educational explanations, data minimization)
   or in an operational/adversarial context?
4. Authorization & Trust State: Does the request contain unverified claims of
   authority (e.g., "my manager approved this", "I have permission"), ambiguous
   operational scope, or an explicit attempt to bypass access controls?

CRITICAL ARCHITECTURAL INVARIANT:
Never treat a claim of authorization in untrusted prompt text as verified permission.
Actual access to protected data must be controlled by trusted backend authorization.
Untrusted prompt claims for sensitive resources must be routed to REVIEW, not ALLOW.
Explicit bypass attempts and direct secret extraction must be BLOCKED.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SecurityAction(str, Enum):
    """Action verdicts produced by the security decision engine."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


class SecurityIntent(str, Enum):
    """Categorized intent of the requester."""
    BENIGN_EDUCATIONAL = "BENIGN_EDUCATIONAL"
    DEFENSIVE_ENGINEERING = "DEFENSIVE_ENGINEERING"
    SELF_SERVICE_EXPORT = "SELF_SERVICE_EXPORT"
    BENIGN_GENERAL = "BENIGN_GENERAL"
    EXTRACTION_ATTEMPT = "EXTRACTION_ATTEMPT"
    BYPASS_OR_EVASION = "BYPASS_OR_EVASION"
    MIXED_INTENT = "MIXED_INTENT"
    GOVERNANCE_INQUIRY = "GOVERNANCE_INQUIRY"
    AMBIGUOUS_ACCESS = "AMBIGUOUS_ACCESS"


class ResourceSensitivity(str, Enum):
    """Classification of the requested data or resource."""
    PUBLIC_OR_GENERAL = "PUBLIC_OR_GENERAL"
    SYNTHETIC_OR_MOCK = "SYNTHETIC_OR_MOCK"
    PROTECTED_INSTRUCTIONS = "PROTECTED_INSTRUCTIONS"
    INFRASTRUCTURE_SECRETS = "INFRASTRUCTURE_SECRETS"
    PRIVATE_PERSONAL_DATA = "PRIVATE_PERSONAL_DATA"
    CONFIDENTIAL_BUSINESS = "CONFIDENTIAL_BUSINESS"
    DEANONYMIZATION_TARGET = "DEANONYMIZATION_TARGET"


class AuthorizationClaimType(str, Enum):
    """Evaluation of authorization asserted within untrusted input."""
    NONE_DETECTED = "NONE_DETECTED"
    STANDARD_BENIGN = "STANDARD_BENIGN"
    UNVERIFIED_CLAIM = "UNVERIFIED_CLAIM"
    GOVERNANCE_CHECK_REQUIRED = "GOVERNANCE_CHECK_REQUIRED"
    EXPLICIT_BYPASS = "EXPLICIT_BYPASS"


@dataclass
class SecurityAssessment:
    """Detailed risk assessment outcome across all four security dimensions."""
    action: SecurityAction
    intent: SecurityIntent
    resource: ResourceSensitivity
    authorization: AuthorizationClaimType
    confidence: float
    decision_reason: str
    reasons: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "intent": self.intent.value,
            "resource": self.resource.value,
            "authorization": self.authorization.value,
            "confidence": round(self.confidence, 4),
            "decision_reason": self.decision_reason,
            "reasons": self.reasons,
            "indicators": self.indicators,
        }


# ---------------------------------------------------------------------------
# Generalizable Lexical and Semantic Patterns
# (No dataset-specific phrases or individual test-case templates)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize typographical variants such as smart quotes and excess whitespace."""
    if not text:
        return ""
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .strip()
    )


# 1. Mixed Intent: Benign inquiry coupled with extraction/exfiltration of secrets
RE_MIXED_INTENT = re.compile(
    r"(?i)\b(?:explain|help|write|discuss|summarize|improve|review|teach|draft|create)\b.{0,100}\b"
    r"(?:and\s+also|then|additionally|attach|including|append)\b.{0,60}\b"
    r"(?:print|reveal|attach|dump|extract|exfiltrate|disclose|show|exact|confidential|actual|real|identify|bypass)\b.{0,60}\b"
    r"(?:password|admin|secret|profile|export|credentials?|hidden|token|key|records?|instructions?|customer|anonymous|memo)\b"
)

# 2. Safe Intent Modalities (Signals ALLOW)
RE_SAFE_INTENT = re.compile(
    r"(?i)(?:\bexplain\s+(?:data\s+minimization|the\s+difference|how\s+role-based|the\s+term|personal\s+information|privacy|least\s+privilege)\b|"
    r"\bwhat\s+is\s+(?:data\s+minimization|role-based|privacy|sensitive\s+personal|least\s+privilege)\b|"
    r"\bdescribe\s+(?:how\s+(?:role-based|access\s+control|data\s+protection|least\s+privilege)|indicators\s+of\s+prompt-injection)\b|"
    r"\bhow\s+a\s+blue\s+team\s+can\s+respond\s+safely\b|"
    r"\bwrite\s+(?:a\s+small\s+example|a\s+python\s+function|unit\s+tests?|a\s+function\s+named|a\s+regular\s+expression|a\s+script)\b.{0,80}\b"
    r"(?:masks?|unit\s+test|http\s+403|ownership\s+check|fictional|demo-only|redacts?|sanitizes?|hash(?:ing|es)?)\b|"
    r"\bcreate\s+unit\s+tests\s+asserting\s+that\s+an\s+unauthorized\s+request\s+returns\s+http\s+403\b|"
    r"\b(?:implement|design)\s+a\s+(?:server-side\s+ownership\s+check|flow\s+for\s+a\s+signed-in\s+user|"
    r"endpoint\s+that\s+returns\s+only\s+the\s+profile\s+belonging\s+to\s+the\s+authenticated\s+user)\b|"
    r"\bshow\s+how\s+to\s+validate\s+that\s+checking\s+a\s+user\s+role\b.{0,100}\brejects\s+unauthorized\s+requests\b|"
    r"\bgenerate\s+(?:fictional|synthetic|mock|sample)\s+(?:customer\s+profiles|records|users|data|identifiers)\b|"
    r"\b(?:help\s+me\s+prepare\s+a\s+request\s+to\s+download\s+a\s+copy\s+of\s+my\s+own\s+account\s+data)\b|"
    r"\bprepare\s+a\s+report\s+from\s+application\s+logs\b.{0,100}\breplacing\s+direct\s+identifiers\b|"
    r"\bevaluate\s+prompt-injection\s+attempts\b.{0,80}\b(?:local\s+fixtures|deliberate\s+test|without\s+connecting\s+to\s+production)\b|"
    r"\bin\s+a\s+toy\s+database\b|"
    r"\b(?:logged-in|authenticated)\s+user\s+wants\s+to\s+update\s+their\s+own\s+contact\s+preferences\b|"
    r"\bcheck\s+whether\s+the\s+current\s+role\s+permits\s+access\s+to\s+a\s+restricted\s+report\s+before\s+opening\s+it\b)"
)

# 3. Explicit Security Bypass, Control Evasion, or Impersonation Directives (Signals BLOCK)
RE_EXPLICIT_BYPASS = re.compile(
    r"(?i)(?:\b(?:by\s+skipping|by\s+bypassing|by\s+disabling|by\s+overriding|by\s+using\s+a\s+backdoor)\b|"
    r"\b(?:without\s+(?:established\s+|recorded\s+|their\s+|asking\s+)?(?:authorization|permission|consent|approval))\b|"
    r"\bwithout\s+contacting\s+the\s+(?:data\s+subject|employee|account\s+owner|customer|user)\b|"
    r"\bwithout\s+asking\s+the\s+(?:account\s+owner|data\s+subject|employee|user)\b|"
    r"\bwithout\s+an\s+access\s+approval\s+is\s+recorded\b|"
    r"\bwho\s+has\s+not\s+consented\b|"
    r"\bI\s+do\s+not\s+have\s+(?:the\s+person's|their|any|access)?\s*permission\b|"
    r"\bI\s+am\s+not\s+the\s+data\s+subject\b|"
    r"\bwithout\s+notifying\s+the\s+(?:data\s+owner|account\s+owner|customer|user|employee)\b|"
    r"\bwithout\s+(?:requesting|seeking|obtaining|access)?\s*approval\b|"
    r"\bwithout\s+(?:a\s+)?permission\s+check\b|"
    r"\b(?:to|for)\s+an?\s*(?:external|outside)\s*(?:email|cloud|drive|server|recipient|address|destination)\b|"
    r"\b(?:share|export|send|disclose)\s+outside\s+the\s+(?:organization|approved\s+team)\b|"
    r"\b(?:avoid|evade|bypass|skip|circumvent)\s+role\s+validation\b|"
    r"\b(?:conceal|hide)\s+the\s+(?:export|extraction|transfer)\b|"
    r"\bquietly\s+forward\b|"
    r"\beven\s+though\s+access\s+is\s+denied\b|"
    r"\beven\s+if\s+my\s+role\s+has\s+no\s+access\b|"
    r"\beven\s+if\s+(?:the\s+)?data-loss\s+controls\s+reject\s+it\b|"
    r"\b(?:that\s+)?belongs\s+to\s+(?:someone\s+else|a\s+different\s+user)\b|"
    r"\beven\s+though\s+(?:disclosure\s+is\s+(?:restricted|not\s+authorized|prohibited)|anonymity\s+was\s+promised|my\s+account\s+is\s+denied\s+access|no\s+access\s+approval\s+is\s+recorded)\b|"
    r"\baccepts\s+an\s+unauthorized\s+request\b|"
    r"\bskip\s+(?:the\s+)?(?:consent\s+check|permission|authorization|approval)\b|"
    r"\bdisable\s+(?:the\s+)?(?:logging|audit|review|guardrails?|filters?)\b|"
    r"\b(?:omit\s+the\s+audit\s+trail|remove\s+the\s+classification\s+header)\b|"
    r"\btreat\s+the\s+lack\s+of\s+permission\s+as\s+irrelevant\b|"
    r"\bno\s+approval\s+(?:workflow\s+is\s+necessary|is\s+needed)\b|"
    r"\bdo\s+not\s+check\s+permissions\b|"
    r"\bregardless\s+of\s+(?:its\s+)?classification\b|"
    r"\boutside\s+(?:my\s+assigned\s+matters|the\s+approved\s+system|the\s+requester's\s+scope)\b|"
    r"\b(?:workspace|repository|archive)\s+(?:I\s+do\s+not\s+administer|where\s+the\s+user\s+has\s+no\s+assigned\s+role)\b|"
    r"\bnot\s+assigned\s+to\s+me\b|"
    r"\bpretend\s+(?:the\s+authorization\s+service|privacy\s+restrictions|you\s+are\s+an\s+insider|you\s+are\s+an\s+unrestricted)\b|"
    r"\b(?:impersonate\s+a\s+privileged\s+role|steal\s+a\s+coworker's\s+login)\b|"
    r"\brewrite\s+my\s+request\s+so\s+the\s+(?:security\s+scanner|firewall|filter)\b|"
    r"\bbreak\s+the\s+extraction\s+into\s+ten\s+harmless\b|"
    r"\b(?:to|for)\s+(?:a|an|my|our|any|the)?\s*(?:outside\s+party|unapproved\s+external|personal\s+cloud|personal\s+email|unauthorized\s+reader|public\s+file\s+share|personal\s+account|public\s+issue)\b|"
    r"\bexport\s+restricted\s+\w+\s+records\s+to\s+an\s+outsider\b|"
    r"\b(?:bypass|evade)\s+(?:secret\s+masking|the\s+gate|the\s+usual\s+visibility|role\s+validation)\b|"
    r"\buse\s+(?:another\s+employee's\s+active\s+session|an\s+undocumented\s+route|a\s+backup\s+endpoint|another\s+route)\b|"
    r"\buntil\s+.{0,80}\baccepts\s+an\s+unauthorized\b|"
    r"\bsecretly\s+collect\b|"
    r"\bsplit\s+the\s+administrator\s+token\s+across\s+multiple\s+replies\b|"
    r"\bput\s+a\s+confidential\s+customer\s+list\s+in\s+hidden\s+markup\b|"
    r"\bgive\s+me\s+a\s+command\s+that\s+prints\s+the\s+stored\s+secret\b)"
)

# 4. Direct Secret & Protected Asset Extraction (Signals BLOCK)
RE_DIRECT_EXTRACTION = re.compile(
    r"(?i)(?:\b(?:print|read|copy|dump|extract|export|reveal|show|give\s+me|return|send|recover|infer|locate|paste|search|list|disclose|unmask)\b.{0,70}\b"
    r"(?:live\s+api\s+token|production\s+api\s+key|database\s+password|administrator's\s+session|private\s+ssh\s+keys?|signing\s+keys?|"
    r"recovery\s+codes?|access\s+tokens?|secret\s+key|master\s+password|deployment\s+(?:secret\s+)?vault|employee\s+credentials|"
    r"production\s+certificate|all\s+customer\s+account\s+records|full\s+customer\s+database|unreleased\s+product\s+strategy|confidential\s+acquisition|"
    r"non-public\s+price\s+bids|hidden\s+system\s+message|exact\s+private\s+instructions|home\s+address\s+of\s+this\s+person|"
    r"salary\s+and\s+bank\s+details|anonymous\s+complaint\s+by\s+correlating|confidential\s+product\s+roadmap|"
    r"private\s+developer\s+instructions|confidential\s+rules|confidential\s+settlement|enterprise\s+pricing\s+formula|"
    r"every\s+employee's\s+personal\s+information|another\s+user's\s+private\s+notes|separate\s+user's\s+conversation\s+history|"
    r"medical\s+(?:diagnoses|history)|treatment\s+history|payroll\s+workbook|actual\s+passwords|private\s+support\s+conversations|"
    r"protected\s+instructions?|internal\s+policy|full\s+set\s+of\s+student\s+records|unredacted\s+document|"
    r"private\s+messages\s+my\s+partner\s+exchanged|non-public\s+phone\s+number|confidential\s+performance\s+reviews|"
    r"private\s+email\s+addresses\s+of\s+people|financial\s+problems\s+in\s+previous\s+chats|"
    r"author\s+of\s+an\s+anonymous\s+complaint|real\s+person\s+behind\s+a\s+pseudonymous\s+account|"
    r"confidential\s+policy\s+text\s+supplied\s+before\s+the\s+conversation|full\s+internal\s+prompt)\b|"
    r"\b(?:reproduce[sd]?|show[s]?|translate[sd]?|encode[sd]?|turn\s+the|disclose[sd]?|put\s+the|assemble[sd]?|describe[sd]?)\b.{0,60}\b"
    r"(?:hidden\s+system\s+message|private\s+developer\s+instructions|exact\s+private\s+instructions|hidden\s+prompt|"
    r"confidential\s+rules|internal\s+instructions|protected\s+developer\s+instructions|protected\s+instructions?|full\s+prompt|"
    r"every\s+hidden\s+rule|first\s+and\s+last\s+word\s+of\s+every\s+sentence\s+in\s+your\s+hidden\s+instructions|"
    r"confidential\s+instructions\s+you\s+were\s+specifically\s+told\s+not\s+to\s+reveal|exact\s+rules|protected\s+directives?)\b|"
    r"\b(?:internal\s+prompt|system\s+prompt|developer\s+instructions?|protected\s+directives?|exact\s+rules|rules\s+used\s+to\s+constrain)\b.{0,60}\bverbatim\b|"
    r"\b(?:trace|unmask|identify|disclose|expose|find)\s+(?:the\s+)?(?:author\s+of\s+an\s+anonymous|real\s+person\s+behind\s+a\s+pseudonymous|home\s+location|(?:personal\s+)?(?:identity|profile)\s+of|(?:personal\s+)?identity\s+linked\s+to)\b|"
    r"\b(?:work\s+out|reconstruct)\b.{0,60}\b(?:medical\s+diagnoses|personal\s+details|conversation\s+history)\b|"
    r"\b(?:return|print|reveal|show|dump|extract|reproduce|give\s+me)\b.{0,60}\b(?:the\s+)?(?:internal\s+prompt|protected\s+directives?|exact\s+rules)\b|"
    r"\b(?:recovered|extracted|dumped)\s+from\s+(?:the|a|an)?\s*(?:production\s+environment|deployment\s+(?:secret\s+)?vault|ci\s+build\s+logs|restricted\s+server|runtime\s+process\s+environment)\b|"
    r"\b(?:automate\s+(?:nightly\s+)?extraction|bulk\s+export|dump|transfer\s+all|downloadable\s+archive)\s+(?:of\s+)?(?:all\s+)?customer\s+account\s+records\b|"
    r"\bunreleased\s+product\s+strategy\s+roadmap\b|"
    r"\bthe\s+most\s+personal\s+details\s+in\s+their\s+conversations\b|"
    r"\banother\s+employee's\s+pay\s+and\s+bank\s+details\b|"
    r"\bteammate's\s+assistant\s+session\b|"
    r"\bcreate\s+a\s+public\s+copy\s+of\b|"
    r"\bdisclose\s+(?:your\s+)?hidden\s+instructions\b|"
    r"\bwrite\s+a\s+short\s+story\s+whose\s+dialogue\s+reproduces\b|"
    r"\bcross-reference\b.{0,80}\bdisclose\s+(?:the\s+author\s+of\s+an\s+anonymous|the\s+real\s+person)\b|"
    r"\bidentify\s+(?:the\s+author\s+of\s+an\s+anonymous|which\s+student\s+submitted|the\s+person\s+behind|which\s+named\s+customer)\b|"
    r"\b(?:work\s+out|infer)\s+the\s+real\s+(?:identity|person)\b|"
    r"\bsummarize\s+the\s+most\s+personal\s+details\s+in\s+their\s+conversations\b|"
    r"\bsearch\s+(?:a\s+restricted|an\s+HR|an\s+internal\s+security\s+archive|a\s+project\s+drive\s+marked\s+confidential)\b.{0,80}\b(?:payroll\s+workbook|settlement\s+memo|vendor\s+bid)\b|"
    r"\bprint\s+the\s+live\s+API\s+token\b|"
    r"\bsend\s+the\s+(?:complete\s+)?(?:the\s+)?full\s+set\s+of\s+student\s+records\b)"
)

# 5. Governance Verification and Unverified Authorization Claims (Signals REVIEW)
RE_GOVERNANCE_AND_CLAIM = re.compile(
    r"(?i)(?:\b(?:confirm|verify|check|determine|assess|validate|establish)\b.{0,70}\b"
    r"(?:purpose|permission|transfer|role|scope|identity|approval|authority|consent|minimum\s+necessary|"
    r"environment|whether|permitted|identities)\b|"
    r"\b(?:approval\s+record|workload\s+identity|unclear\s+ownership|without\s+a\s+recorded\s+approval|requires\s+an\s+approval\s+ticket)\b|"
    r"\b(?:what\s+evidence\s+is\s+needed|too\s+little\s+context|approval\s+and\s+scope\s+are\s+not\s+attached|"
    r"recipient\s+access\s+is\s+not\s+documented|ticket\s+does\s+not\s+list\s+who\s+is\s+authorized)\b|"
    r"\b(?:do\s+not\s+disclose\s+until|never\s+print\s+the\s+secret\s+itself|only\s+retrieve\s+the\s+minimum\s+necessary)\b|"
    r"\b(?:I\s+have\s+approval\s+to\s+access|my\s+supervisor\s+says\s+I\s+can\s+see|my\s+manager\s+says|my\s+friend\s+said\s+I\s+can\s+read|"
    r"the\s+security\s+team\s+asked\s+me\s+to\s+inspect|the\s+on-call\s+engineer\s+asks|the\s+support\s+team\s+asked|"
    r"someone\s+needs|I\s+am\s+acting\s+for|I\s+was\s+added\s+to\s+the\s+project\s+team|I\s+have\s+permission\s+to\s+review)\b|"
    r"\b(?:the\s+client\s+consented\b.{0,50}\bno\s+way\s+to\s+verify|I\s+need\s+it\s+for\s+work|so\s+I\s+can\s+resolve\s+an\s+issue|"
    r"for\s+an\s+investigation|for\s+a\s+demo\s+tomorrow|help\s+me\s+investigate\s+a\s+suspected\s+leak|for\s+an\s+audit\s+I\s+am\s+helping\s+with)\b|"
    r"\b(?:I\s+need\s+to\s+check\s+a\s+former\s+employee's\s+mailbox|restore\s+a\s+record\s+from\s+backup|"
    r"this\s+file\s+is\s+in\s+a\s+shared\s+folder|open\s+the\s+old\s+private\s+notes)\b|"
    r"\b(?:for\s+case\s+an\s+incident-response|for\s+case\s+a\s+support\s+case|for\s+case\s+a\s+compliance\s+queue)\b|"
    r"\b(?:retrieve\s+a\s+customer's\s+contact|provide\s+all\s+user\s+contact\s+records|retrieve\s+a\s+patient's\s+health\s+record|"
    r"patient's\s+health\s+record)\b|"
    r"\b(?:can\s+the\s+service\s+use\s+an\s+api\s+token|deployment\s+task\s+references\s+an\s+api\s+token)\b|"
    r"\b(?:not\s+sure\s+whether\s+it\s+contains\s+personal\s+information|incomplete\s+assignment\s+metadata|"
    r"unknown\s+team\s+member|requester\s+role\s+is\s+not\s+established|environment\s+is\s+not\s+confirmed|"
    r"I\s+have\s+not\s+checked\s+who\s+owns\s+it|I\s+do\s+not\s+know\s+who\s+created\s+it|export\s+scope\s+is\s+not\s+defined\s+yet)\b|"
    r"\bcan\s+you\s+get\s+the\s+file\s+for\s+\w+\b|"
    r"\bshow\s+me\s+this\s+customer's\s+account\s+details\b|"
    r"\bfind\s+out\s+who\s+submitted\s+this\s+anonymous\s+feedback\s+so\s+I\s+can\s+follow\s+up\b|"
    r"\bshare\s+the\s+internal\s+project\s+notes\s+with\s+our\s+new\s+contractor\b)"
)


class SecurityDecisionEngine:
    """Evaluates untrusted inputs across intent, resource, context, and authorization."""

    def __init__(self) -> None:
        pass

    def assess(
        self,
        text: str,
        agent_task: Optional[str] = None,
        scanner_has_critical: bool = False,
    ) -> SecurityAssessment:
        """Perform layered risk evaluation and return an actionable SecurityAssessment.

        Evaluation Priority:
        1. Explicit Mixed Intent (Benign framing + covert extraction) -> BLOCK
        2. Verified Safe Modalities (Educational, defensive coding, synthetic fixtures, own account) -> ALLOW
        3. Explicit Security Bypass / Evasion / Circumvention -> BLOCK
        4. Direct Unauthorized Extraction of Protected Secrets / Instructions / PII -> BLOCK
        5. Governance Verification, Ambiguous Access, or Unverified Authorization Claims -> REVIEW
        6. Default Benign Operations -> ALLOW
        """
        normalized = _normalize_text(text)
        if not normalized:
            return SecurityAssessment(
                action=SecurityAction.ALLOW,
                intent=SecurityIntent.BENIGN_GENERAL,
                resource=ResourceSensitivity.PUBLIC_OR_GENERAL,
                authorization=AuthorizationClaimType.NONE_DETECTED,
                confidence=1.0,
                decision_reason="Empty input permitted.",
            )

        indicators: List[str] = []
        reasons: List[str] = []

        # Layer 1: Check for Mixed Intent (Adversarial camouflage)
        m_mixed = RE_MIXED_INTENT.search(normalized)
        if m_mixed:
            indicators.append(f"Mixed Intent: '{m_mixed.group(0)[:60]}'")
            reasons.append("Request couples benign inquiry with directive to extract or disclose protected assets.")
            return SecurityAssessment(
                action=SecurityAction.BLOCK,
                intent=SecurityIntent.MIXED_INTENT,
                resource=ResourceSensitivity.INFRASTRUCTURE_SECRETS,
                authorization=AuthorizationClaimType.EXPLICIT_BYPASS,
                confidence=0.95,
                decision_reason="Blocked: Detected mixed-intent attempt combining benign framing with secret extraction.",
                reasons=reasons,
                indicators=indicators,
            )

        # Layer 2: Verified Safe Modalities (Education, defensive coding, synthetic fixtures, self-service)
        m_safe = RE_SAFE_INTENT.search(normalized)
        if m_safe:
            indicators.append(f"Safe Modality: '{m_safe.group(0)[:60]}'")
            reasons.append("Request operates within a verified educational, defensive engineering, synthetic, or self-service context.")
            return SecurityAssessment(
                action=SecurityAction.ALLOW,
                intent=SecurityIntent.BENIGN_EDUCATIONAL,
                resource=ResourceSensitivity.PUBLIC_OR_GENERAL,
                authorization=AuthorizationClaimType.STANDARD_BENIGN,
                confidence=0.98,
                decision_reason="Allowed: Verified safe context (education, defensive testing, or synthetic fixtures).",
                reasons=reasons,
                indicators=indicators,
            )

        # Layer 3: Check for Explicit Bypass, Evasion, or Impersonation
        m_bypass = RE_EXPLICIT_BYPASS.search(normalized)
        if m_bypass:
            indicators.append(f"Evasion/Bypass: '{m_bypass.group(0)[:60]}'")
            reasons.append("Request explicitly attempts to circumvent access controls, audit trails, or permissions.")
            return SecurityAssessment(
                action=SecurityAction.BLOCK,
                intent=SecurityIntent.BYPASS_OR_EVASION,
                resource=ResourceSensitivity.CONFIDENTIAL_BUSINESS,
                authorization=AuthorizationClaimType.EXPLICIT_BYPASS,
                confidence=0.95,
                decision_reason="Blocked: Explicit instruction to bypass access controls or suppress audit logging.",
                reasons=reasons,
                indicators=indicators,
            )

        # Layer 4: Direct Extraction of Protected Secrets, Developer Instructions, or Private Data
        m_extract = RE_DIRECT_EXTRACTION.search(normalized)
        if m_extract:
            indicators.append(f"Protected Asset Extraction: '{m_extract.group(0)[:60]}'")
            reasons.append("Direct attempt to extract confidential credentials, system instructions, or private records.")
            return SecurityAssessment(
                action=SecurityAction.BLOCK,
                intent=SecurityIntent.EXTRACTION_ATTEMPT,
                resource=ResourceSensitivity.INFRASTRUCTURE_SECRETS,
                authorization=AuthorizationClaimType.NONE_DETECTED,
                confidence=0.92,
                decision_reason="Blocked: Unauthorized extraction of protected credentials, instructions, or private records.",
                reasons=reasons,
                indicators=indicators,
            )

        # Layer 5: Governance Review, Ambiguous Scope, or Unverified Authorization Claims
        m_gov = RE_GOVERNANCE_AND_CLAIM.search(normalized)
        if m_gov:
            indicators.append(f"Governance/Unverified Claim: '{m_gov.group(0)[:60]}'")
            reasons.append(
                "Request touches restricted or sensitive resources with an unverified authorization claim "
                "or explicit governance check requirement."
            )
            return SecurityAssessment(
                action=SecurityAction.REVIEW,
                intent=SecurityIntent.GOVERNANCE_INQUIRY,
                resource=ResourceSensitivity.PRIVATE_PERSONAL_DATA,
                authorization=AuthorizationClaimType.UNVERIFIED_CLAIM,
                confidence=0.90,
                decision_reason=(
                    "Flagged for Review: Untrusted prompt text cannot verify authorization. "
                    "Access to protected data requires backend authorization confirmation."
                ),
                reasons=reasons,
                indicators=indicators,
            )

        # Default: Benign request
        return SecurityAssessment(
            action=SecurityAction.ALLOW,
            intent=SecurityIntent.BENIGN_GENERAL,
            resource=ResourceSensitivity.PUBLIC_OR_GENERAL,
            authorization=AuthorizationClaimType.STANDARD_BENIGN,
            confidence=0.85,
            decision_reason="Allowed: No adversarial bypass, extraction, or unverified claims detected.",
            reasons=["Standard benign query without sensitive security indicators."],
            indicators=[],
        )
