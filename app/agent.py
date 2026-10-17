"""Demo Summarizer Agent for PromptShield (PS3).

A deterministic, local, extractive text summarizer that demonstrates the
end-to-end firewall-protected agent workflow. This is NOT an LLM — it uses
word-frequency-based sentence scoring to produce a concise summary.

The agent receives content ONLY after the PromptShield firewall evaluates it:
- ALLOW:    Agent receives the original text.
- SANITIZE: Agent receives only the sanitized (redacted) text.
- BLOCK:    Agent is NEVER invoked. No summary is generated.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional


# Tracks whether the agent was actually invoked (used by tests)
_invocation_log: list[dict] = []


def reset_invocation_log() -> None:
    """Clear the invocation log (for testing)."""
    _invocation_log.clear()


def get_invocation_count() -> int:
    """Return the number of times the agent was invoked."""
    return len(_invocation_log)


@dataclass
class AgentResult:
    """Result of the demo agent's summarization."""
    invoked: bool
    agent_task: Optional[str]
    input_text: Optional[str]  # Exact text the agent received (null if blocked)
    summary: Optional[str]     # Generated summary (null if blocked)
    agent_note: str            # Human-readable explanation of what happened

    def to_dict(self) -> dict:
        return {
            "invoked": self.invoked,
            "agent_task": self.agent_task,
            "input_text": self.input_text,
            "summary": self.summary,
            "agent_note": self.agent_note,
        }


# Common English stop words to exclude from frequency scoring
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "so", "if",
    "this", "that", "it", "its", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "about", "above", "after", "again",
    "also", "am", "as", "because", "before", "between", "from", "here",
    "into", "out", "over", "own", "same", "then", "there", "through",
    "under", "until", "up", "while",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation rules."""
    # Split on sentence-ending punctuation followed by whitespace or end
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def summarize(text: str, agent_task: Optional[str] = None, max_sentences: int = 3) -> str:
    """Produce an extractive summary using word-frequency sentence scoring.

    This is a deterministic heuristic summarizer — not an LLM.
    The agent_task is used as a label in the output to indicate the
    summarization context.

    Args:
        text: The content to summarize (already firewall-approved).
        agent_task: The intended task description (used as instruction label).
        max_sentences: Maximum sentences to include in the summary.

    Returns:
        A string containing the extractive summary.
    """
    if not text or not text.strip():
        return "(No content to summarize.)"

    sentences = _split_sentences(text)
    if not sentences:
        return "(No content to summarize.)"

    # If the text is very short (1-2 sentences), return it directly
    if len(sentences) <= max_sentences:
        task_label = f"[Task: {agent_task}] " if agent_task else ""
        return f"{task_label}{text.strip()}"

    # Build word frequency table (excluding stop words)
    words = _tokenize(text)
    word_freq = Counter(w for w in words if w not in _STOP_WORDS)

    if not word_freq:
        # All stop words — just return the first few sentences
        task_label = f"[Task: {agent_task}] " if agent_task else ""
        return f"{task_label}{' '.join(sentences[:max_sentences])}"

    # Normalize frequencies
    max_freq = max(word_freq.values())
    normalized = {w: freq / max_freq for w, freq in word_freq.items()}

    # Score each sentence by sum of normalized word frequencies
    scored = []
    for i, sent in enumerate(sentences):
        tokens = _tokenize(sent)
        score = sum(normalized.get(t, 0.0) for t in tokens)
        # Slight position bias: earlier sentences get a small boost
        position_boost = 1.0 / (1.0 + i * 0.1)
        scored.append((i, sent, score * position_boost))

    # Select top sentences, preserving original order
    scored.sort(key=lambda x: x[2], reverse=True)
    top_indices = sorted([s[0] for s in scored[:max_sentences]])
    selected = [sentences[i] for i in top_indices]

    task_label = f"[Task: {agent_task}] " if agent_task else ""
    return f"{task_label}{' '.join(selected)}"


def invoke_agent(
    firewall_decision: str,
    forwarded_text: Optional[str],
    agent_task: Optional[str] = None,
) -> AgentResult:
    """Invoke the demo summarizer agent based on the firewall's decision.

    This function enforces the PromptShield contract:
    - BLOCK:    Agent is NOT invoked. Returns blocked result.
    - ALLOW:    Agent receives the original text (forwarded_text).
    - SANITIZE: Agent receives only the sanitized text (forwarded_text).

    Args:
        firewall_decision: The firewall verdict ("ALLOW", "SANITIZE", or "BLOCK").
        forwarded_text: The text forwarded by the firewall (None if blocked).
        agent_task: The intended agent task description.

    Returns:
        AgentResult with invocation status, input received, and summary.
    """
    decision = firewall_decision.upper()

    if decision == "BLOCK" or forwarded_text is None:
        # Agent is NOT invoked — firewall blocked the content
        result = AgentResult(
            invoked=False,
            agent_task=agent_task,
            input_text=None,
            summary=None,
            agent_note=(
                "Agent was NOT invoked. The PromptShield firewall blocked this "
                "content due to detected prompt-injection threats. No text was "
                "forwarded and no summary was generated."
            ),
        )
        # Do NOT log an invocation — the agent was never called
        return result

    # Agent IS invoked with the firewall-approved content
    summary = summarize(forwarded_text, agent_task=agent_task)

    result = AgentResult(
        invoked=True,
        agent_task=agent_task,
        input_text=forwarded_text,
        summary=summary,
        agent_note=(
            f"Agent invoked with firewall decision: {decision}. "
            f"{'Original text was forwarded.' if decision == 'ALLOW' else 'Only sanitized (redacted) text was forwarded — malicious spans were removed by the firewall.'}"
            " Summary was generated using a deterministic extractive summarizer (not an LLM)."
        ),
    )

    # Log the invocation for test verification
    _invocation_log.append({
        "decision": decision,
        "input_text": forwarded_text,
        "agent_task": agent_task,
    })

    return result
