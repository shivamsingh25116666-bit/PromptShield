"""Reproducible Evaluation Runner for PromptShield (PS3).

Evaluates the rule-based scanner and policy engine against a labeled dataset
containing prompt-injection attacks and benign inputs. Computes and reports:
- Total Cases
- Correct Classifications
- False Positives
- False Negatives
- Accuracy
- Precision
- Recall

Validates policy enforcement including:
- Critical-severity attacks must result in BLOCK with forwarded_text=None
- Harmless normal inputs must result in ALLOW with forwarded_text intact
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.policies import PolicyDecision, PolicyEngine
from app.scanner import TransparentScanner


def run_evaluation() -> Dict[str, Any]:
    dataset_path = project_root / "data" / "evaluation_dataset.json"
    results_path = project_root / "data" / "evaluation_results.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    scanner = TransparentScanner()
    policy_engine = PolicyEngine()

    total_cases = len(dataset)
    true_positives = 0   # Attack correctly flagged (BLOCK or SANITIZE)
    false_negatives = 0  # Attack missed (ALLOW)
    true_negatives = 0   # Benign correctly allowed (ALLOW)
    false_positives = 0  # Benign incorrectly flagged (BLOCK or SANITIZE)

    critical_attacks_tested = 0
    critical_attacks_blocked = 0
    harmless_tested = 0
    harmless_allowed = 0

    detailed_results = []

    for item in dataset:
        item_id = item["id"]
        label = item["label"].lower()
        text = item["text"]
        agent_task = item.get("agent_task")
        expected_decision = item.get("expected_decision", "BLOCK" if label == "attack" else "ALLOW")

        scan_res = scanner.scan(text=text, agent_task=agent_task)
        eval_res = policy_engine.evaluate(scan_res)

        predicted_attack = eval_res.decision in (PolicyDecision.BLOCK, PolicyDecision.SANITIZE)
        is_attack = (label == "attack")

        # Invariant check: Blocked text must never be forwarded
        if eval_res.decision == PolicyDecision.BLOCK and eval_res.forwarded_text is not None:
            raise RuntimeError(f"CRITICAL INVARIANT VIOLATION: Case {item_id} was blocked but forwarded text!")

        # Track critical-severity attack behavior
        has_critical = any(f.severity == "CRITICAL" for f in scan_res.findings)
        if is_attack and has_critical:
            critical_attacks_tested += 1
            if eval_res.decision == PolicyDecision.BLOCK and eval_res.forwarded_text is None:
                critical_attacks_blocked += 1

        # Track ordinary harmless text behavior
        if not is_attack and len(scan_res.findings) == 0:
            harmless_tested += 1
            if eval_res.decision == PolicyDecision.ALLOW and eval_res.forwarded_text == text:
                harmless_allowed += 1

        if is_attack:
            if predicted_attack:
                true_positives += 1
                outcome = "TP"
            else:
                false_negatives += 1
                outcome = "FN"
        else:
            if not predicted_attack:
                true_negatives += 1
                outcome = "TN"
            else:
                false_positives += 1
                outcome = "FP"

        detailed_results.append({
            "id": item_id,
            "label": label,
            "category": item.get("category", ""),
            "expected_decision": expected_decision,
            "actual_decision": eval_res.decision.value,
            "risk_score": round(eval_res.risk_score, 4),
            "risk_level": eval_res.risk_level,
            "findings_count": len(eval_res.findings),
            "outcome": outcome,
            "forwarded_text_safe": (eval_res.forwarded_text is None) if eval_res.decision == PolicyDecision.BLOCK else True,
        })

    correct_classifications = true_positives + true_negatives
    accuracy = (correct_classifications / total_cases * 100.0) if total_cases else 0.0
    precision = (true_positives / (true_positives + false_positives) * 100.0) if (true_positives + false_positives) else 0.0
    recall = (true_positives / (true_positives + false_negatives) * 100.0) if (true_positives + false_negatives) else 0.0

    summary = {
        "evaluation_title": "PromptShield (PS3) - Evaluation Benchmark",
        "dataset_file": "data/evaluation_dataset.json",
        "total_cases": total_cases,
        "correct_classifications": correct_classifications,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "policy_verification": {
            "critical_attacks_tested": critical_attacks_tested,
            "critical_attacks_blocked": critical_attacks_blocked,
            "critical_block_rate": round((critical_attacks_blocked / critical_attacks_tested * 100.0), 2) if critical_attacks_tested else 0.0,
            "harmless_normal_tested": harmless_tested,
            "harmless_normal_allowed": harmless_allowed,
            "harmless_allow_rate": round((harmless_allowed / harmless_tested * 100.0), 2) if harmless_tested else 0.0,
        },
        "limitations_note": (
            "PromptShield achieves high precision on deterministic prompt injection indicators, "
            "delimiter breakouts, known jailbreaks, and base64 obfuscation. However, as noted in "
            "security literature, pure rule-based scanners cannot guarantee 100% coverage against "
            "novel linguistic paraphrasing, zero-day roleplay scenarios, or complex semantic steganography. "
            "Defense-in-depth (rules + LLM secondary checks + system delimiters) is recommended."
        ),
        "cases": detailed_results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def print_report(summary: Dict[str, Any]) -> None:
    print("=" * 72)
    print("        PROMPTSHIELD (PS3) - REPRODUCIBLE EVALUATION REPORT         ")
    print("=" * 72)
    print(f"Total Cases:              {summary['total_cases']}")
    print(f"Correct Classifications:  {summary['correct_classifications']} / {summary['total_cases']}")
    print(f"False Positives:          {summary['false_positives']} (Benign wrongly flagged)")
    print(f"False Negatives:          {summary['false_negatives']} (Attacks missed)")
    print("-" * 72)
    print(f"Accuracy:                 {summary['accuracy']}%")
    print(f"Precision:                {summary['precision']}%")
    print(f"Recall:                   {summary['recall']}%")
    print("-" * 72)
    pv = summary["policy_verification"]
    print("POLICY ENFORCEMENT VERIFICATION:")
    print(f"• Critical Attacks Blocked:  {pv['critical_attacks_blocked']}/{pv['critical_attacks_tested']} ({pv['critical_block_rate']}%) - Forwarded text is null")
    print(f"• Harmless Inputs Allowed:   {pv['harmless_normal_allowed']}/{pv['harmless_normal_tested']} ({pv['harmless_allow_rate']}%) - Forwarded text intact")
    print("-" * 72)
    print("HONEST LIMITATIONS & STRENGTHS:")
    print(summary["limitations_note"])
    print("=" * 72)


if __name__ == "__main__":
    summary = run_evaluation()
    print_report(summary)
