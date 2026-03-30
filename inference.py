"""
inference.py — Baseline inference script for Bug Triage OpenEnv.

Uses the OpenAI client with API_BASE_URL + MODEL_NAME + HF_TOKEN env vars.
Runs the agent against all 3 tasks and produces reproducible scores.

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=hf_...
    python inference.py
"""

import os
import json
import sys
from openai import OpenAI

from environment import BugTriageEnv, Action, Priority, IssueLabel, IssueStatus, Observation
from tasks import GRADERS

# ─── Config ───────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(
    api_key=HF_TOKEN or "dummy",
    base_url=API_BASE_URL,
)

SYSTEM_PROMPT = """You are an expert engineering manager triaging GitHub issues.

For each issue you receive, you must output a JSON object with exactly these fields:
{
  "issue_id": <integer>,
  "priority": <"critical"|"high"|"medium"|"low"|"wont_fix">,
  "label": <"bug"|"feature"|"performance"|"security"|"documentation"|"question"|"duplicate">,
  "status": <"open"|"in_progress"|"needs_info"|"closed">,
  "assignee": <string or null>,
  "comment": <string, 1-2 sentences explaining your decision>,
  "estimated_fix_hours": <float or null>
}

Priority guidelines:
- critical: security vulnerabilities, production crashes, data loss
- high: significant bugs affecting many users, performance regressions in production
- medium: bugs affecting some users, broken non-critical features
- low: minor bugs, feature requests, documentation
- wont_fix: out of scope, duplicate of another issue

Available engineers: alice (16h), bob (12h), carol (20h), dave (8h)

Output ONLY the JSON object, no explanation."""


def obs_to_prompt(obs: Observation) -> str:
    issue = obs.current_issue
    lines = [
        f"Issue #{issue.id}: {issue.title}",
        f"Author: {issue.author} {'(first-time contributor)' if issue.is_first_time_contributor else ''}",
        f"Body: {issue.body}",
    ]
    if issue.comments:
        lines.append("Comments:\n" + "\n".join(f"  - {c}" for c in issue.comments))
    if issue.stack_trace:
        lines.append(f"Stack trace:\n{issue.stack_trace}")
    if issue.affected_users_count:
        lines.append(f"Affected users: {issue.affected_users_count}")
    if issue.reactions:
        lines.append(f"Reactions: {issue.reactions}")
    lines.append(f"\nTeam capacity: {obs.team_capacity}")
    lines.append(f"Inbox remaining: {obs.inbox_size} | SLA breaches so far: {obs.sla_breached_count}")
    return "\n".join(lines)


def llm_agent(obs: Observation) -> Action:
    """Call the LLM and parse its response into an Action."""
    prompt = obs_to_prompt(obs)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)

    return Action(
        issue_id=data["issue_id"],
        priority=Priority(data["priority"]),
        label=IssueLabel(data["label"]),
        status=IssueStatus(data["status"]),
        assignee=data.get("assignee"),
        comment=data.get("comment"),
        estimated_fix_hours=data.get("estimated_fix_hours"),
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Bug Triage OpenEnv — Baseline Inference")
    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print("=" * 60)

    all_scores = {}

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n▶ Running task: {task_id}")
        try:
            grader = GRADERS[task_id]
            score = grader(llm_agent)
            all_scores[task_id] = score
            print(f"  ✓ Score: {score:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_scores[task_id] = 0.0

    overall = sum(all_scores.values()) / len(all_scores)
    all_scores["overall"] = round(overall, 4)

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for k, v in all_scores.items():
        print(f"  {k:10s}: {v:.4f}")
    print("=" * 60)

    # Write scores to file for automated validation
    with open("baseline_scores.json", "w") as f:
        json.dump(all_scores, f, indent=2)
    print("\nScores written to baseline_scores.json")


if __name__ == "__main__":
    main()
