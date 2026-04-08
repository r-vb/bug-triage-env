"""
inference.py - Sample baseline inference script for Bug Triage OpenEnv.

This version keeps the repo's existing local environment flow, but updates the
runner to a clearer sample-style structure with:
- async main entrypoint
- structured START / STEP / END logs
- safer model-response parsing
- per-task and overall score reporting

Usage:
    $env:API_BASE_URL = "https://api.openai.com/v1"
    $env:MODEL_NAME = "gpt-4o-mini"
    $env:OPENAI_API_KEY = "sk-..."
    python inference.py
"""

import asyncio
import json
import os
import sys

from openai import OpenAI

from environment import Action, BugTriageEnv, IssueLabel, IssueStatus, Observation, Priority
from tasks import TASKS


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("API_KEY")
    or "dummy"
)
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "350"))
TASK_IDS = ["easy", "medium", "hard"]
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.7"))


SYSTEM_PROMPT = """You are an expert engineering manager triaging GitHub issues.

For each issue you receive, output a JSON object with exactly these fields:
{
  "issue_id": <integer>,
  "priority": <"critical"|"high"|"medium"|"low"|"wont_fix">,
  "label": <"bug"|"feature"|"performance"|"security"|"documentation"|"question"|"duplicate">,
  "status": <"open"|"in_progress"|"needs_info"|"closed">,
  "assignee": <string or null>,
  "comment": <string, 1-2 sentences explaining the decision>,
  "estimated_fix_hours": <float or null>
}

Priority guidelines:
- critical: security vulnerabilities, production crashes, data loss
- high: significant bugs affecting many users, performance regressions in production
- medium: bugs affecting some users, broken non-critical features
- low: minor bugs, feature requests, documentation
- wont_fix: out of scope, duplicate of another issue

Available engineers: alice (16h), bob (12h), carol (20h), dave (8h)

Output ONLY the JSON object, with no markdown fences and no extra text."""


def log_start(task: str, model: str) -> None:
    info = TASKS[task]
    print(
        (
            f"[START] task={task} "
            f'task_name="{info["name"]}" '
            f'difficulty={info["difficulty"]} '
            f'model="{model}"'
        ),
        flush=True,
    )


def log_step(step: int, task: str, action: Action, reward: float, done: bool, error: str | None) -> None:
    error_value = json.dumps(error) if error is not None else "null"
    comment_value = json.dumps(action.comment) if action.comment is not None else "null"
    assignee_value = json.dumps(action.assignee) if action.assignee is not None else "null"
    print(
        (
            f"[STEP] task={task} step={step} issue_id={action.issue_id} "
            f"priority={action.priority.value} label={action.label.value} "
            f"status={action.status.value} assignee={assignee_value} "
            f"estimated_fix_hours={action.estimated_fix_hours} "
            f"reward={reward:.4f} done={str(done).lower()} "
            f"comment={comment_value} error={error_value}"
        ),
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, sla_breaches: int) -> None:
    print(
        (
            f"[END] task={task} success={str(success).lower()} "
            f"steps={steps} score={score:.4f} sla_breaches={sla_breaches}"
        ),
        flush=True,
    )


def obs_to_prompt(obs: Observation) -> str:
    issue = obs.current_issue
    lines = [
        f"Issue #{issue.id}: {issue.title}",
        f"Author: {issue.author} {'(first-time contributor)' if issue.is_first_time_contributor else ''}".rstrip(),
        f"Body: {issue.body}",
    ]
    if issue.comments:
        lines.append("Comments:\n" + "\n".join(f"  - {comment}" for comment in issue.comments))
    if issue.stack_trace:
        lines.append(f"Stack trace:\n{issue.stack_trace}")
    if issue.affected_users_count is not None:
        lines.append(f"Affected users: {issue.affected_users_count}")
    if issue.reactions:
        lines.append(f"Reactions: {issue.reactions}")
    lines.append(f"Team capacity: {obs.team_capacity}")
    lines.append(
        f"Inbox remaining: {obs.inbox_size} | Triaged: {obs.triaged_count} | SLA breaches so far: {obs.sla_breached_count}"
    )
    return "\n".join(lines)


def build_fallback_action(obs: Observation) -> Action:
    issue = obs.current_issue
    return Action(
        issue_id=issue.id,
        priority=Priority.MEDIUM,
        label=IssueLabel.BUG,
        status=IssueStatus.OPEN,
        assignee=None,
        comment="Fallback triage decision after model response parsing failed.",
        estimated_fix_hours=None,
    )


def parse_action(raw: str, obs: Observation) -> Action:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    data = json.loads(text.strip())
    data["issue_id"] = obs.current_issue.id
    return Action(
        issue_id=data["issue_id"],
        priority=Priority(data["priority"]),
        label=IssueLabel(data["label"]),
        status=IssueStatus(data["status"]),
        assignee=data.get("assignee"),
        comment=data.get("comment"),
        estimated_fix_hours=data.get("estimated_fix_hours"),
    )


def get_model_action(client: OpenAI, obs: Observation, history: list[str]) -> Action:
    user_prompt = obs_to_prompt(obs)
    if history:
        recent_history = "\n".join(history[-5:])
        user_prompt = f"{user_prompt}\n\nRecent triage history:\n{recent_history}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            return build_fallback_action(obs)
        return parse_action(text, obs)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return build_fallback_action(obs)


async def run_task(client: OpenAI, task_id: str) -> float:
    env = BugTriageEnv(task_id=task_id, seed=42)
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    sla_breaches = 0

    log_start(task=task_id, model=MODEL_NAME)

    try:
        obs = env.reset()

        while True:
            if obs.inbox_size == 0:
                break

            step = steps_taken + 1
            action = await asyncio.to_thread(get_model_action, client, obs, history)
            next_obs, reward, done, info = env.step(action)

            rewards.append(reward or 0.0)
            steps_taken = step
            sla_breaches = info.sla_breaches

            log_step(
                step=step,
                task=task_id,
                action=action,
                reward=reward or 0.0,
                done=done,
                error=None,
            )

            history.append(
                f"Step {step}: issue {action.issue_id} -> "
                f"priority={action.priority.value}, label={action.label.value}, reward={reward or 0.0:+.2f}"
            )

            obs = next_obs
            if done:
                break

        max_total_reward = float(len(rewards)) if rewards else 1.0
        score = sum(rewards) / max_total_reward
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score

    finally:
        log_end(
            task=task_id,
            success=success,
            steps=steps_taken,
            score=score,
            sla_breaches=sla_breaches,
        )


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 60)
    print("Bug Triage OpenEnv - Sample Inference")
    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print("=" * 60)

    all_scores: dict[str, float] = {}

    for task_id in TASK_IDS:
        print(f"\nRunning task: {task_id}", flush=True)
        try:
            all_scores[task_id] = round(await run_task(client, task_id), 4)
            print(f"  Score: {all_scores[task_id]:.4f}", flush=True)
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr, flush=True)
            all_scores[task_id] = 0.0
            print(
                f"[END] task={task_id} success=false steps=0 score=0.0000 error={json.dumps(str(exc))}",
                flush=True,
            )

    overall = sum(all_scores.values()) / len(TASK_IDS)
    all_scores["overall"] = round(overall, 4)

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for key, value in all_scores.items():
        print(f"  {key:10s}: {value:.4f}")
    print("=" * 60)

    with open("baseline_scores.json", "w", encoding="utf-8") as handle:
        json.dump(all_scores, handle, indent=2)
    print("\nScores written to baseline_scores.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
