import json
import os
import time
from typing import List

from openai import OpenAI

from environment import Action, BugTriageEnv, IssueLabel, IssueStatus, Observation, Priority


BENCHMARK = "bug-triage-openenv"
TASK_NAME = "all"
TASK_IDS = ["easy", "medium", "hard"]
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_RUNTIME_SECONDS = 20 * 60


def load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = "None" if error is None else json.dumps(error)
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={json.dumps([round(r, 4) for r in rewards])}",
        flush=True,
    )


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Missing required environment variable: HF_TOKEN")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

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

Output ONLY the JSON object.
"""


def obs_to_prompt(obs: Observation) -> str:
    issue = obs.current_issue
    lines = [
        f"Issue #{issue.id}: {issue.title}",
        f"Author: {issue.author}",
        f"Body: {issue.body}",
        f"Inbox size: {obs.inbox_size}",
        f"Triaged count: {obs.triaged_count}",
        f"Team capacity: {json.dumps(obs.team_capacity)}",
        f"SLA breached count: {obs.sla_breached_count}",
    ]
    if issue.comments:
        lines.append("Comments:\n" + "\n".join(issue.comments))
    if issue.stack_trace:
        lines.append(f"Stack trace:\n{issue.stack_trace}")
    if issue.affected_users_count:
        lines.append(f"Affected users: {issue.affected_users_count}")
    return "\n".join(lines)


def safe_parse_action(raw: str, obs: Observation) -> Action:
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned.strip())
        return Action(
            issue_id=data["issue_id"],
            priority=Priority(data["priority"]),
            label=IssueLabel(data["label"]),
            status=IssueStatus(data["status"]),
            assignee=data.get("assignee"),
            comment=data.get("comment"),
            estimated_fix_hours=data.get("estimated_fix_hours"),
        )
    except Exception:
        return Action(
            issue_id=obs.current_issue.id,
            priority=Priority.MEDIUM,
            label=IssueLabel.BUG,
            status=IssueStatus.OPEN,
            assignee=None,
            comment="Fallback due to parsing error",
            estimated_fix_hours=None,
        )


def get_model_action(obs: Observation, task_id: str, step: int, history: List[str]) -> Action:
    prompt = "\n\n".join(
        [
            f"Task: {task_id}",
            f"Step: {step}",
            obs_to_prompt(obs),
            "Recent history:",
            "\n".join(history[-3:]) if history else "None",
        ]
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    raw = response.choices[0].message.content or ""
    return safe_parse_action(raw, obs)


def summarize_action(task_id: str, action: Action) -> str:
    return (
        f"task={task_id},issue_id={action.issue_id},priority={action.priority.value},"
        f"label={action.label.value},status={action.status.value},"
        f"assignee={action.assignee},estimated_fix_hours={action.estimated_fix_hours}"
    )


def main() -> None:
    history: List[str] = []
    rewards: List[float] = []
    task_scores: dict[str, float] = {}
    steps_taken = 0
    global_step = 0
    success = False
    start_time = time.time()

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        for task_id in TASK_IDS:
            env = BugTriageEnv(task_id=task_id, seed=42)
            task_rewards: List[float] = []
            obs = env.reset()

            try:
                while not env.state()["done"]:
                    if time.time() - start_time > MAX_RUNTIME_SECONDS:
                        raise TimeoutError("Inference exceeded runtime limit")

                    global_step += 1
                    action = get_model_action(obs, task_id, global_step, history)
                    action_summary = summarize_action(task_id, action)

                    next_obs, reward, done, info = env.step(action)
                    reward = reward or 0.0

                    rewards.append(reward)
                    task_rewards.append(reward)
                    steps_taken = global_step

                    log_step(
                        step=global_step,
                        action=action_summary,
                        reward=reward,
                        done=done,
                        error=None,
                    )

                    history.append(
                        f"Task {task_id} step {global_step}: {action_summary} -> reward {reward:.4f}"
                    )

                    obs = next_obs
                    if done:
                        break

                task_score = info.total_reward if task_rewards else 0.0
                task_scores[task_id] = round(max(0.0, min(1.0, task_score)), 4)

            except Exception as exc:
                task_scores[task_id] = 0.0
                log_step(
                    step=global_step + 1,
                    action=f"task={task_id},status=error",
                    reward=0.0,
                    done=True,
                    error=str(exc),
                )

            finally:
                # Keep state isolated across tasks.
                del env

        overall = sum(task_scores.values()) / len(TASK_IDS)
        overall = round(max(0.0, min(1.0, overall)), 4)
        task_scores["overall"] = overall

        with open("baseline_scores.json", "w", encoding="utf-8") as f:
            json.dump(task_scores, f, indent=2)

        success = overall >= SUCCESS_SCORE_THRESHOLD

    finally:
        score = task_scores.get("overall", 0.0)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
