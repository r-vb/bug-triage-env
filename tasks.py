"""
Tasks and graders for the Bug Triage OpenEnv environment.
Each task has a clear objective and a deterministic grader (score 0.0–1.0).
"""

from environment import BugTriageEnv, Action, Priority, IssueLabel, IssueStatus, GROUND_TRUTH, PRIORITY_ORDER
from typing import Callable


# ─── Task Definitions ─────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "id": "easy",
        "name": "Basic Issue Classification",
        "description": (
            "Triage 3 straightforward GitHub issues: a documentation fix, "
            "a feature request, and a user question. The agent must correctly "
            "label and assign priority to clearly-typed issues with no ambiguity."
        ),
        "difficulty": "easy",
        "n_issues": 3,
        "expected_score_range": [0.7, 1.0],
        "issue_ids": [1003, 1005, 1007],
        "success_criteria": (
            "All 3 issues correctly labeled. Priority within 1 level of ground truth."
        ),
        "graders": [
            {
                "name": "grade_task_easy",
                "type": "python_function",
                "entrypoint": "tasks:grade_task_easy",
                "deterministic": True,
                "score_range": [0.0, 1.0],
            }
        ],
    },
    "medium": {
        "id": "medium",
        "name": "Mixed Backlog Triage",
        "description": (
            "Triage 4 issues including real bugs, a performance regression, "
            "and a duplicate. Agent must identify the duplicate and avoid "
            "over-prioritizing it, while catching the broken email flow."
        ),
        "difficulty": "medium",
        "n_issues": 4,
        "expected_score_range": [0.5, 0.85],
        "issue_ids": [1001, 1004, 1006, 1009],
        "success_criteria": (
            "Duplicate correctly labeled. Email bug at MEDIUM or higher. "
            "Performance issue at HIGH. Assignees within team capacity."
        ),
        "graders": [
            {
                "name": "grade_task_medium",
                "type": "python_function",
                "entrypoint": "tasks:grade_task_medium",
                "deterministic": True,
                "score_range": [0.0, 1.0],
            }
        ],
    },
    "hard": {
        "id": "hard",
        "name": "Security Incident + Overloaded Team",
        "description": (
            "Triage 5 issues including two security vulnerabilities, a critical "
            "memory leak, and a performance bug — with a team at limited capacity. "
            "Agent must correctly escalate security issues and avoid overloading engineers."
        ),
        "difficulty": "hard",
        "n_issues": 5,
        "expected_score_range": [0.3, 0.75],
        "issue_ids": [1002, 1008, 1010, 1001, 1004],
        "success_criteria": (
            "Both security issues marked CRITICAL. Memory leak marked CRITICAL. "
            "No engineer assigned more hours than their capacity. SLA breaches = 0."
        ),
        "graders": [
            {
                "name": "grade_task_hard",
                "type": "python_function",
                "entrypoint": "tasks:grade_task_hard",
                "deterministic": True,
                "score_range": [0.0, 1.0],
            }
        ],
    },
}


# ─── Graders ──────────────────────────────────────────────────────────────────

def _priority_score(predicted: Priority, ground_truth: Priority) -> float:
    if predicted == ground_truth:
        return 1.0
    pi = PRIORITY_ORDER.index(predicted)
    gi = PRIORITY_ORDER.index(ground_truth)
    diff = abs(pi - gi)
    return max(0.0, 1.0 - diff * 0.4)


def _label_score(predicted: IssueLabel, ground_truth: IssueLabel) -> float:
    return 1.0 if predicted == ground_truth else 0.0


def run_agent_on_task(agent_fn: Callable, task_id: str, seed: int = 42) -> dict:
    """
    Run an agent function against a task and return graded results.

    agent_fn: callable(observation: Observation) -> Action
    Returns: {"score": float, "details": dict}
    """
    env = BugTriageEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    results = []

    while True:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        results.append({
            "issue_id": info.triaged_issues[-1]["issue_id"],
            "reward": reward,
            "action": info.triaged_issues[-1]["action"],
        })
        if done:
            break

    final_score = total_reward / max(1, steps)
    return {
        "task_id": task_id,
        "score": round(final_score, 4),
        "steps": steps,
        "sla_breaches": info.sla_breaches,
        "results": results,
    }


def grade_task_easy(agent_fn: Callable, seed: int = 42) -> float:
    """
    Grader for 'easy' task.
    Score = average per-issue reward across 3 clear-cut issues.
    Full score if all labels correct and priorities within 1 level.
    """
    result = run_agent_on_task(agent_fn, "easy", seed)
    return result["score"]


def grade_task_medium(agent_fn: Callable, seed: int = 42) -> float:
    """
    Grader for 'medium' task.
    Adds bonus checks:
    - Duplicate issue correctly labeled as 'duplicate' (+0.1 bonus)
    - Email bug at MEDIUM or higher (penalty if LOW/WONT_FIX)
    """
    result = run_agent_on_task(agent_fn, "medium", seed)
    base_score = result["score"]

    bonus = 0.0
    penalty = 0.0

    for r in result["results"]:
        iid = r["issue_id"]
        action = r["action"]
        if iid == 1009 and action["label"] == IssueLabel.DUPLICATE:
            bonus += 0.05
        if iid == 1006:
            p = Priority(action["priority"])
            if p in [Priority.LOW, Priority.WONT_FIX]:
                penalty += 0.15

    final = base_score + bonus - penalty
    return round(max(0.0, min(1.0, final)), 4)


def grade_task_hard(agent_fn: Callable, seed: int = 42) -> float:
    """
    Grader for 'hard' task.
    Strict: security issues MUST be CRITICAL. SLA breaches heavily penalized.
    """
    result = run_agent_on_task(agent_fn, "hard", seed)
    base_score = result["score"]

    security_ids = {1002, 1010}
    critical_ids = {1002, 1008}
    security_penalty = 0.0
    sla_penalty = result["sla_breaches"] * 0.1

    for r in result["results"]:
        iid = r["issue_id"]
        action = r["action"]
        p = Priority(action["priority"])
        if iid in security_ids and p != Priority.CRITICAL:
            security_penalty += 0.2
        if iid in critical_ids and p not in [Priority.CRITICAL, Priority.HIGH]:
            security_penalty += 0.1

    final = base_score - security_penalty - sla_penalty
    return round(max(0.0, min(1.0, final)), 4)


# ─── Registry ─────────────────────────────────────────────────────────────────

GRADERS = {
    "easy": grade_task_easy,
    "medium": grade_task_medium,
    "hard": grade_task_hard,
}

TASK_LIST = list(TASKS.values())
TASK_COUNT = len(TASKS)
TASKS_WITH_GRADERS = sum(1 for task in TASKS.values() if task.get("graders"))
GRADER_SPECS = {
    task_id: task.get("graders", [])
    for task_id, task in TASKS.items()
}


def grade_all(agent_fn: Callable) -> dict:
    """Run all 3 graders and return combined results."""
    scores = {}
    for task_id, grader in GRADERS.items():
        scores[task_id] = grader(agent_fn)
    overall = sum(scores.values()) / len(scores)
    return {
        "scores": scores,
        "overall": round(overall, 4),
    }
