"""
Bug Triage OpenEnv Environment
A real-world simulation of GitHub issue triage for AI agent training.
"""

import random
import json
from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    WONT_FIX = "wont_fix"

class IssueLabel(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    QUESTION = "question"
    DUPLICATE = "duplicate"

class IssueStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    NEEDS_INFO = "needs_info"
    CLOSED = "closed"


# ─── Typed Models ─────────────────────────────────────────────────────────────

class GitHubIssue(BaseModel):
    id: int
    title: str
    body: str
    author: str
    comments: list[str] = Field(default_factory=list)
    reactions: dict[str, int] = Field(default_factory=dict)
    created_at: str
    is_first_time_contributor: bool = False
    affected_users_count: Optional[int] = None
    stack_trace: Optional[str] = None
    reproducible: Optional[bool] = None


class Action(BaseModel):
    issue_id: int
    priority: Priority
    label: IssueLabel
    status: IssueStatus
    assignee: Optional[str] = None
    comment: Optional[str] = None
    estimated_fix_hours: Optional[float] = None


class Observation(BaseModel):
    current_issue: GitHubIssue
    inbox_size: int
    triaged_count: int
    team_capacity: dict[str, float]  # engineer -> available hours
    sla_breached_count: int
    step_number: int
    max_steps: int


class Reward(BaseModel):
    value: float
    breakdown: dict[str, float]
    message: str


class EnvInfo(BaseModel):
    done: bool
    total_reward: float
    triaged_issues: list[dict]
    sla_breaches: int


# ─── Issue Bank ───────────────────────────────────────────────────────────────

ISSUE_BANK = [
    {
        "id": 1001,
        "title": "App crashes on login with special characters in password",
        "body": "When I use a password containing `!@#$%` the app throws a 500 error and I cannot log in. This worked before v2.3.0.",
        "author": "user_jane99",
        "comments": [
            "Same issue here, affects our whole team",
            "Workaround: use alphanumeric only passwords for now",
            "This is blocking our onboarding of 200 new users"
        ],
        "reactions": {"👍": 47, "😱": 12},
        "created_at": "2024-01-15T09:23:00Z",
        "is_first_time_contributor": False,
        "affected_users_count": 200,
        "stack_trace": "TypeError: Cannot read properties of undefined (reading 'escape')\n  at AuthService.hashPassword (auth.service.js:45)\n  at UserController.login (user.controller.js:112)",
        "reproducible": True
    },
    {
        "id": 1002,
        "title": "SQL injection possible in search endpoint",
        "body": "The `/api/search?q=` endpoint does not sanitize input. I was able to execute `' OR 1=1--` and get all user records.",
        "author": "security_researcher_42",
        "comments": ["CVE assigned: CVE-2024-XXXX", "Please fix ASAP, this is critical"],
        "reactions": {"😱": 89, "👍": 34},
        "created_at": "2024-01-14T16:00:00Z",
        "is_first_time_contributor": False,
        "affected_users_count": None,
        "stack_trace": None,
        "reproducible": True
    },
    {
        "id": 1003,
        "title": "Typo in README installation instructions",
        "body": "Step 4 says `npm install` but it should be `npm ci` for reproducible builds. Minor but confusing for newcomers.",
        "author": "helpful_hank",
        "comments": [],
        "reactions": {"👍": 3},
        "created_at": "2024-01-16T11:00:00Z",
        "is_first_time_contributor": True,
        "affected_users_count": None,
        "stack_trace": None,
        "reproducible": None
    },
    {
        "id": 1004,
        "title": "Dashboard page takes 45 seconds to load with >1000 records",
        "body": "We have 1500 users and the /dashboard route takes 45+ seconds. Before it was instant. Seems like an N+1 query issue.",
        "author": "perf_watcher",
        "comments": [
            "Confirmed with profiler: 1501 DB queries per page load",
            "Our enterprise clients are complaining",
        ],
        "reactions": {"👍": 23, "🐌": 8},
        "created_at": "2024-01-13T08:30:00Z",
        "is_first_time_contributor": False,
        "affected_users_count": 50,
        "stack_trace": None,
        "reproducible": True
    },
    {
        "id": 1005,
        "title": "Feature request: dark mode support",
        "body": "Would be great to have a dark mode toggle. Many users have requested this. Would help with accessibility too.",
        "author": "night_owl_dev",
        "comments": ["Would love this!", "+1", "High priority for me"],
        "reactions": {"👍": 156, "❤️": 45},
        "created_at": "2024-01-10T14:00:00Z",
        "is_first_time_contributor": False,
        "affected_users_count": None,
        "stack_trace": None,
        "reproducible": None
    },
    {
        "id": 1006,
        "title": "Email notifications not sent when user is invited to team",
        "body": "When adding a new member to a team, they never receive the invitation email. Checked spam folder too.",
        "author": "team_admin_bob",
        "comments": ["Confirmed, affects all new invitations since Jan 12"],
        "reactions": {"👍": 18},
        "created_at": "2024-01-12T10:00:00Z",
        "is_first_time_contributor": False,
        "affected_users_count": 30,
        "stack_trace": "Error: SMTP connection timeout after 30s\n  at Mailer.send (mailer.js:78)",
        "reproducible": True
    },
    {
        "id": 1007,
        "title": "How do I export data to CSV?",
        "body": "I can't find where to export my data to CSV. Is this feature available? I've looked through all the menus.",
        "author": "confused_user_123",
        "comments": ["Check Settings > Data > Export", "It's a bit hidden, I agree"],
        "reactions": {"👍": 5},
        "created_at": "2024-01-16T13:00:00Z",
        "is_first_time_contributor": True,
        "affected_users_count": None,
        "stack_trace": None,
        "reproducible": None
    },
    {
        "id": 1008,
        "title": "Memory leak in WebSocket handler causes server crash after 24h",
        "body": "Production server runs out of memory every ~24 hours. Heap dump shows WebSocket connections are never cleaned up after disconnect.",
        "author": "devops_diana",
        "comments": [
            "Server crashed 3 times this week, costing us ~$2000 in incident response",
            "Heap dump attached",
            "Hotfix needed urgently"
        ],
        "reactions": {"😱": 34, "👍": 28},
        "created_at": "2024-01-11T07:00:00Z",
        "is_first_time_contributor": False,
        "affected_users_count": 500,
        "stack_trace": "FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory",
        "reproducible": True
    },
    {
        "id": 1009,
        "title": "Duplicate of #987 - login button sometimes unresponsive",
        "body": "Sometimes the login button doesn't respond on first click. Need to click twice. Duplicate of issue #987.",
        "author": "another_user",
        "comments": ["This is a known issue, tracked in #987"],
        "reactions": {"👍": 2},
        "created_at": "2024-01-15T15:00:00Z",
        "is_first_time_contributor": True,
        "affected_users_count": None,
        "stack_trace": None,
        "reproducible": None
    },
    {
        "id": 1010,
        "title": "API rate limiting not working — can make unlimited requests",
        "body": "The documented rate limit is 100 req/min but I've been making 10,000 req/min with no throttling. This could be abused.",
        "author": "api_tester_pro",
        "comments": ["Verified. No 429 responses observed", "This is a security concern"],
        "reactions": {"👍": 15, "😱": 22},
        "created_at": "2024-01-14T09:00:00Z",
        "is_first_time_contributor": False,
        "affected_users_count": None,
        "stack_trace": None,
        "reproducible": True
    },
]

TEAM = {
    "alice": 16.0,   # hours available this sprint
    "bob": 12.0,
    "carol": 20.0,
    "dave": 8.0,
}

# Ground truth for grading
GROUND_TRUTH = {
    1001: {"priority": Priority.HIGH, "label": IssueLabel.BUG},
    1002: {"priority": Priority.CRITICAL, "label": IssueLabel.SECURITY},
    1003: {"priority": Priority.LOW, "label": IssueLabel.DOCUMENTATION},
    1004: {"priority": Priority.HIGH, "label": IssueLabel.PERFORMANCE},
    1005: {"priority": Priority.LOW, "label": IssueLabel.FEATURE},
    1006: {"priority": Priority.MEDIUM, "label": IssueLabel.BUG},
    1007: {"priority": Priority.LOW, "label": IssueLabel.QUESTION},
    1008: {"priority": Priority.CRITICAL, "label": IssueLabel.BUG},
    1009: {"priority": Priority.LOW, "label": IssueLabel.DUPLICATE},
    1010: {"priority": Priority.HIGH, "label": IssueLabel.SECURITY},
}

PRIORITY_ORDER = [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW, Priority.WONT_FIX]


# ─── Environment ──────────────────────────────────────────────────────────────

class BugTriageEnv:
    """
    OpenEnv-compliant Bug Triage environment.

    An AI agent acts as an engineering manager triaging a backlog of GitHub issues.
    For each issue, it must assign: priority, label, status, and optionally an assignee.
    Rewards are based on accuracy of triage decisions compared to ground truth.
    """

    def __init__(self, task_id: str = "easy", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._state: dict[str, Any] = {}
        self.reset()

    def reset(self) -> Observation:
        """Reset environment to initial state, return first observation."""
        self._rng = random.Random(self.seed)

        issues = self._get_issues_for_task()
        self._state = {
            "issues": issues,
            "inbox": list(issues),  # copy
            "triaged": [],
            "team_capacity": dict(TEAM),
            "sla_breached": 0,
            "step": 0,
            "max_steps": len(issues),
            "total_reward": 0.0,
            "done": False,
        }
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, EnvInfo]:
        """
        Process one triage action.
        Returns: (observation, reward, done, info)
        """
        if self._state["done"]:
            raise RuntimeError("Episode is done. Call reset() first.")

        issue = self._state["inbox"][0]
        reward_breakdown = {}

        # ── Priority accuracy ──────────────────────────────────────
        gt = GROUND_TRUTH.get(issue["id"])
        if gt:
            priority_score = self._score_priority(action.priority, gt["priority"])
            label_score = 1.0 if action.label == gt["label"] else 0.0
        else:
            priority_score = 0.5
            label_score = 0.5

        reward_breakdown["priority_accuracy"] = priority_score * 0.5
        reward_breakdown["label_accuracy"] = label_score * 0.3

        # ── Assignee capacity check ────────────────────────────────
        capacity_score = 0.0
        if action.assignee and action.assignee in self._state["team_capacity"]:
            hours = self._state["team_capacity"][action.assignee]
            est = action.estimated_fix_hours or 4.0
            if hours >= est:
                capacity_score = 1.0
                self._state["team_capacity"][action.assignee] -= est
            else:
                capacity_score = 0.3  # overloaded engineer
        elif action.assignee is None and action.status == IssueStatus.NEEDS_INFO:
            capacity_score = 0.8  # valid: waiting for info, no need to assign
        reward_breakdown["capacity_management"] = capacity_score * 0.1

        # ── Comment quality ────────────────────────────────────────
        comment_score = 0.0
        if action.comment:
            words = len(action.comment.split())
            comment_score = min(1.0, words / 15)  # reward substantive comments
        reward_breakdown["comment_quality"] = comment_score * 0.1

        # ── SLA penalty for missed critical/high ───────────────────
        sla_penalty = 0.0
        if gt and gt["priority"] in [Priority.CRITICAL, Priority.HIGH]:
            if action.priority not in [Priority.CRITICAL, Priority.HIGH]:
                sla_penalty = -0.3
                self._state["sla_breached"] += 1
        reward_breakdown["sla_penalty"] = sla_penalty

        total_reward = sum(reward_breakdown.values())
        total_reward = max(0.0, min(1.0, total_reward))

        # ── Advance state ──────────────────────────────────────────
        self._state["inbox"].pop(0)
        self._state["triaged"].append({
            "issue_id": issue["id"],
            "action": action.model_dump(),
            "reward": total_reward,
        })
        self._state["step"] += 1
        self._state["total_reward"] += total_reward
        done = len(self._state["inbox"]) == 0
        self._state["done"] = done

        obs = self._make_observation() if not done else self._make_final_observation()

        info = EnvInfo(
            done=done,
            total_reward=self._state["total_reward"] / max(1, self._state["step"]),
            triaged_issues=self._state["triaged"],
            sla_breaches=self._state["sla_breached"],
        )

        return obs, total_reward, done, info

    def state(self) -> dict:
        """Return full current state (for inspection/debugging)."""
        return dict(self._state)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_issues_for_task(self) -> list[dict]:
        if self.task_id == "easy":
            ids = [1003, 1005, 1007]  # documentation, feature, question — clear-cut
        elif self.task_id == "medium":
            ids = [1001, 1004, 1006, 1009]  # bugs + performance + duplicate
        else:  # hard
            ids = [1002, 1008, 1010, 1001, 1004]  # security + critical + ambiguous
        bank = {i["id"]: i for i in ISSUE_BANK}
        return [bank[i] for i in ids if i in bank]

    def _make_observation(self) -> Observation:
        if not self._state["inbox"]:
            return self._make_final_observation()
        issue_data = self._state["inbox"][0]
        return Observation(
            current_issue=GitHubIssue(**issue_data),
            inbox_size=len(self._state["inbox"]),
            triaged_count=self._state["step"],
            team_capacity=self._state["team_capacity"],
            sla_breached_count=self._state["sla_breached"],
            step_number=self._state["step"],
            max_steps=self._state["max_steps"],
        )

    def _make_final_observation(self) -> Observation:
        dummy_issue = GitHubIssue(
            id=0, title="All issues triaged", body="", author="system",
            created_at="2024-01-16T00:00:00Z"
        )
        return Observation(
            current_issue=dummy_issue,
            inbox_size=0,
            triaged_count=self._state["step"],
            team_capacity=self._state["team_capacity"],
            sla_breached_count=self._state["sla_breached"],
            step_number=self._state["step"],
            max_steps=self._state["max_steps"],
        )

    def _score_priority(self, predicted: Priority, ground_truth: Priority) -> float:
        """Partial credit for off-by-one priority errors."""
        if predicted == ground_truth:
            return 1.0
        pi = PRIORITY_ORDER.index(predicted)
        gi = PRIORITY_ORDER.index(ground_truth)
        diff = abs(pi - gi)
        if diff == 1:
            return 0.5
        if diff == 2:
            return 0.2
        return 0.0
