# 🐛 Bug Triage OpenEnv

A real-world OpenEnv environment that simulates **GitHub issue triage** — one of the most common and high-value tasks in software engineering.

An AI agent acts as an engineering manager processing a backlog of GitHub issues. For each issue, it must assign a **priority**, **label**, **status**, and optionally an **engineer** — while managing team capacity and SLA compliance.

---

## Motivation

Issue triage is a task every engineering team does daily. Poor triage leads to:
- Security vulnerabilities sitting unaddressed
- SLA breaches on customer-facing bugs  
- Engineers overloaded while others are idle

This environment trains agents to triage intelligently using real signals: stack traces, user impact counts, reactions, comments, and team capacity.

---

## Environment Description

### Action Space

| Field | Type | Values |
|-------|------|--------|
| `issue_id` | int | ID of the issue |
| `priority` | enum | `critical`, `high`, `medium`, `low`, `wont_fix` |
| `label` | enum | `bug`, `feature`, `performance`, `security`, `documentation`, `question`, `duplicate` |
| `status` | enum | `open`, `in_progress`, `needs_info`, `closed` |
| `assignee` | str \| null | Engineer username |
| `comment` | str \| null | Triage rationale |
| `estimated_fix_hours` | float \| null | Estimated hours to fix |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `current_issue` | object | Full GitHub issue (title, body, comments, reactions, stack trace, affected users) |
| `inbox_size` | int | Issues remaining |
| `triaged_count` | int | Issues already triaged |
| `team_capacity` | dict | Engineer → available hours |
| `sla_breached_count` | int | Critical/high issues under-prioritized so far |
| `step_number` | int | Current step |
| `max_steps` | int | Total steps in episode |

### Reward Function

Per-step reward in `[0.0, 1.0]`:

| Component | Weight | Description |
|-----------|--------|-------------|
| Priority accuracy | 0.50 | Exact match = 1.0, off-by-1 = 0.5, off-by-2 = 0.2 |
| Label accuracy | 0.30 | Correct label = 1.0, else 0.0 |
| Capacity management | 0.10 | Assignee has enough hours |
| Comment quality | 0.10 | Substantive comment = higher score |
| SLA penalty | −0.30 | Applied when critical/high issue is under-prioritized |

---

## Tasks

### Easy — Basic Issue Classification
- **Issues:** 3 (documentation typo, feature request, user question)  
- **Goal:** Correctly label and prioritize clearly-typed issues  
- **Expected score:** 0.70–1.00  
- **Key challenge:** Resist over-prioritizing user questions

### Medium — Mixed Backlog Triage  
- **Issues:** 4 (bugs, performance regression, duplicate)  
- **Goal:** Identify the duplicate, catch the broken email flow at MEDIUM+  
- **Expected score:** 0.50–0.85  
- **Key challenge:** Distinguishing a real bug from a known duplicate

### Hard — Security Incident + Overloaded Team  
- **Issues:** 5 (2 security vulns, memory leak, perf bug, login bug)  
- **Goal:** Escalate both security issues to CRITICAL, manage capacity  
- **Expected score:** 0.30–0.75  
- **Key challenge:** SQL injection and API rate-limit bypass must both be CRITICAL; team is at limited capacity

---

## API Reference

### `POST /reset`
```json
{ "task_id": "easy", "seed": 42 }
```
Returns initial `Observation`.

### `POST /step`
```json
{
  "task_id": "easy",
  "action": {
    "issue_id": 1003,
    "priority": "low",
    "label": "documentation",
    "status": "open",
    "assignee": "alice",
    "comment": "Minor typo in README, low effort fix.",
    "estimated_fix_hours": 0.5
  }
}
```
Returns `{ observation, reward, done, info }`.

### `GET /state?task_id=easy`
Returns full internal state.

### `GET /tasks`
Lists all tasks with descriptions.

---

## Setup & Running

### Local
```bash
pip install -r requirements.txt
python server.py
# Server starts at http://localhost:7860
```

### Docker
```bash
docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

### Run Baseline Inference
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_hf_token

python inference.py
```

---

## Baseline Scores

Measured with `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Router at temperature 0:

| Task | Score |
|------|-------|
| easy | 0.7511 |
| medium | 0.8192 |
| hard | 0.8600 |
| **overall** | **0.8101** |

---

## Project Structure

```
bug-triage-env/
├── environment.py    # Core OpenEnv environment (BugTriageEnv)
├── tasks.py          # Task definitions + graders
├── server.py         # FastAPI HTTP server
├── server/
│   └── app.py        # Entry point for openenv multi-mode deployment
├── inference.py      # Baseline inference script
├── openenv.yaml      # OpenEnv metadata
├── pyproject.toml    # Project metadata and dependencies
├── uv.lock           # Locked dependencies
├── Dockerfile        # Container for HF Spaces
├── requirements.txt
└── README.md
```

---

## Qualification Checks

- ✅ HF Space deploys and responds to `/reset`
- ✅ `openenv.yaml` present with typed models
- ✅ `step()`, `reset()`, `state()` implemented
- ✅ Dockerfile builds and runs
- ✅ 3+ tasks with graders scoring 0.0–1.0
- ✅ `inference.py` in root, uses OpenAI client, reads env vars
- ✅ Graders are deterministic (seed-based)
- ✅ Runtime < 20min, runs on 2vCPU / 8GB

---

*Built for the OpenEnv Hackathon by Team Axiom Minds.*

![Axiom Minds](https://cdn-avatars.huggingface.co/v1/production/uploads/69c433c26aa4fa149f7148fa/cfb8Gv0t2JhKie0Wyb8JY.png)
