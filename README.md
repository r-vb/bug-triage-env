---
title: Bug Triage Openenv
emoji: 🐛
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# Bug Triage OpenEnv

Bug Triage OpenEnv is a compact evaluation environment for AI agents that simulates GitHub issue triage. An agent acts like an engineering manager: it sees one issue at a time, chooses a priority, label, status, optional assignee, optional effort estimate, and a short rationale, then receives a reward based on triage quality, SLA handling, and team-capacity awareness.

The repository includes the environment, deterministic task graders, a FastAPI server, a demo landing page, and a sample baseline inference runner.

## What is in this repo

- `environment.py`: core environment, typed models, issue bank, reward logic
- `tasks.py`: three benchmark tasks plus deterministic graders
- `server.py`: FastAPI app and HTTP endpoints
- `server/app.py`: OpenEnv deployment entrypoint
- `inference.py`: sample baseline runner using the OpenAI client
- `index.html`: static demo UI served from `/`
- `openenv.yaml`: OpenEnv metadata and task/schema description
- `baseline_scores.json`: latest recorded baseline results
- `Dockerfile`: container setup for local Docker or Hugging Face Spaces

## Environment overview

Each step presents the agent with a GitHub-style issue containing fields like title, body, comments, reactions, stack trace, reproducibility, and affected-user count. The agent returns a structured action:

```json
{
  "issue_id": 1001,
  "priority": "high",
  "label": "bug",
  "status": "in_progress",
  "assignee": "carol",
  "comment": "This is a user-facing regression affecting many users.",
  "estimated_fix_hours": 6.0
}
```

Reward is clipped to `0.0-1.0` and combines:

- priority accuracy: `0.50`
- label accuracy: `0.30`
- capacity management: `0.10`
- comment quality: `0.10`
- SLA penalty for under-prioritizing critical/high issues: `-0.30`

## Tasks

| Task | Difficulty | Issues | Goal |
|------|------------|--------|------|
| `easy` | easy | 3 | Classify straightforward documentation, feature, and question issues |
| `medium` | medium | 4 | Handle a mixed backlog with bugs, performance regression, and a duplicate |
| `hard` | hard | 5 | Escalate security and production-critical issues while respecting team capacity |

The current issue bank includes examples like:

- login crashes
- SQL injection
- performance regressions
- duplicate issues
- documentation fixes
- support questions
- memory leaks
- broken invitation emails

## API

The FastAPI server exposes:

- `GET /`: serves `index.html` for browsers and returns JSON to JSON clients
- `GET /health`: simple health check
- `GET /metadata`: environment metadata, tasks, and grader registry
- `GET /schema`: JSON schema for action/observation/state
- `GET /tasks`: task list and grader info
- `GET /state?task_id=easy`: current internal environment state
- `GET /baseline-scores`: contents of `baseline_scores.json`
- `POST /reset`: start or restart a task
- `POST /step`: submit one triage action
- `POST /mcp`: placeholder MCP response
- `GET /docs`: Swagger UI from FastAPI

Example reset request:

```json
{
  "task_id": "easy",
  "seed": 42
}
```

Concrete `curl` examples:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d "{\"task_id\":\"easy\",\"seed\":42}"
```

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d "{\"task_id\":\"easy\",\"action\":{\"issue_id\":1003,\"priority\":\"low\",\"label\":\"documentation\",\"status\":\"open\",\"assignee\":null,\"comment\":\"This is a minor documentation issue with low urgency.\",\"estimated_fix_hours\":null}}"
```

## Local setup

### Option 1: pip

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install openenv-core
python server.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install openenv-core
python server.py
```

### Option 2: uv

```bash
uv sync
uv run python server.py
```

The app listens on `http://localhost:7860`.

## Baseline inference

`inference.py` runs all three tasks against a chat model through the OpenAI client, prints structured logs, and writes the final scores to `baseline_scores.json`.

Supported environment variables:

| Variable | Default | Notes |
|----------|---------|-------|
| `API_BASE_URL` | `https://api.openai.com/v1` | Base URL passed to the OpenAI client |
| `MODEL_NAME` | `gpt-4o-mini` | Chat model used for triage |
| `OPENAI_API_KEY` | none | Preferred API key variable |
| `HF_TOKEN` | none | Also accepted as API key input |
| `API_KEY` | none | Generic fallback key variable |
| `TEMPERATURE` | `0.0` | Sampling temperature |
| `MAX_TOKENS` | `350` | Max completion tokens per step |
| `SUCCESS_SCORE_THRESHOLD` | `0.7` | Threshold used for per-task success logging |

Example:

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4o-mini"
$env:HF_TOKEN = "hf-..."
python inference.py
```

The runner emits `[START]`, `[STEP]`, and `[END]` log lines and overwrites `baseline_scores.json` with the latest results.

## Current baseline scores

Latest checked-in scores from `baseline_scores.json`:

| Task | Score |
|------|-------|
| easy | 0.8000 |
| medium | 0.7508 |
| hard | 0.8500 |
| overall | 0.8003 |

## Docker

```bash
docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

The container uses `python:3.11-slim` and starts the service with `python server.py`.

## Validation

This repo includes `pre_validate.sh` for pre-submission checks:

```bash
./pre_validate.sh https://r-vb-bug-triage-env.hf.space
```

It is intended to verify deployment behavior, Docker buildability, and OpenEnv validation before submission.

## Project structure

```text
bug-triage-env/
├── environment.py
├── tasks.py
├── server.py
├── server/
│   └── app.py
├── inference.py
├── index.html
├── openenv.yaml
├── baseline_scores.json
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Notes

- The environment is deterministic for a given `task_id` and `seed`.
- The server keeps in-memory task state and requires `POST /reset` before `POST /step`.
- `GET /baseline-scores` simply returns the checked-in JSON file, so results reflect the most recent local inference run that was committed.

## Qualification checks

| Check | Status | Evidence |
|------|--------|----------|
| OpenEnv metadata | Yes | `openenv.yaml` is present and documents the environment, tasks, schemas, and grader metadata. |
| Environment lifecycle | Yes | `BugTriageEnv` implements `reset()`, `step()`, and `state()`. |
| API endpoints | Yes | The FastAPI app exposes `POST /reset`, `POST /step`, `GET /tasks`, `GET /health`, and related routes. |
| Deterministic graded tasks | Yes | The repository includes three deterministic graded tasks: `easy`, `medium`, and `hard`. |
| Baseline inference runner | Yes | `inference.py` is in the repo root, uses the OpenAI client, reads environment variables, emits structured logs, and writes `baseline_scores.json`. |
| Container support | Yes | The Dockerfile runs the service on port `7860` for containerized local use and Space-style deployment. |
| Pre-submission validation | Yes | `pre_validate.sh` is included for validation workflows before submission. |

*Built for the OpenEnv Hackathon by Team Axiom Minds.*

![Team Axiom Minds](https://cdn-avatars.huggingface.co/v1/production/uploads/69c433c26aa4fa149f7148fa/cfb8Gv0t2JhKie0Wyb8JY.png)
