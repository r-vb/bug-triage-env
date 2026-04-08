from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from environment import BugTriageEnv, Action, Observation, EnvInfo
from tasks import (
    TASKS,
    TASK_LIST,
    TASK_COUNT,
    TASKS_WITH_GRADERS,
    GRADER_SPECS,
    SINGLE_GRADER_SPECS,
)

app = FastAPI(
    title="Bug Triage OpenEnv",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: dict[str, BugTriageEnv] = {}


def _get_env(task_id: str) -> BugTriageEnv:
    if task_id not in _envs:
        raise HTTPException(status_code=404, detail="Call /reset first")
    return _envs[task_id]


def _read_baseline_scores() -> dict:
    scores_path = os.path.join(os.path.dirname(__file__), "baseline_scores.json")
    if not os.path.exists(scores_path):
        return {}

    try:
        with open(scores_path, encoding="utf-8") as f:
            import json

            return json.load(f)
    except Exception:
        return {}


def _task_payloads() -> list[dict]:
    payloads = []
    for task in TASK_LIST:
        payload = {
            key: value
            for key, value in task.items()
            if key not in {"grader", "grader_spec"}
        }
        payload["grader"] = task.get("grader_spec")
        payload["has_grader"] = callable(task.get("grader"))
        payloads.append(payload)
    return payloads


def _grader_registry() -> dict[str, list[dict]]:
    return GRADER_SPECS


def _single_grader_registry() -> dict[str, dict | None]:
    return SINGLE_GRADER_SPECS


def _tasks_with_graders() -> int:
    return TASKS_WITH_GRADERS


# ─── Schemas ──────────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    task_id: Optional[str] = "easy"
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: EnvInfo


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def root(request: Request):
    # If validator/tool explicitly asks for JSON
    if request.headers.get("accept") == "application/json":
        return {"status": "ok", "service": "bug-triage-openenv"}

    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())

    return HTMLResponse(content="""
    <html>
      <body style="background:#090c10;color:#3fb950;font-family:monospace;padding:40px">
        <h1>Bug Triage OpenEnv</h1>
        <p>Endpoints: /reset /step /state /tasks /health /docs</p>
      </body>
    </html>
    """)


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "bug-triage-openenv",
        "description": (
            "A real-world OpenEnv environment that simulates GitHub issue triage "
            "with three deterministic graded tasks."
        ),
        "task_count": TASK_COUNT,
        "tasks_with_graders": _tasks_with_graders(),
        "grader_count": _tasks_with_graders(),
        "tasks": _task_payloads(),
        "graders": _grader_registry(),
        "grader_registry": _single_grader_registry(),
    }


@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "issues": {"type": "array"},
                "inbox": {"type": "array"},
                "triaged": {"type": "array"},
                "team_capacity": {"type": "object"},
                "sla_breached": {"type": "integer"},
                "step": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "total_reward": {"type": "number"},
                "done": {"type": "boolean"},
            },
        },
    }


@app.post("/mcp")
async def mcp(_: Request):
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32601,
                "message": "MCP tools are not implemented for this environment.",
            },
        }
    )


@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id = body.get("task_id", "easy")
    seed = body.get("seed", 42)

    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}")

    # Prevent memory growth
    if len(_envs) > 10:
        _envs.clear()

    env = BugTriageEnv(task_id=task_id, seed=seed)
    _envs[task_id] = env

    return env.reset()


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.task_id)

    try:
        obs, reward, done, info = env.step(req.action)

        if not (0.0 <= reward <= 1.0):
            raise ValueError(f"Invalid reward: {reward}")

        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(task_id: str = "easy"):
    env = _get_env(task_id)
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": _task_payloads(),
        "count": TASK_COUNT,
        "task_count": TASK_COUNT,
        "tasks_with_graders": _tasks_with_graders(),
        "grader_count": _tasks_with_graders(),
        "all_have_graders": _tasks_with_graders() == TASK_COUNT,
        "graders": _grader_registry(),
        "grader_registry": _single_grader_registry(),
    }


@app.get("/baseline-scores")
def baseline_scores():
    return {"scores": _read_baseline_scores()}


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
