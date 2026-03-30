"""
FastAPI server exposing OpenEnv HTTP API.
Endpoints: POST /step, POST /reset, GET /state, GET /tasks, GET /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from environment import BugTriageEnv, Action, Observation, EnvInfo

app = FastAPI(
    title="Bug Triage OpenEnv",
    description="Real-world GitHub issue triage environment for AI agent training.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store (one env per task_id) ──────────────────────────────
_envs: dict[str, BugTriageEnv] = {}


def _get_env(task_id: str) -> BugTriageEnv:
    if task_id not in _envs:
        raise HTTPException(status_code=404, detail=f"No active session for task '{task_id}'. Call /reset first.")
    return _envs[task_id]


# ─── Request/Response schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: EnvInfo


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "bug-triage-openenv", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    """Reset the environment for a given task. Returns initial observation."""
    env = BugTriageEnv(task_id=req.task_id, seed=req.seed)
    _envs[req.task_id] = env
    obs = env.reset()
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """Take one triage action. Returns observation, reward, done, info."""
    env = _get_env(req.task_id)
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state(task_id: str = "easy"):
    """Return the full internal state of the environment (for debugging)."""
    env = _get_env(task_id)
    return env.state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions and difficulty."""
    from tasks import TASKS
    return {"tasks": list(TASKS.values())}


@app.get("/")
def root():
    return {
        "name": "Bug Triage OpenEnv",
        "description": "Real-world GitHub issue triage environment",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
        "spec": "openenv v1",
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
