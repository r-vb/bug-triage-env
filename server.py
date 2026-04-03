from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi import Request
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from environment import BugTriageEnv, Action, Observation, EnvInfo
from tasks import TASKS

app = FastAPI(
    title="Bug Triage OpenEnv",
    version="1.0.1",
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
    return {"status": "ok"}


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
    return {"tasks": list(TASKS.values())}


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
