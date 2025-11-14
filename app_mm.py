# -*- coding: utf-8 -*-
"""
Minimal HTTP + image upload wrapper around your Qwen3 + MCP stack.
- FastAPI backend with a tiny HTML chat UI
- Upload images; the LLM calls MCP tools using JSON arguments (no <tool_call> tags)
- Uses the same JSON-based RoundRunner flow as the benchmark pipeline
Run:  uvicorn app_mm:app --host 0.0.0.0 --port 8000
Env:  MODEL_PATH=/path/to/qwen   CUDA_VISIBLE_DEVICES="0,1,2,3"
"""

import os, json, asyncio, hashlib, textwrap, argparse, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- import your MCP host helper ----
from mcp_host import MCPHost  # assumes same as your current project
from models import create_model_driver
from round_runner import RoundRunner, strip_think

# ---------- CLI arguments ----------
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--MODEL_PATH", required=True)
parser.add_argument("--max_step", type=int, default=4)
parser.add_argument("--max_concurrent", type=int, default=4)
parser.add_argument("--TOP_TOOLS", type=int, default=4)
parser.add_argument("--max_new_tokens", type=int, default=20480)
ARGS, _ = parser.parse_known_args()

# ---------- Model & tokenizer ----------
MODEL_PATH = ARGS.MODEL_PATH
CUDA = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
if CUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

CLI_MAX_STEP = ARGS.max_step
CLI_MAX_CONCURRENT = ARGS.max_concurrent
CLI_TOP_TOOLS = ARGS.TOP_TOOLS
CLI_MAX_NEW_TOKENS = ARGS.max_new_tokens

# Lazy globals initialized on startup
tokenizer: Optional[AutoTokenizer] = None  # kept for backward compat; unused with driver
model: Optional[AutoModelForCausalLM] = None  # kept for backward compat; unused with driver
host: Optional[MCPHost] = None
model_driver = None

# In-memory sessions: session_id -> list[{"role":..., "content":...}]
SESSIONS: Dict[str, List[Dict[str, str]]] = {}

# Directory to hold uploaded images
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


# --------- Helpers ---------

# ------------- Image store -------------
class ImageInfo(BaseModel):
    image_id: str
    file_path: str
    url: str

def _save_image(file: UploadFile) -> Optional[ImageInfo]:
    data = file.file.read()
    # Ignore empty files (no image selected or zero bytes)
    if not data:
        return None
    sha = hashlib.sha256(data).hexdigest()[:16]
    ext = Path(file.filename or "upload").suffix or ".bin"
    image_id = f"{sha}{ext}"
    path = MEDIA_DIR / image_id
    with open(path, "wb") as f:
        f.write(data)
    return ImageInfo(image_id=image_id, file_path=str(path.resolve()), url=f"/media/{image_id}")

# ------------- FastAPI app -------------
app = FastAPI(title="Qwen3 + MCP (with images)")

# serve uploaded files
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR), html=False), name="media")


@app.on_event("startup")
async def _startup():
    global tokenizer, model, host, model_driver
    # Load model via driver (local path or API model name)
    if MODEL_PATH is None:
        raise RuntimeError("Please set MODEL_PATH to local path or API model name")
    model_driver = create_model_driver(MODEL_PATH, max_new_tokens=CLI_MAX_NEW_TOKENS)
    # Start MCP host (reads mcp_servers.json in cwd)
    host = MCPHost(Path("mcp_servers.json"))
    await host.start()


@app.on_event("shutdown")
async def _shutdown():
    global host
    if host:
        await host.stop()
        host = None



# open html
with open("app_mm.html", "r", encoding="utf-8") as f:
    HTML = f.read()



@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


class ToolStep(BaseModel):
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None


class ChatResponse(BaseModel):
    reply: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    images: List[ImageInfo] = []
    steps: List[ToolStep] = []


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    text: str = Form(...),
    session_id: str = Form("default"),
    images: List[UploadFile] = File(default=[])
):
    """
    Accept a user message + optional images.
    The LLM is instructed to pass image_paths (string array) to MCP tools.
    """
    assert host is not None

    # ---- Persist images and collect manifest ----
    uploaded: List[ImageInfo] = []
    for f in images or []:
        try:
            info = _save_image(f)
            if info is not None:
                uploaded.append(info)
        except Exception as e:
            # Only non-empty upload failures should error; empty files have been ignored
            return JSONResponse({"error": f"upload failed: {e}"}, status_code=400)

    # ---- Init session if first time ----
    history = SESSIONS.setdefault(session_id, [])
    # Add the user message (we also append image manifest to the visible content)
    if uploaded:
        text_with_hint = text + "\n\n[附带图片]\n" + "\n".join(
            [f"- {u.image_id}: file={u.file_path} url={u.url}" for u in uploaded]
        )
    else:
        text_with_hint = text
    history.append({"role": "user", "content": text_with_hint})

    # ---- Multi-step with RoundRunner (prepare/work/end) ----
    MAX_STEPS = CLI_MAX_STEP
    TOP_TOOLS = CLI_TOP_TOOLS
    steps: List[ToolStep] = []
    final_reply: Optional[str] = None

    runner = RoundRunner(host=host, model_driver=model_driver, max_step=MAX_STEPS, max_concurrent=CLI_MAX_CONCURRENT, top_tools=TOP_TOOLS)
    uploaded_paths = [u.file_path for u in uploaded]
    rr = await runner.run(history=history, last_user=text_with_hint, uploaded_file_paths=uploaded_paths)
    round_groups: List[List[Dict[str, Any]]] = rr.get("round_groups", [])
    dialogues: List[Dict[str, str]] = rr.get("dialogues", [])

    # Convert to ToolStep list for backward-compatible response fields
    for group in round_groups:
        for entry in group:
            tc = {"name": entry.get("name"), "arguments": entry.get("arguments", {})}
            steps.append(ToolStep(tool_call=tc, tool_result=entry.get("result")))

    # ---- Final summarization pass (always produce a final reply) ----
    SUMMARIZE_SYSTEM = textwrap.dedent("""
        You are finalizing the conversation. Produce ONLY the final answer in natural language.
        Do NOT include any <tool_call> tags or mention tools explicitly. Be concise and accurate,
        relying on the prior tool results contained in the conversation.
    """).strip()
    SUMMARIZE_USER = "Now provide the final answer based on the above tool results. Do not output any <tool_call> blocks."

    summary_messages = history + [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": SUMMARIZE_USER},
    ]

    final_visible, _ = await asyncio.get_event_loop().run_in_executor(
        None, lambda: model_driver.generate_once(summary_messages)
    )
    final_reply = strip_think(final_visible)
    history.append({"role": "assistant", "content": final_reply})

    # Prepare backward-compatible fields (last step if exists)
    last_tool_call: Optional[Dict[str, Any]] = steps[-1].tool_call if steps else None
    last_tool_result: Optional[str] = steps[-1].tool_result if steps else None

    # ---- Logging to ./log/{timestamp}_{session_id}.json ----
    try:
        LOG_DIR = Path("log")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_path = LOG_DIR / f"{ts}_{session_id}.json"
        log_steps: List[Dict[str, Any]] = []
        for idx, group in enumerate(round_groups, start=1):
            log_steps.append({
                "step": idx,
                "calls": group,
            })
        payload = {
            "session_id": session_id,
            "request_text": text,
            "images": [u.dict() for u in uploaded],
            "steps": log_steps,
            "dialogue": dialogues,
            "max_step": CLI_MAX_STEP,
            "max_concurrent": CLI_MAX_CONCURRENT,
            "final_reply": final_reply,
        }
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # logging failure should not crash the API
        pass

    return ChatResponse(
        reply=final_reply,
        tool_call=last_tool_call,
        tool_result=last_tool_result,
        images=uploaded,
        steps=steps,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_mm:app", host="0.0.0.0", port=8000, reload=False)

