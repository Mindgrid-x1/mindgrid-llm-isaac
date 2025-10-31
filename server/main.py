from __future__ import annotations

import os, json, time, typing as T, re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .validators import validate_plan
import httpx

#                                Config

BACKEND = os.getenv("BACKEND", "MOCK").upper()  # MOCK | OPENAI_COMPAT | LLAMA_CPP
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-noop")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instruct")

LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080")
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "llama-3.1-8b-instruct-q4")

        #                       Types

class ToolFunction(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI tool calling

class ChatMessage(BaseModel):
    role: str
    content: T.Optional[T.Union[str, list]] = None
    tool_calls: T.Optional[list] = None
    function_call: T.Optional[ToolFunction] = None

class ToolDef(BaseModel):
    type: str
    function: dict

class ChatRequest(BaseModel):
    model: str = Field(default=OPENAI_MODEL)
    messages: list[ChatMessage]
    tools: T.Optional[list[ToolDef]] = None
    tool_choice: T.Optional[T.Union[str, dict]] = None
    stream: T.Optional[bool] = False

#                            Helpers

CANON = ("Pick", "Place", "MoveToPose", "PullWithCompliance", "Release")
def prune_and_order_actions(plan: dict, scene: dict, goal: str) -> dict:
    scene_ids = set()
    try:
        scene_ids = {o["id"] for o in scene.get("objects", []) if isinstance(o, dict) and "id" in o}
    except Exception:
        pass

    kept = []
    for a in plan.get("actions", []):
        skill = a.get("skill")
        args = a.get("args", {}) if isinstance(a.get("args"), dict) else {}

        # drop actions missing required args or referring to unknown ids
        if skill == "Pick":
            if "object_id" not in args or args["object_id"] not in scene_ids:
                continue
        elif skill == "Place":
            if ("object_id" not in args or "surface_id" not in args or
                args["object_id"] not in scene_ids or args["surface_id"] not in scene_ids):
                continue
        elif skill == "MoveToPose":
            if "pose" not in args:
                continue
        elif skill == "PullWithCompliance":
            if "drawer_id" not in args or args["drawer_id"] not in scene_ids:
                continue
        elif skill == "Release":
            # optional: allow no args or require object_id if preseznt
            if "object_id" in args and args["object_id"] not in scene_ids:
                continue

        kept.append({
            "skill": skill,
            "args": args,
            "commitPoint": bool(a.get("commitPoint", True)),
            "riskCost": max(0.0, float(a.get("riskCost", 1.0)))
        })

    # order: MoveToPose (prep) -> Pick / PullWithCompliance -> Place -> Release
    order = {"MoveToPose": 0, "Pick": 1, "PullWithCompliance": 1, "Place": 2, "Release": 3}
    kept.sort(key=lambda x: order.get(x["skill"], 5))

    # dedupe adjacent identical actions
    deduped = []
    for a in kept:
        if not deduped or a != deduped[-1]:
            deduped.append(a)

    plan["actions"] = deduped
    return plan

def _strip_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.M)

def extract_first_balanced_json_obj(s: str) -> dict:
    """Grab the first balanced {...} JSON object from a possibly messy string."""
    import json
    if not isinstance(s, str):
        raise ValueError("Not a string")
    s = _strip_fences(s)
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(s[start:i+1])
                    except Exception:
                        start = None
    raise ValueError("No balanced JSON object found")

def extract_json_obj_or_first_from_list(s: str):
    """Return a JSON object if top-level is object or first element of a list."""
    import json
    if not isinstance(s, str):
        raise ValueError("Not a string")
    s = _strip_fences(s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass
    # fallback to first balanced object
    return extract_first_balanced_json_obj(s)

def normalize_plan(plan: dict | str, fallback_goal: str | None = None) -> dict:
    """Coerce common LLM quirks to our schema: actions list, bools, riskCost number, etc."""
    import json
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            plan = {}
    if not isinstance(plan, dict):
        plan = {}

    if "goal" not in plan and fallback_goal:
        plan["goal"] = fallback_goal

    # accept actions/steps/plan
    actions = plan.get("actions") or plan.get("steps") or plan.get("plan") or []
    if isinstance(actions, dict):
        actions = [actions]
    norm = []
    for a in actions:
        if isinstance(a, str):
            parts = a.strip().split()
            skill = parts[0] if parts else "Unknown"
            args = {"object_id": parts[1]} if len(parts) > 1 else {}
            norm.append({"skill": skill, "args": args, "commitPoint": True, "riskCost": 1})
            continue
        if not isinstance(a, dict):
            continue
        skill = a.get("skill") or a.get("action") or a.get("name") or "Unknown"
        args = a.get("args") or a.get("parameters") or {}
        commit = a.get("commitPoint")
        if isinstance(commit, str):
            commit = commit.lower() in ("true", "1", "yes", "y")
        risk = a.get("riskCost", 1)
        try:
            risk = float(risk)
            risk = max(0.0, risk)
        except Exception:
            risk = 1.0
        norm.append({
            "skill": str(skill),
            "args": args if isinstance(args, dict) else {},
            "commitPoint": True if commit is None else bool(commit),
            "riskCost": risk
        })
    plan["actions"] = norm
    # lists
    for k in ("assumptions", "expected_post"):
        v = plan.get(k)
        if v is None:
            plan[k] = []
        elif not isinstance(v, list):
            plan[k] = [str(v)]
        else:
            plan[k] = [str(x) if not isinstance(x, str) else x for x in v]
    return plan
def _is_effective_plan(plan: dict) -> bool:
    if not isinstance(plan, dict):
        return False
    if not plan.get("goal"):
        return False
    if not plan.get("actions"):
        return False
    return True

def remap_actions_to_schema(plan: dict) -> dict:
    """Map free-form skills/args to canonical set."""
    acts = []
    for a in plan.get("actions", []):
        raw = (a.get("skill") or a.get("action") or "").lower()
        args_in = a.get("args") if isinstance(a.get("args"), dict) else {}

        # skill heuristics
        if "open" in raw and "drawer" in raw:
            skill = "PullWithCompliance"
        elif "move" in raw and "pose" in raw:
            skill = "MoveToPose"
        elif "grasp" in raw or "pick" in raw:
            skill = "Pick"
        elif "place" in raw or "put" in raw:
            skill = "Place"
        elif "release" in raw or ("open" in raw and "gripper" in raw):
            skill = "Release"
        else:
            skill = "MoveToPose" if "move" in raw else "Pick"

        # arg mapping
        obj = (args_in.get("object") or args_in.get("whichOne") or
               args_in.get("obj_id") or args_in.get("object_id"))
        surface = (args_in.get("target") or args_in.get("whatOnObjId") or
                   args_in.get("surface_id"))
        pose = (args_in.get("pose") or args_in.get("target_pose") or
                args_in.get("goal_pose"))

        args = {}
        if skill == "Pick":
            if obj: args["object_id"] = obj
        elif skill == "Place":
            if obj: args["object_id"] = obj
            if surface: args["surface_id"] = surface
        elif skill == "MoveToPose":
            if pose: args["pose"] = pose
        elif skill == "PullWithCompliance":
            if obj: args["drawer_id"] = obj
            args.setdefault("distance_cm", 15)
            args.setdefault("max_force_n", 20)
        elif skill == "Release":
            if obj: args["object_id"] = obj

        commit = a.get("commitPoint")
        if isinstance(commit, str):
            commit = commit.lower() in ("true", "1", "yes", "y")
        commit = True if commit is None else bool(commit)

        risk = a.get("riskCost", 1)
        try:
            risk = max(0.0, float(risk))
        except Exception:
            risk = 1.0

        acts.append({"skill": skill, "args": args, "commitPoint": commit, "riskCost": risk})

    plan["actions"] = acts
    return plan

def autofill_expected_post(plan: dict) -> dict:
    """If model forgot expected_post, add simple symbolic checks."""
    if plan.get("expected_post"):
        return plan
    posts = []
    for a in plan.get("actions", []):
        if a["skill"] == "Place" and "object_id" in a["args"] and "surface_id" in a["args"]:
            posts.append(f'on({a["args"]["object_id"]}, {a["args"]["surface_id"]})')
        if a["skill"] == "PullWithCompliance" and "drawer_id" in a["args"]:
            posts.append(f'open({a["args"]["drawer_id"]})')
    plan["expected_post"] = posts
    return plan

def extract_ids_from_goal(goal: str, scene: dict):
    if not isinstance(goal, str) or not isinstance(scene, dict):
        return None, None
    object_ids = [o.get("id") for o in scene.get("objects", []) if isinstance(o, dict)]
    obj = next((oid for oid in object_ids if oid and oid in goal), None)
    surf = next((oid for oid in object_ids if oid and oid in goal and (
        "counter" in oid or "table" in oid or "surface" in oid)), None)
    return obj, surf

def repair_plan_with_goal(plan: dict, goal: str, scene: dict | None):
    """Ensure the plan actually satisfies the goal (example., includes Place step)."""
    if not scene:
        return plan

    obj_id, surf_id = extract_ids_from_goal(goal, scene)
    actions = plan.get("actions", [])

    # Ensure we pick/place when goal says place/put
    if obj_id and surf_id and ("place" in goal.lower() or "put" in goal.lower()):
        has_pick = any(a["skill"] == "Pick" and a["args"].get("object_id") == obj_id for a in actions)
        has_place = any(a["skill"] == "Place" and a["args"].get("object_id") == obj_id and a["args"].get("surface_id") == surf_id for a in actions)

        if not has_pick:
            actions = [{"skill": "Pick", "args": {"object_id": obj_id}, "commitPoint": True, "riskCost": 1.0}] + actions
        if not has_place:
            actions.append({"skill": "Place", "args": {"object_id": obj_id, "surface_id": surf_id}, "commitPoint": True, "riskCost": 1.0})

        plan["actions"] = actions
        exp = plan.get("expected_post", [])
        if f"on({obj_id}, {surf_id})" not in exp:
            exp.append(f"on({obj_id}, {surf_id})")
        plan["expected_post"] = exp

    # Drawer goals -> ensure open(drawer_id) expectation if present
    if "drawer" in goal.lower():
        for a in actions:
            if a["skill"] == "PullWithCompliance" and "drawer_id" in a["args"]:
                did = a["args"]["drawer_id"]
                exp = plan.get("expected_post", [])
                if f"open({did})" not in exp:
                    exp.append(f"open({did})")
                plan["expected_post"] = exp
                break

    return plan

ROBOT_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "robot.plan",
        "description": "Return a safe SkillGraph plan for the goal, using object_ids from scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skill": {"type": "string"},
                            "args": {"type": "object"},
                            "commitPoint": {"type": "boolean"},
                            "riskCost": {"type": "number", "minimum": 0}
                        },
                        "required": ["skill", "args"]
                    }
                },
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "expected_post": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["goal", "actions"]
        }
    }
}

def _extract_scene_and_goal(messages: list[ChatMessage]) -> tuple[dict, str]:
    """Find a user message containing JSON with 'scene' and 'goal'."""
    scene, goal = None, None
    for m in reversed(messages):
        if m.role == "user" and m.content:
            if isinstance(m.content, str):
                txt = m.content
                try:
                    obj = json.loads(txt)
                    scene = obj.get("scene", scene)
                    goal = obj.get("goal", goal)
                    if scene and goal:
                        break
                except Exception:
                    pass
                # fallback markers
                if "<scene" in txt and "</scene>" in txt:
                    s = txt.split("<scene", 1)[1].split(">", 1)[1].split("</scene>", 1)[0]
                    try:
                        scene = json.loads(s.strip())
                    except Exception:
                        pass
                if "<goal>" in txt and "</goal>" in txt:
                    goal = txt.split("<goal>", 1)[1].split("</goal>", 1)[0].strip()
    if not scene or not goal:
        raise HTTPException(400, "Provide a user message with JSON containing 'scene' and 'goal'. See samples/ and scripts/quick_test.py.")
    return scene, goal

# -------------------- MOCK backend --------------------

def mock_plan(scene: dict, goal: str) -> dict:
    """Tiny rule-based plan generator for CI/demos."""
    actions, assumptions, expected = [], [], []

    objects = {o["id"]: o for o in scene.get("objects", [])}
    types = {o["id"]: o.get("type", "") for o in scene.get("objects", [])}

    def find_first(t):
        for oid, tp in types.items():
            if tp == t:
                return oid
        return None

    if "mug" in goal.lower():
        mug, counter = None, None
        for oid, o in objects.items():
            if o.get("type") == "mug":
                mug = oid
            if o.get("type") in ("counter", "table", "surface"):
                counter = oid
        sponge = find_first("sponge")
        if sponge:
            actions.append({"skill": "Clear", "args": {"object_id": sponge, "dest_surface": "sink_tray"}, "commitPoint": False, "riskCost": 1})
        if mug:
            actions.append({"skill": "Pick", "args": {"object_id": mug}, "commitPoint": True, "riskCost": 1})
            if counter:
                actions.append({"skill": "Place", "args": {"object_id": mug, "surface_id": counter}, "commitPoint": True, "riskCost": 1})
        assumptions.append("handle graspable if present")
        if mug and counter:
            expected.append(f"on({mug}, {counter})")

    elif "drawer" in goal.lower() and "open" in goal.lower():
        drawer = find_first("drawer") or "drawer_1"
        actions.append({"skill": "GraspHandle", "args": {"drawer_id": drawer}, "commitPoint": True, "riskCost": 1})
        actions.append({"skill": "PullToWaypoint", "args": {"drawer_id": drawer, "distance_cm": 5}, "commitPoint": False, "riskCost": 1})
        actions.append({"skill": "PullWithCompliance", "args": {"drawer_id": drawer, "distance_cm": 15, "max_force_n": 20}, "commitPoint": True, "riskCost": 2})
        expected.append(f"open({drawer})")

    else:
        first_obj = next((oid for oid, t in types.items() if t not in ("surface", "table", "counter")), None)
        surface = next((oid for oid, t in types.items() if t in ("surface", "table", "counter")), None)
        if first_obj and surface:
            actions.append({"skill": "Pick", "args": {"object_id": first_obj}, "commitPoint": True, "riskCost": 1})
            actions.append({"skill": "Place", "args": {"object_id": first_obj, "surface_id": surface}, "commitPoint": True, "riskCost": 1})
            expected.append(f"on({first_obj}, {surface})")
        else:
            actions.append({"skill": "Pause", "args": {"reason": "insufficient-scene"}, "commitPoint": True, "riskCost": 0})

    return {"goal": goal, "actions": actions, "assumptions": assumptions, "expected_post": expected}

# -------------------- OPENAI_COMPAT backend --------------------

SYSTEM_PROMPT = (open(os.path.join(os.path.dirname(__file__), "..", "prompts", "system_sgfc.md"), "r", encoding="utf-8").read()
                 if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "prompts", "system_sgfc.md"))
                 else "You only call robot.plan.")

async def openai_compat_plan(messages: list[dict]) -> dict:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    is_ollama = ("11434" in OPENAI_BASE_URL) or (os.getenv("OLLAMA_MODE") == "1")
    if is_ollama:
        sys_prompt = SYSTEM_PROMPT + (
            "\nReturn ONLY a JSON object with keys: goal, actions, assumptions, expected_post. "
            "Each action must have skill (string), args (object), commitPoint (boolean), riskCost (number>=0). "
            "Allowed skills: Pick, Place, MoveToPose, PullWithCompliance, Release. "
            "Args keys allowed: object_id, surface_id, pose, drawer_id, distance_cm, max_force_n. "
            "No prose, no code fences, no variables or expressions."
        )
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "system", "content": sys_prompt}] + messages,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
            "keep_alive": "30m"
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code >= 300:
                raise HTTPException(r.status_code, f"Ollama error: {r.text}")
            data = r.json()

        raw = data["choices"][0]["message"]["content"]
        # Drop obviously-bad expected_post content (expressions)
        raw = re.sub(r'"expected_post"\s*:\s*\[[\s\S]*?\]', '"expected_post": []', raw)
        try:
            plan = extract_json_obj_or_first_from_list(raw)
        except Exception as e:
            print("OLLAMA_RAW_CONTENT (unparseable):\n", raw)
            # minimal shell; handler will normalize or fallback
            plan = {"goal": "", "actions": [], "assumptions": [], "expected_post": []}
        return plan

    # Non-Ollama (vLLM/OpenAI-compatible) → use tools
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": [{
            "type": "function",
            "function": ROBOT_PLAN_TOOL["function"]
        }],
        "tool_choice": {"type": "function", "function": {"name": "robot.plan"}},
        "stream": False
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 300:
            raise HTTPException(r.status_code, f"Upstream error: {r.text}")
        data = r.json()
    try:
        tool_call = data["choices"][0]["message"]["tool_calls"][0]
        args_json = tool_call["function"]["arguments"]
        return json.loads(args_json)
    except Exception as e:
        raise HTTPException(500, f"Invalid tool call from upstream: {e}; payload={data}")

# -------------------- LLAMA_CPP backend --------------------

async def llama_cpp_plan(messages: list[dict]) -> dict:
    url = f"{LLAMA_CPP_BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    sys_prompt = SYSTEM_PROMPT + "\nYou must output ONLY JSON as arguments to robot.plan."
    payload = {"model": LLAMA_CPP_MODEL, "messages": [{"role": "system", "content": sys_prompt}] + messages, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 300:
            raise HTTPException(r.status_code, f"llama.cpp upstream error: {r.text}")
        data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
        start = content.find("{"); end = content.rfind("}")
        return json.loads(content[start:end+1])
    except Exception as e:
        raise HTTPException(500, f"Could not parse JSON plan from llama.cpp: {e}; payload={data}")

# -------------------- FastAPI --------------------

app = FastAPI(title="Mindgrid Robot LLM (Starter)")

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    try:
        scene, goal = _extract_scene_and_goal(req.messages)
    except Exception:
        scene, goal = None, None


    messages_dicts = [m.model_dump() if isinstance(m, BaseModel) else m for m in req.messages]


    if BACKEND == "MOCK":
        if not (scene and goal):
            raise HTTPException(400, "MOCK backend requires user message JSON with 'scene' and 'goal'. See samples/ and scripts/quick_test.py.")
        plan = mock_plan(scene, goal)
    elif BACKEND == "OPENAI_COMPAT":
        try:
            plan = await openai_compat_plan(messages_dicts)
        except Exception as e:
            print("UPSTREAM_PARSE_ERROR:", e)
            if scene and goal:
                plan = mock_plan(scene, goal)
            else:
                raise
    elif BACKEND == "LLAMA_CPP":
        plan = await llama_cpp_plan(messages_dicts)
    else:
        raise HTTPException(400, f"Unknown BACKEND={BACKEND}")

    # Normalize → Canonicalize → Repair → Autofill → Prune/Order
    plan = normalize_plan(plan, goal)
    plan = remap_actions_to_schema(plan)
    plan = repair_plan_with_goal(plan, goal or "", scene or {})
    plan = autofill_expected_post(plan)
    plan = prune_and_order_actions(plan, scene or {}, goal or "")

    # If still empty or useless, fallback to MOCK
    if not _is_effective_plan(plan) and scene and goal:
        plan = mock_plan(scene, goal)

    # Validate (and final fallback if you prefer)
    errors = validate_plan(plan)
    if errors:
        print("PLAN_VALIDATION_ERRORS:", errors, "\nPLAN:", plan)
        if scene and goal:
            plan = mock_plan(scene, goal)
            errors = validate_plan(plan)
            if errors:
                raise HTTPException(422, f"Fallback plan invalid: {errors}")
        else:
            raise HTTPException(422, f"Plan failed schema validation: {errors}")

    # Wrap as OpenAI tool call
    tool_call = {
        "id": "call_robot_plan",
        "type": "function",
        "function": {"name": "robot.plan", "arguments": json.dumps(plan, ensure_ascii=False)}
    }
    response = {
        "id": "chatcmpl-mindgrid-" + str(int(time.time())),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "tool_calls": [tool_call]},
            "finish_reason": "tool_calls"
        }]
    }
    return JSONResponse(response)

@app.get("/healthz")
def healthz():
    return {"ok": True, "backend": BACKEND}
