"""Load τ²-bench results JSON into a small normalized model.

Only Format A (τ² results: {info, tasks[], simulations[]}) is supported for now.
Format B (τ-bench v1 JSONL) has no inline ground truth and comes later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Fallback for tools that never appear in any action_checks of a run.
WRITE_PREFIXES = ("modify_", "cancel_", "return_", "exchange_", "book_", "update_")
AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}


@dataclass
class Action:
    name: str
    args: dict
    compare_args: list | None = None  # None = compare all args


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    msg_idx: int
    error: bool | None = None  # filled from the matching tool message
    result: str = ""


@dataclass
class Message:
    role: str  # assistant | user | tool
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: bool = False
    id: str | None = None


@dataclass
class Sim:
    task_id: str
    trial: int
    reward: float
    termination: str
    messages: list[Message]
    expected: list[Action]
    db_match: bool | None
    nl_failed: list[str]
    tool_types: dict[str, str]  # shared per-run map name -> read|write|generic
    user_script: str = ""       # the user simulator's scenario (lowercased JSON), what it may say

    @property
    def passed(self) -> bool:
        return self.reward >= 1.0

    def tool_type(self, name: str) -> str:
        if name in self.tool_types:
            return self.tool_types[name]
        return "write" if name.startswith(WRITE_PREFIXES) else "read"

    def calls(self) -> list[ToolCall]:
        return [tc for m in self.messages if m.role == "assistant" for tc in m.tool_calls]


@dataclass
class Run:
    path: Path
    model: str
    sims: list[Sim]
    user_model: str = ""  # the LLM playing the customer


def _content(c) -> str:
    if c is None:
        return ""
    return c if isinstance(c, str) else json.dumps(c)


def load_tau2(path: str | Path) -> Run:
    path = Path(path)
    data = json.loads(path.read_text())

    expected_by_task: dict[str, list[Action]] = {}
    script_by_task: dict[str, str] = {}
    for t in data["tasks"]:
        script_by_task[str(t["id"])] = json.dumps(t.get("user_scenario") or {}).lower()
        ec = t.get("evaluation_criteria") or {}
        expected_by_task[str(t["id"])] = [
            Action(a["name"], a.get("arguments") or {}, a.get("compare_args"))
            for a in ec.get("actions") or []
            if a.get("requestor", "assistant") == "assistant"
        ]

    tool_types: dict[str, str] = {}
    for s in data["simulations"]:
        for ac in (s.get("reward_info") or {}).get("action_checks") or []:
            tool_types[ac["action"]["name"]] = ac["tool_type"]

    sims = []
    for s in data["simulations"]:
        msgs: list[Message] = []
        calls_by_id: dict[str, ToolCall] = {}
        for i, m in enumerate(s["messages"]):
            tcs = []
            for tc in m.get("tool_calls") or []:
                args = tc.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                call = ToolCall(tc.get("id") or "", tc["name"], args, i)
                tcs.append(call)
                calls_by_id[call.id] = call
            msg = Message(m["role"], _content(m.get("content")), tcs, bool(m.get("error")), m.get("id"))
            msgs.append(msg)
            if msg.role == "tool" and msg.id in calls_by_id:
                calls_by_id[msg.id].error = msg.error
                calls_by_id[msg.id].result = msg.content

        ri = s.get("reward_info") or {}
        db = ri.get("db_check") or {}
        nl_failed = [a["nl_assertion"] for a in ri.get("nl_assertions") or [] if not a.get("met")]
        nl_failed += [f"communicate: {c['info']}" for c in ri.get("communicate_checks") or [] if not c.get("met")]
        sims.append(Sim(
            task_id=str(s["task_id"]),
            trial=int(s.get("trial") or 0),
            reward=float(ri.get("reward") or 0.0),
            termination=s.get("termination_reason") or "",
            messages=msgs,
            expected=expected_by_task.get(str(s["task_id"]), []),
            db_match=db.get("db_match"),
            nl_failed=nl_failed,
            tool_types=tool_types,
            user_script=script_by_task.get(str(s["task_id"]), ""),
        ))

    info = data.get("info") or {}
    model = (info.get("agent_info") or {}).get("llm") or path.stem
    user_model = (info.get("user_info") or {}).get("llm") or ""
    return Run(path, str(model), sims, str(user_model))
