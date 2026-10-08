"""Build minimal τ²-shaped results JSON for tests and the CI gate fixtures."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

_ids = itertools.count()

WRITE_TYPES = {"cancel_pending_order": "write", "modify_user_address": "write",
               "find_user_id_by_email": "read", "find_user_id_by_name_zip": "read",
               "get_order_details": "read", "transfer_to_human_agents": "generic"}


def user(text):
    return {"role": "user", "content": text}


def call(name, error=False, result="ok", **args):
    """An assistant tool call followed by its tool result message."""
    cid = f"call-{next(_ids)}"
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": cid, "name": name, "arguments": args, "requestor": "assistant"}]},
        {"role": "tool", "id": cid, "content": result, "error": error, "requestor": "assistant"},
    ]


def say(text):
    return {"role": "assistant", "content": text, "tool_calls": None}


def auth(email="ann@example.com"):
    return [user(f"my email is {email}"), *call("find_user_id_by_email", email=email, result="ann_1")]


def flat(*parts):
    out = []
    for p in parts:
        out.extend(p if isinstance(p, list) else [p])
    return out


def sim(task_id, messages, reward=1.0, db_match=True, trial=0, termination="user_stop",
        nl=None):
    return {
        "task_id": str(task_id), "trial": trial, "termination_reason": termination,
        "messages": messages,
        "reward_info": {
            "reward": reward,
            "db_check": None if db_match is None else {"db_match": db_match},
            "nl_assertions": [{"nl_assertion": a, "met": False} for a in nl or []],
            "action_checks": [{"action": {"name": n}, "tool_type": t} for n, t in WRITE_TYPES.items()],
        },
    }


def task(task_id, actions, script=None):
    """script: the customer's known_info; None = no user scenario at all."""
    scenario = {"instructions": {"known_info": script}} if script is not None else None
    return {"id": str(task_id), "user_scenario": scenario, "evaluation_criteria": {"actions": [
        {"name": n, "arguments": a, "requestor": "assistant", "compare_args": None} for n, a in actions
    ]}}


CANCEL = ("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"})


def good_messages():
    return flat(auth(), user("cancel order W1 please"), say("Confirm cancel #W1?"), user("yes"),
                call("cancel_pending_order", order_id="#W1", reason="no longer needed"))


def bad_messages():
    """Cancels the wrong order -> wrong_args."""
    return flat(auth(), user("cancel order W1 or W2"), say("Confirm cancel #W2?"), user("yes"),
                call("cancel_pending_order", order_id="#W2", reason="no longer needed"))


def results(tasks, sims, model="test"):
    return {"info": {"agent_info": {"llm": model}}, "tasks": tasks, "simulations": sims}


def run_file(path: Path, n_tasks: int, n_pass: int, trials: int = 1, model="test") -> Path:
    """n_tasks cancel tasks; the first n_pass pass on every trial, the rest fail with wrong_args."""
    tasks = [task(i, [CANCEL]) for i in range(n_tasks)]
    sims = [sim(i, good_messages(), trial=t) if i < n_pass
            else sim(i, bad_messages(), reward=0.0, db_match=False, trial=t)
            for i in range(n_tasks) for t in range(trials)]
    path.write_text(json.dumps(results(tasks, sims, model)))
    return path
