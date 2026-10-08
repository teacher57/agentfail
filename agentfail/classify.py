"""Rule-based failure classifier for τ² simulations.

Every sim gets a set of labels (with evidence); failed sims also get one
`primary` label chosen by PRIORITY — the most likely root cause — and a `fault`:
"user" when the simulated customer caused it, else "agent".
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .loader import AUTH_TOOLS, Action, Sim, ToolCall

PRIORITY = [
    "timeout",
    "user_invented_info",  # only primary when the agent acted on it (see classify)
    "fabricated_args",     # only primary when the fabricated value reached a write call
    "wrong_args",
    "unneeded_transfer",   # handed off to a human instead of doing the expected writes
    "missing_action",
    "extra_write",
    "premature_action",  # write before user authentication
    "db_mismatch_only",
    "nl_assertion_fail",
    "unknown",
]
# Recorded as evidence but never primary: too common in passing sims to be a root cause.
SECONDARY = ["unconfirmed_write"]
USER_FAULTS = {"user_invented_info"}

TIMEOUT_REASONS = {"timeout", "max_steps", "too_many_errors"}
# Free-text / enum args that are legitimately not copied from the conversation.
FREE_TEXT_KEYS = {"reason", "summary", "expression", "thought", "explanation"}
TRANSFER_TOOL = "transfer_to_human_agents"
# Identifiers a customer could only know from their script: order ids, zips, emails, user ids.
USER_ID_PATTERN = re.compile(
    r"#?\bW\d{7}\b|\b\d{5}\b|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\b[a-z]+_[a-z]+_\d{4}\b", re.I)
AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|confirm(ed)?|go ahead|proceed|sure|please do|correct|that's right|do it|ok(ay)?)\b",
    re.I,
)


@dataclass
class Finding:
    label: str
    detail: str


@dataclass
class Result:
    sim: Sim
    findings: list[Finding] = field(default_factory=list)
    primary: str | None = None  # None for passing sims

    @property
    def fault(self) -> str | None:
        if self.primary is None:
            return None
        return "user" if self.primary in USER_FAULTS else "agent"

    @property
    def labels(self) -> set[str]:
        return {f.label for f in self.findings}


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        # list order rarely matters for the DB effect (item_ids etc.)
        return sorted((_norm(x) for x in v), key=repr)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _args_match(exp: Action, call: ToolCall) -> bool:
    keys = exp.args.keys() if exp.compare_args is None else exp.compare_args
    return all(_norm(exp.args.get(k)) == _norm(call.args.get(k)) for k in keys)


def _arg_diff(exp: Action, call: ToolCall) -> str:
    keys = exp.args.keys() if exp.compare_args is None else exp.compare_args
    diffs = [f"{k}: {call.args.get(k)!r} != {exp.args.get(k)!r}"
             for k in keys if _norm(exp.args.get(k)) != _norm(call.args.get(k))]
    return "; ".join(diffs)


def _scalars(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _scalars(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _scalars(x)
    elif isinstance(v, str):
        yield v
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        yield str(v)


def _fabricated(sim: Sim) -> list[tuple[ToolCall, str, str]]:
    """Tool-call arg values that never appeared in an earlier user/tool message."""
    out = []
    seen = ""
    for m in sim.messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                for key, val in tc.args.items():
                    if key in FREE_TEXT_KEYS:
                        continue
                    for s in _scalars(val):
                        needle = s.strip().lower().lstrip("#")
                        if len(needle) >= 2 and needle not in seen:
                            out.append((tc, key, s))
        else:
            seen += "\n" + m.content.lower()
    return out


def _user_invented(sim: Sim) -> list[tuple[str, list[ToolCall]]]:
    """Identifiers the customer stated that are neither in their script nor said earlier.

    Returns each value with the tool calls that later used it. Without a script there's
    nothing to check against, so nothing is flagged.
    """
    if sim.user_script in ("", "{}"):
        return []
    out = []
    seen = ""
    for i, m in enumerate(sim.messages):
        if m.role == "user":
            for tok in USER_ID_PATTERN.findall(m.content):
                val = tok.lower().lstrip("#")
                if val not in sim.user_script and val not in seen and val not in (v for v, _ in out):
                    used = [tc for tc in sim.calls() if tc.msg_idx > i
                            and any(val in str(x).lower() for x in _scalars(tc.args))]
                    out.append((val, used))
        seen += "\n" + m.content.lower()
    return out


def _match_writes(sim: Sim, writes: list[ToolCall]):
    # gold trajectories include writes that error in the env, so match against all calls
    exp_writes = [a for a in sim.expected if sim.tool_type(a.name) == "write"]
    unused = list(writes)
    unmatched = []
    for exp in exp_writes:
        hit = next((c for c in unused if c.name == exp.name and _args_match(exp, c)), None)
        if hit:
            unused.remove(hit)
        else:
            unmatched.append(exp)
    wrong, missing = [], []
    for exp in unmatched:
        near = next((c for c in unused if c.name == exp.name), None)
        if near:
            unused.remove(near)
            wrong.append((exp, near))
        else:
            missing.append(exp)
    extra = [c for c in unused if not c.error]  # an errored extra write changes nothing
    return wrong, missing, extra


def classify(sim: Sim) -> Result:
    r = Result(sim)
    add = lambda label, detail: r.findings.append(Finding(label, detail))

    if sim.termination in TIMEOUT_REASONS:
        add("timeout", f"termination_reason={sim.termination}, {len(sim.messages)} msgs")

    writes = [c for c in sim.calls() if sim.tool_type(c.name) == "write"]

    fab = _fabricated(sim)
    fab_in_write = False
    for tc, key, val in fab:
        is_write = sim.tool_type(tc.name) == "write"
        fab_in_write |= is_write
        add("fabricated_args", f"{tc.name}.{key}={val!r} not seen before{' (WRITE)' if is_write else ''}")

    authed_ok = any(c.name in AUTH_TOOLS and c.error is False for c in sim.calls())
    invented = _user_invented(sim)
    user_blocking = False
    for val, used in invented:
        names = sorted({c.name for c in used})
        # the agent acting on the customer's bad info: it got written, or it blocked authentication
        blocking = any(sim.tool_type(c.name) == "write" for c in used) or (bool(used) and not authed_ok)
        user_blocking |= blocking
        add("user_invented_info", f"user said {val!r}, not in their script"
                                  + (f"; agent used it in {', '.join(names)}" if names else "")
                                  + (" (blocked the task)" if blocking else ""))

    transfers = [c for c in sim.calls() if c.name == TRANSFER_TOOL]
    if transfers and not any(a.name == TRANSFER_TOOL for a in sim.expected):
        add("unneeded_transfer", "agent transferred to a human; task expected it to resolve the request")

    wrong, missing, extra = _match_writes(sim, writes)
    for exp, call in wrong:
        add("wrong_args", f"{exp.name}: {_arg_diff(exp, call)}")
    for exp in missing:
        add("missing_action", f"{exp.name}({_short(exp.args)})")
    for call in extra:
        add("extra_write", f"{call.name}({_short(call.args)})")

    # premature: write before successful auth; unconfirmed: no "yes" from the user right before it
    authed = False
    last_user = ""
    for m in sim.messages:
        if m.role == "user":
            last_user = m.content
        wrote_ok = False
        for tc in m.tool_calls:
            if tc.name in AUTH_TOOLS and tc.error is False:
                authed = True
            elif sim.tool_type(tc.name) == "write":
                wrote_ok |= tc.error is False
                if not authed:
                    add("premature_action", f"{tc.name} before user authentication")
                elif not AFFIRMATIVE.search(last_user):
                    add("unconfirmed_write", f"{tc.name} without explicit user confirmation "
                                            f"(last user: {last_user[:60]!r})")
        if wrote_ok:  # one confirmation covers one successful turn of writes; retries keep it
            last_user = ""

    if sim.passed:
        return r

    if sim.db_match is False and not (wrong or missing or extra):
        add("db_mismatch_only", "expected writes all matched but final DB differs")
    if sim.db_match is True:
        for a in sim.nl_failed:
            add("nl_assertion_fail", a[:120])

    labels = r.labels
    if "fabricated_args" in labels and not fab_in_write:
        labels = labels - {"fabricated_args"}
    if "user_invented_info" in labels and not user_blocking:
        labels = labels - {"user_invented_info"}
    if "unneeded_transfer" in labels and not missing:  # the transfer didn't cost any write
        labels = labels - {"unneeded_transfer"}
    r.primary = next((p for p in PRIORITY if p in labels), "unknown")
    return r


def _short(args: dict, n: int = 80) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return s if len(s) <= n else s[: n - 1] + "…"


def summarize(results: list[Result]) -> dict:
    failed = [r for r in results if r.primary]
    return {
        "n": len(results),
        "passed": len(results) - len(failed),
        "primary": Counter(r.primary for r in failed),
        "fault": Counter(r.fault for r in failed),
        "any_failed": Counter(l for r in failed for l in r.labels),
        "any_passed": Counter(l for r in results if not r.primary for l in r.labels),
    }
